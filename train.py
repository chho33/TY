from datasets import get_data_loader
import valid
from yolo_utils import torch_utils
from yolo_utils.utils import model_info
from models import Darknet
import torch
import torch.nn.functional as F
import click
#from argparse import ArgumentParser
import time
from collections import defaultdict
import os
dirname = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(dirname,"data")

@click.command()
@click.option("-d","--data",help="data files, seperated by comma")
@click.option("-w","--window_min",default=2,help="window minutes")
@click.option("-s","--step_min",default=2,help="step minutes")
@click.option("-a","--num_augment",default=10,help="how many times want to shuffle data to augment")
@click.option("-u","--shuffle", default=True,help="shuffle for every epoch")
@click.option("-b","--batch_size", default=32,help="batch size")
@click.option("-r","--random_seed", default=2019,help="random seed")
@click.option("-n","--num_workers", default=4,help="number of workers")
@click.option("-m","--pin_memory", default=False,help="pin memory")
@click.option("-t","--test_size", default=0.05,help="test size")
@click.option("-v","--valid_size", default=0.1,help="validation size")
@click.option('--epochs', type=int, default=100, help='number of epochs')
@click.option('--accumulated-batches', type=int, default=1, help='number of batches before optimizer step')
@click.option('--cfg', type=str, default='cfg/yolov3-tiny.cfg', help='cfg file path')
@click.option('--weights', type=str, default='weights', help='path to store weights')
@click.option('-r','--resume', default=True, type=bool, help='resume training flag')
@click.option('--var', type=float, default=0, help='test variable')
def main(**attrs):
    window_min = attrs["window_min"]
    window_size = int(window_min*60*1000/200)
    step_min = attrs["step_min"]
    step_size = int(step_min*60*1000/200)
    attrs["window_size"] = window_size
    attrs["step_size"] = step_size
    attrs["start"] = 0 
    data_paths = attrs["data"].split(',')
    data_paths = list(map(lambda x:os.path.join(data_dir,x),data_paths))
    attrs["data_paths"]=data_paths
    attrs.pop("window_min",None)
    attrs.pop("step_min",None)
    attrs.pop("data",None)
    print(attrs)
    train_loader,val_loader,_ = get_data_loader(attrs)
    #return train_loader,val_loader,test_loader
    img_size = window_size 
    train(
        attrs["cfg"],
        #opt.data_cfg,
        train_loader,
        val_loader,
        img_size=img_size,
        resume=attrs["resume"],
        epochs=attrs["epochs"],
        batch_size=attrs["batch_size"],
        accumulated_batches=attrs["accumulated_batches"],
        weights=attrs["weights"],
        #multi_scale=opt.multi_scale,
        #freeze_backbone=opt.freeze,
        var=attrs["var"],
    )

def train(
        cfg,
        #data_cfg,
        train_loader,
        val_loader,
        img_size=600,
        resume=False,
        epochs=100,
        batch_size=16,
        accumulated_batches=1,
        weights='weights',
        #multi_scale=False,
        #freeze_backbone=True,
        var=0,
):
    device = torch_utils.select_device()

    #if multi_scale:  # pass maximum multi_scale size
    #    img_size = 608
    #else:
    #    torch.backends.cudnn.benchmark = True  # unsuitable for multiscale
    torch.backends.cudnn.benchmark = True

    latest = os.path.join(weights, 'latest.pt')
    best = os.path.join(weights, 'best.pt')

    # Configure run
    #train_path = parse_data_cfg(data_cfg)['train']

    # Initialize model
    model = Darknet(cfg, img_size)

    # Get train_loader 
    #train_loader= LoadImagesAndLabels(train_path, batch_size, img_size, multi_scale=multi_scale, augment=True)

    lr0 = 0.001
    if resume:
        checkpoint = torch.load(latest, map_location='cpu')

        # Load weights to resume from
        model.load_state_dict(checkpoint['model'])

        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
        model.to(device).train()

        # # Transfer learning (train only YOLO layers)
        # for i, (name, p) in enumerate(model.named_parameters()):
        #     if p.shape[0] != 650:  # not YOLO layer
        #         p.requires_grad = False

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=.9)

        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_loss = checkpoint['best_loss']

        del checkpoint  # current, saved

    else:
        start_epoch = 0
        best_loss = float('inf')

        # Initialize model with darknet53 weights (optional)
        #load_darknet_weights(model, os.path.join(weights, 'darknet53.conv.74'))

        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
        model.to(device).train()

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=.9)

    # Set scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[54, 61], gamma=0.1)

    model_info(model)
    t0 = time.time()
    for epoch in range(epochs):
        epoch += start_epoch

        print(('%8s%12s' + '%10s' * 7) % (
            'Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))

        # Update scheduler (automatic)
        # scheduler.step()

        # Update scheduler (manual)  at 0, 54, 61 epochs to 1e-3, 1e-4, 1e-5
        if epoch > 50:
            lr = lr0 / 10
        else:
            lr = lr0
        for g in optimizer.param_groups:
            g['lr'] = lr

        # Freeze darknet53.conv.74 for first epoch
        #if freeze_backbone:
        #    if epoch == 0:
        #        for i, (name, p) in enumerate(model.named_parameters()):
        #            if int(name.split('.')[1]) < 75:  # if layer < 75
        #                p.requires_grad = False
        #    elif epoch == 1:
        #        for i, (name, p) in enumerate(model.named_parameters()):
        #            if int(name.split('.')[1]) < 75:  # if layer < 75
        #                p.requires_grad = True

        ui = -1
        rloss = defaultdict(float)  # running loss
        optimizer.zero_grad()
        for i, (imgs, targets) in enumerate(train_loader):
            #print('imgs_shape: ',imgs.shape)
            p3d = (0,img_size-1+2,0,0+2,0,0,0,0)
            imgs = F.pad(imgs,p3d, mode='constant') 
            #print('imgs_shape: ',imgs.shape)
            if sum([len(x) for x in targets]) < 1:  # if no targets continue
                continue

            # SGD burn-in
            if (epoch == 0) & (i <= 1000):
                lr = lr0 * (i / 1000) ** 4
                for g in optimizer.param_groups:
                    g['lr'] = lr

            # Compute loss, compute gradient, update parameters
            loss = model(imgs.to(device), targets, var=var)
            loss.backward()

            # accumulate gradient for x batches before optimizing
            if ((i + 1) % accumulated_batches == 0) or (i == len(train_loader) - 1):
                optimizer.step()
                optimizer.zero_grad()

            # Running epoch-means of tracked metrics
            ui += 1
            for key, val in model.losses.items():
                rloss[key] = (rloss[key] * ui + val) / (ui + 1)

            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, len(train_loader) - 1), rloss['xy'],
                rloss['wh'], rloss['conf'], rloss['cls'],
                rloss['loss'], model.losses['nT'], time.time() - t0)
            t0 = time.time()
            print(s)

        # Update best loss
        loss_per_target = rloss['loss'] / rloss['nT']
        if loss_per_target < best_loss:
            best_loss = loss_per_target

        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'best_loss': best_loss,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, latest)

        # Save best checkpoint
        if best_loss == loss_per_target:
            os.system('cp ' + latest + ' ' + best)

        # Save backup weights every 5 epochs
        if (epoch > 0) & (epoch % 5 == 0):
            os.system('cp ' + latest + ' ' + os.path.join(weights, 'backup{}.pt'.format(epoch)))

        # Calculate mAP
        with torch.no_grad():
            mAP, R, P = valid.valid(cfg, val_loader, weights=latest, batch_size=batch_size, img_size=img_size)

        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '%11.3g' * 3 % (mAP, P, R) + '\n')


if __name__ == '__main__':
    main()
