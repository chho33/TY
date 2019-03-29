from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from utils import *

class SignalsAndLabels(Dataset):
    def __init__(self,data):
        self.data = data

    def __len__(self):
        return len(self.data) 

    def __getitem__(self,idx):
        return self.data[idx] 

def get_data_loader(attrs):
                    #batch_size,
                    #random_seed=2019,
                    #shuffle=True,
                    #num_workers=4,
                    #pin_memory=False):
    batch_size = attrs["batch_size"]
    shuffle = attrs["shuffle"]
    random_seed = attrs["random_seed"]
    num_workers = attrs["num_workers"]
    pin_memory = attrs["pin_memory"]
    test_size = attrs["test_size"]
    valid_size = attrs["valid_size"]
    data = list(get_data(attrs))
    data = data[:30]
    train,test = train_test_split(data,test_size=test_size)
    #train,valid = train_test_split(train,test_size=valid_size)
    train = SignalsAndLabels(train)
    test = SignalsAndLabels(test)

    train_transform = transforms.Compose([
            transforms.ToTensor()
    ])
    valid_trainsform = train_transform

    num_train = len(train)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        train, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        train, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader,valid_loader,test_loader
