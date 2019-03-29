import numpy as np
import os
import json
from copy import deepcopy
import torch
#from sklearn.model_selection import train_test_split

dir_name = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(dir_name,'data')

# 200ms --> one data
window_min = 20
window_size = int(window_min*60*1000/200) 
step_min = 5 
step_size = int(step_min*60*1000/200)
start = 0
end = int(11.5*60*60*1000/200) 

def get_split_list(mat,dims):
    mat_len = mat.shape[0] if isinstance(mat,np.ndarray) else len(mat)
    arrs = []
    for i in range(mat_len):
        row = mat[i]
        for dim in dims[1::][::-1]:
            row = split_list(row,dim) 
        arrs.append(row) 
    return arrs 

def split_list(row,dim):
    while isinstance(row[0],list) or isinstance(row[0],np.ndarray):
        return list(map(lambda x:split_list(list(x),dim), row))
    if not isinstance(row,np.ndarray):
        row = np.array(row)
    return np.split(row,dim)
    # return list of array: [arr,arr,arr,...]

def min_to_size(minute):
    return int(minute*60*1000/200)

def window_slide_cut(data=None,window_size=window_size,step_size=step_size,start=start,end=end,count=True):
    total_steps = end - start
    for i in range(total_steps)[:(-window_size+1):step_size]:
        first = i
        last = i+window_size
        if count:
            score = counter(first,last,data)
            yield (first,last,score)
        else:
            yield (first,last)

# wight score by the share of one element's time period the time point own 
def counter(first,last,data):
    scores = 0 
    anchor = 0
    for i,d in enumerate(data):
        d_start = d[0] 
        d_end = d[1] 
        if first >= d_start:
            if first < d_end:
                score = 1-((first-d_start)/(d_end-d_start))
                score = score * int(np.sign(d[2]))
                scores += score
            else:
                scores += 0
            anchor = i
        elif last < d_end:
            if last >= d_start:
                score = 1-((last-d_start)/(d_end-d_start))
                score = score * int(np.sign(d[2]))
                scores += score
            else:
                scores += 0
            break 
        elif i > anchor:
            scores += int(np.sign(d[2]))
    return scores 

def fillna_to_0(x):
    if not x or x is None:
        return 0
    return x

def fillnas_to_0(data):
    return list(map(lambda x:fillna(x),data))

def fillnas_to_same(data):
    for i,d in enumerate(data):
        if not d or d is None:
            data[i] = data[i-1]
    return data

# transform one record 
def transform_to_raw(window,raws,count=True):
    if count:
        first, last, score = window 
        data = padding(first,last,raws)
        return (data,score)
    else:
        first, last = window 
        data = padding(first,last,raws)
        return data

def padding(first,last,raws):
    data = raws[first:last+1]
    data = np.pad(data,(len(raws[:first]),end-len(raws[:last+1])),'constant',constant_values=(0.0))
    return data

# model metrics
# root mean squared error (rmse) for regression
def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# mean squared error (mse) for regression
def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)

# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    #R-squared = Explained variation / Total variation
    #the higher the R-squared, the better the model fits your data
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

# R^2
def r_square_loss(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - ( 1 - SS_res/(SS_tot + K.epsilon()))

# huber loss
def huber(true, pred, delta):
    loss = np.where(np.abs(true-pred) < delta , 0.5*((true-pred)**2), delta*np.abs(true - pred) - 0.5*(delta**2))
    return np.sum(loss)

# log cosh loss
def logcosh(true, pred):
    loss = np.log(np.cosh(pred - true))
    return np.sum(loss)

# dataset generator
def get_data(attrs): 
    # data_paths, val_split, test_split, window_size, step_size, start, end, num_augment=0
    data_paths = attrs["data_paths"]
    if not isinstance(data_paths,list):
        data_paths = [data_paths]
    
    for data_path in data_paths:
        with open(data_path,"r") as f:
            data = json.load(f)
        groups = data["OnOffPoint"]
        groups = sorted(groups, key=lambda x:x[0])
        # time of third element not equal to 0 or 1 overlap with others.
        end_of_seq = groups[-1][1]
        #groups = list(filter(lambda x:x[2]==1 or x[2]==0,groups))
        groups = remove_false(groups) 
        groups_filled = fill_groups_space(groups,end_of_seq)
        raws = data["CurrentOnRec"]
        raws = fillnas_to_same(raws)
        for d in get_one_data(groups_filled,raws,attrs):
            yield d

        # if augment_times >1:
        for _ in range(attrs["num_augment"]):
            groups,raws = shuffle_data(groups_filled,raws)
            groups_filled = fill_groups_space(groups,end_of_seq)
            for d in get_one_data(groups_filled,raws,attrs):
                yield d

def get_one_data(groups_filled,raws,attrs):
    groups_link_list = groups_link_list_generator(groups_filled)
    end = int(np.ceil(groups_filled[-1][1]/attrs["window_size"])*attrs["window_size"])
    windows = list(window_slide_cut(window_size=attrs['window_size'],step_size=attrs['step_size'],start=attrs['start'],end=end,count=False))
    data = yolo_data_generator(windows,groups_link_list,raws,target=1)
    return data 

def remove_false(groups):
    return list(filter(lambda x:x[2]==1, groups))

def fill_groups_space(groups,end_of_seq,start_of_seq=0,fill_num=-1):
    '''
    fill the gap in original groups
    '''
    #groups = deepcopy(groups)
    groups_filled = []
    while True:
        if len(groups) == 1:
            groups_filled.append(groups.pop())
            break
        queue = groups.pop(0)
        end = groups[0][0]-1
        if end > queue[1]:
            # queue generated for filling the space 
            new_queue = [queue[1]+1,end,fill_num]
            groups_filled.append(queue)
            groups_filled.append(new_queue)
        else:
            groups_filled.append(queue)
    if groups_filled[0][0] > 0:
        groups_filled.insert(0,[start_of_seq,groups_filled[0][0]-1,0])
    if groups_filled[-1][1] < end_of_seq:
        groups_filled.append([groups_filled[-1][1]+1,end_of_seq,0])
    return groups_filled

def expand_indices(groups_filled):
    '''
    [[0,200,1],[301,501,0],...] --> [0,1,...,200,301,...,501,...]
    '''
    indices = []
    for g in groups_filled:
        indices+=range(g[0],g[1]+1)
    return indices

def shuffle_data(groups_filled,raws,fill_num=-1):
    np.random.shuffle(groups_filled)
    #raws = [raws[i] for i in indices]
    indices = expand_indices(groups_filled)
    raws = list(map(lambda i:raws[i], indices))
    last_tmp = -1 
    groups = [] 
    for g in groups_filled:
        last = len(range(g[1]-g[0]))+last_tmp+1
        group = [last_tmp+1,last,g[-1]]
        last_tmp = last
        groups.append(group)
    groups = list(filter(lambda x:x[2]!=fill_num,groups))
    return groups,raws

def groups_link_list_generator(groups_filled):
    link_list = []
    for group in groups_filled:
        g_start,g_end,category = group 
        category = int(category)
        time_sequence = list(range(g_start,g_end+1))
        for t in time_sequence:
            # position_type: which position type current time index belong to ==> 0:start; 1:middle; -1:end;  
            position_type = 0 if t == g_start else -1 if t == g_end else 1 
            # (position_type, end of group, category)
            data = (position_type,g_end,category)
            link_list.append(data)
    return link_list

def find_target_signals(window,groups_link_list,target=1):
    # window is closed interval, last is included
    window_start,window_end = window
    exists = []
    try:
        end_attr = groups_link_list[window_end] 
    except IndexError:
        end_attr = groups_link_list[-1] 
    end_last_index = end_attr[1]
    while True:
        try:
            start_attr = groups_link_list[window_start] 
        except IndexError:
            # outside of all data
            break
        pos_type,last_index,cat = start_attr
        if last_index > end_last_index: 
            break
        # means this window block include a complete group data and it corresponds to a "right" signal
        if pos_type == 0 and cat == target:
            # record start and end time
            exists.append([window_start,last_index])
            window_start = last_index+1
        else:
            window_start = last_index+1
    return exists 

def yolo_image_transformer(window,raws):
    img = raws[window[0]:window[1]]
    if len(img) < window_size:
        img = np.pad(img,(0,window_size-len(img)),'constant',constant_values=(0.0))
    img = np.array(img).reshape(-1,1,1)
    return img

def yolo_label_transformer(label,window,target=1):
    '''
    example:
    window:  (186000, 187500)
    labels:  [[186045, 186183], [186184, 186326], [186327, 186491], [186492, 186572], [186573, 186960], [186961, 187067], [187068, 187276]]
    '''
    window_start,window_end = window
    label_start,label_end = label
    start = label_start-window_start 
    end = label_end-window_start 
    width = end-start+1
    w = width/2
    h = 0.5
    x = start+w
    y = 0.5 
    #return [x,y,w,h,group[-1]]
    return [target,x,y,w,h]

def yolo_data_generator(windows,groups_link_list,raws,target=1):
    window_size = windows[0][1] - windows[0][0]
    for window in windows:
        # window == (first,last) 
        labels = find_target_signals(window,groups_link_list,target=target)
        #labels = list(map(lambda d:yolo_label_transformer(d,window,target),labels))
        labels = np.array(list(map(lambda d:yolo_label_transformer(d,window,target),labels)))
        labels = torch.from_numpy(labels)
        labels = labels.type(torch.FloatTensor)
        #print('labels: ',labels.shape)
        #labels = list(labels)
        #print('labels: ',len(labels),labels)

        #labels = np.array(labels).reshape(-1,5).tolist()
        if len(labels) == 0: continue
        img = raws[window[0]:window[1]]
        if len(img) < window_size:
            img = np.pad(img,(0,window_size-len(img)),'constant',constant_values=(0.0))
        img = np.array(img).reshape(-1,1,1)
        img = img.transpose(2, 0, 1)
        #img = np.pad(np.array(img).reshape(-1,1),[(0,0),(0,window_size-1)],'constant',constant_values=(0.0))
        #img = np.expand_dims(img,axis=2)
        img = torch.from_numpy(img)
        img = img.type(torch.FloatTensor)
        #print('img: ',img.shape)
        #labels = np.array(labels).reshape(-1,5).tolist()
        # x is an inmage while y includes multiple labels
        yield (img,labels)
