import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import numpy as np
import pandas as pd
import scipy.io
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import gc
from Unet import UNet
import torch
import sys
gc.collect()
use_gpu = torch.cuda.is_available()

def thresh(x):
    if x == 0:
        return 0
    else:
        return 1

thresh = np.vectorize(thresh, otypes=[np.float])

def create_dataset(paths, width_in, height_in, width_out, height_out, data_indexes, mat):
    x = []
    y = []
    for path in tqdm(paths):
        mat = scipy.io.loadmat(path)
        img_tensor = mat['images']
        fluid_tensor = mat['manualFluid1']
        img_array = np.transpose(img_tensor, (2, 0 ,1)) / 255
        img_array = resize(img_array, (img_array.shape[0], width_in, height_in))
        fluid_array = np.transpose(fluid_tensor, (2, 0 ,1))
        fluid_array = thresh(fluid_array)
        fluid_array  = resize(fluid_array, (fluid_array .shape[0], width_out, height_out))

        for idx in data_indexes:
            x += [np.expand_dims(img_array[idx], 0)]
            y += [np.expand_dims(fluid_array[idx], 0)]
    return np.array(x), np.array(y)

def get_dataset(width_in, height_in, width_out, height_out):
    input_path = os.path.join('2015_BOE_Chiu')
    subject_path = [os.path.join(input_path, 'Subject_0{}.mat'.format(i)) for i in range(1, 10)] + [os.path.join(input_path, 'Subject_10.mat')]
    #subject_path = [os.path.join(input_path, 'Subject_0{}.mat'.format(i)) for i in range(1, 3)]
    m = len(subject_path)
    data_indexes = [10, 15, 20, 25, 28, 30, 32, 35, 40, 45, 50]
    mat = scipy.io.loadmat(subject_path[0])
    img_tensor = mat['images']
    manual_fluid_tensor_1 = mat['manualFluid1']
    img_array = np.transpose(img_tensor, (2, 0, 1))
    manual_fluid_array = np.transpose(manual_fluid_tensor_1, (2, 0, 1))
    x_train, y_train = create_dataset(subject_path[:m-1], width_in, height_in, width_out, height_out, data_indexes, mat)
    x_val, y_val = create_dataset(subject_path[m-1:], width_in, height_in, width_out, height_out, data_indexes, mat)
    return x_train, y_train,x_val,y_val

def train_step(inputs, labels, optimizer, criterion, unet, width_out, height_out):
    optimizer.zero_grad()
    # forward + backward + optimize
    outputs = unet(inputs)
    # outputs.shape =(batch_size, n_classes, img_cols, img_rows)
    outputs = outputs.permute(0, 2, 3, 1)
    # outputs.shape =(batch_size, img_cols, img_rows, n_classes)
    m = outputs.shape[0]
    outputs = outputs.resize(m*width_out*height_out, 2)
    labels = labels.resize(m*width_out*height_out)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss

def get_val_loss(x_val, y_val, width_out, height_out, unet):
    x_val = torch.from_numpy(x_val).float()
    y_val = torch.from_numpy(y_val).long()
    if use_gpu:
        x_val = x_val.cuda()
        y_val = y_val.cuda()
    m = x_val.shape[0]
    outputs = unet(x_val)
    # outputs.shape =(batch_size, n_classes, img_cols, img_rows) 
    outputs = outputs.permute(0, 2, 3, 1)
    # outputs.shape =(batch_size, img_cols, img_rows, n_classes) 
    outputs = outputs.resize(m*width_out*height_out, 2)
    labels = y_val.resize(m*width_out*height_out)
    loss = F.cross_entropy(outputs, labels)
    return loss.data

def train(unet, batch_size, epochs, epoch_lapse, threshold, learning_rate, criterion, optimizer, x_train, y_train, x_val, y_val, width_out, height_out):
    epoch_iter = np.ceil(x_train.shape[0] / batch_size).astype(int)
    t = trange(epochs, leave=True)
    for _ in t:
        total_loss = 0
        for i in range(epoch_iter):
            batch_train_x = torch.from_numpy(x_train[i * batch_size : (i + 1) * batch_size]).float()
            batch_train_y = torch.from_numpy(y_train[i * batch_size : (i + 1) * batch_size]).long()
            if use_gpu:
                batch_train_x = batch_train_x.cuda()
                batch_train_y = batch_train_y.cuda()
            batch_loss = train_step(batch_train_x , batch_train_y, optimizer, criterion, unet, width_out, height_out)
            total_loss += batch_loss
        if (_+1) % epoch_lapse == 0:
            val_loss = get_val_loss(x_val, y_val, width_out, height_out, unet)
            print("Total loss in epoch %f : %f and validation loss : %f" %(_+1, total_loss, val_loss))
    gc.collect()

def plot_examples(unet, datax, datay, num_examples=3):
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(18,4*num_examples))
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        image_arr = unet(torch.from_numpy(datax[image_indx:image_indx+1]).float().cuda()).squeeze(0).detach().cpu().numpy()
        ax[row_num][0].imshow(np.transpose(datax[image_indx], (1,2,0))[:,:,0])
        ax[row_num][1].imshow(np.transpose(image_arr, (1,2,0))[:,:,0])
        ax[row_num][2].imshow(image_arr.argmax(0))
        ax[row_num][3].imshow(np.transpose(datay[image_indx], (1,2,0))[:,:,0])
    plt.show()

def main():
    width_in = 284
    height_in = 284
    width_out = 196
    height_out = 196
    PATH = './unet.pt'
    x_train, y_train, x_val, y_val = get_dataset(width_in, height_in, width_out, height_out)
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    batch_size = 3
    epochs = 1
    epoch_lapse = 50
    threshold = 0.5
    learning_rate = 0.01
    unet = UNet(in_channel=1,out_channel=2)
    if use_gpu:
        unet = unet.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unet.parameters(), lr = 0.01, momentum=0.99)
    if sys.argv[1] == 'train':
        train(unet, batch_size, epochs, epoch_lapse, threshold, learning_rate, criterion, optimizer, x_train, y_train, x_val, y_val, width_out, height_out)
        pass
    else:
        if use_gpu:
            unet.load_state_dict(torch.load(PATH))
        else:
            unet.load_state_dict(torch.load(PATH, map_location='cpu'))
        print(unet.eval())
    plot_examples(unet, x_train, y_train)
    plot_examples(unet, x_val, y_val)

if __name__ == "__main__":
    main()
    pass