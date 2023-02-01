from skimage.transform import resize
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py


def format(path):
    ls_labels = [ 'ADVE', 'Email' , 'Form' , 'Letter' , 'Memo' , 'News' , 'Note' , 'Report', 'Resume' ,'Scientific'  ]

    size = 227
    maxS = 100

    train_x = np.zeros((10*maxS,size,size))
    train_y = np.ones((10*maxS,1))

    count = 0
    for i in range(len(ls_labels)):
        directory_in_str = './'+ls_labels[i]
        print(directory_in_str)
        directory = os.fsencode(directory_in_str)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".tif"):
            if count < maxS:
                I = plt.imread(directory_in_str+'/'+filename)
                img = I[:,:,0]
                img = resize(img, (size, size), anti_aliasing=True)
                train_x[maxS*i+count,:,:] = img
                train_y[maxS*i+count,:] = int((maxS*i+count)/maxS)
                count = count + 1
            else:
                count = 0
            break

    h5_tr_x = h5py.File("train_x.h5", 'w')
    h5_tr_x.create_dataset("train_x", data=np.array(train_x))
    print(h5_tr_x)
    h5_tr_x.close()

    h5_tr_y = h5py.File("train_y.h5", 'w')
    h5_tr_y.create_dataset("train_y", data=np.array(train_y))
    print(h5_tr_y)
    h5_tr_y.close()

    return h5_tr_x, h5_tr_y