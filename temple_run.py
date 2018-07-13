from palette_conversion import *
from config import *
from data_loader import *
from skimage import io

a = io.imread('/Volumes/Transcend/Dataset/segmentation_dataset/fujian/colormaps/cangshan_patch_1002.png')

for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        if (tuple(a[i, j, :]) == tuple([160, 255, 100])):
            a[i, j, :] = [160, 255, 110]

io.imsave('/Volumes/Transcend/Dataset/segmentation_dataset/fujian/colormaps/cangshan_patch_1002.png', a)

