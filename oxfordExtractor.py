from pycocotools.coco import COCO
import os
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from torchvision.datasets import CIFAR10
import shutil
# corrupt: abyssinian 34, egyptian mau 139 145 167 177, beagle 116, chihuahua 121, saint bernard 60, all egyptian mau

def crop(arr, mask):
    x = len(mask[0])
    y = len(mask)
    left = x
    right = 0
    top = y
    bottom = 0
    for i in range(y):
        for j in range(x):
            if (mask[i][j] == 1):
                if (i > bottom):
                    bottom = i
                if (j > right):
                    right = j
                if (i < top):
                    top = i
                if (j < left):
                    left = j
    center = [(bottom + top)/2, (right+left)/2]
    height = bottom - top
    width = right - left
    if (height > width):
        dim = int(height * 1.5)
    else:
        dim = int(width * 1.5)
        
    while dim % 32 != 0:
        dim += 1

    if dim > x:
        dim = x
        # while dim % 32 != 0:
            #     dim -= 1
    if dim > y:
        dim = y
        # while dim % 32 != 0:
            #     dim -= 1

    newTop = int(center[0] - dim/2)
    newBottom = int(center[0] + dim/2)
    while newTop < 0:
        newTop += 1
        newBottom += 1
    while newBottom > y:
        newBottom -= 1
        newTop -= 1
    newLeft = int(center[1] - dim/2)
    newRight = int(center[1] + dim/2)
    while newLeft < 0:
        newLeft += 1
        newRight += 1
    while newRight > x:
        newRight -= 1
        newLeft -= 1
    arrCrop = arr[newTop:newBottom, newLeft:newRight]
    maskCrop = mask[newTop:newBottom, newLeft:newRight]
    return arrCrop, maskCrop

def imageProcess(arr, mask, path):
    arrCrop, maskCrop = crop(arr, mask)
    maskImg = Image.fromarray(np.round(maskCrop).astype(np.uint8))
    newMask = maskImg.resize((32, 32), Image.NEAREST)
    finalMask = np.round(np.array(newMask)).astype(np.uint8)
        
    arrImg = Image.fromarray(arrCrop.astype(np.uint8))
    newImg = arrImg.resize((32, 32), Image.BILINEAR)
    finalImg = np.round(np.array(newImg.convert("RGB"))).astype(np.uint8)

    arr = np.concatenate([finalImg, finalMask[..., None]], axis = -1)

    np.save(path + '.npy', arr)
    np.save('dog' + path[26:] + '.npy', arr)

    #for testing purposes:
    # print(f"{path}:")
    # print(arr)
    # testMask = Image.fromarray(finalMask*255)
    # os.makedirs('verify/dog_oxford', exist_ok=True)
    # newImg.save('verify/' + path[16:] + '.png')
    # testMask.save('verify/' + path[16:] + 'mask.png')


def readFolder(folderName, cat, dataset, breed, numImgs):
    os.makedirs(folderName + '/' + cat + '_' + dataset, exist_ok=True)
    count = 0
    for filename in os.listdir(folderName + '/images'):
        if filename.startswith(breed) and count < numImgs:
            img = np.array(Image.open(folderName + '/images/' + filename))
            mask = np.array(Image.open(folderName + '/annotations/trimaps/' + filename[:-3] + 'png'))
            for i in range(len(mask)):
                for j in range(len(mask[i])):
                    if mask[i][j] == 2:
                        mask[i][j] = 0
                    else:
                        mask[i][j] = 1
            imageProcess(img, mask, 'oxford-iiit-pet/' + cat + '_' + dataset + '/' + filename[:-4])
            count+=1

def main():
    readFolder('oxford-iiit-pet', 'dog', 'oxford', 'yorkshire_terrier', 12)
if __name__ == "__main__":
    main()