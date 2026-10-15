from pycocotools.coco import COCO
import os
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from torchvision.datasets import CIFAR10
import shutil
# 6000 done: boat, horse
# need more: airplane, cat, dog
# more train maybe?: 
coco = COCO("catdog/_annotations.coco.json")

def crop(arr, mask, fullMask):
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
    maskCrop = fullMask[newTop:newBottom, newLeft:newRight]
    return arrCrop, maskCrop

def imageProcess(arr, mask, path, fullMask):
    arrCrop, maskCrop = crop(arr, mask, fullMask)
    maskImg = Image.fromarray(np.round(maskCrop).astype(np.uint8))
    newMask = maskImg.resize((32, 32), Image.NEAREST)
    finalMask = np.round(np.array(newMask)).astype(np.uint8)
        
    arrImg = Image.fromarray(arrCrop.astype(np.uint8))
    newImg = arrImg.resize((32, 32), Image.BILINEAR)
    finalImg = np.round(np.array(newImg.convert("RGB"))).astype(np.uint8)

    arr = np.concatenate([finalImg, finalMask[..., None]], axis = -1)

    np.save(path + '.npy', arr)

    #for testing purposes:
    # print(f"{path}:")
    # print(arr)
    # testMask = Image.fromarray(finalMask*255)
    # os.makedirs('verify/dog_catdog', exist_ok=True)
    # newImg.save('verify/' + path + '.png')
    # testMask.save('verify/' + path + 'mask.png')


def readFolder(folderName, cat, dataset):
    os.makedirs(cat + '_' + dataset, exist_ok=True)
    fname_to_id = {img["file_name"]: img["id"] for img in coco.dataset["images"]}
    for filename in os.listdir(folderName):
        if filename.startswith(cat):
            id = fname_to_id[filename]
            annids = coco.getAnnIds(imgIds=id, catIds=coco.getCatIds(catNms=[cat])[0], iscrowd=None)
            anns = coco.loadAnns(annids)
            img = np.array(Image.open(folderName + '/' + filename))
            fullMask = np.zeros((len(img), len(img[0])))

        #combine all masks for the image into one mask
            for entry in anns:
                mask2 = coco.annToMask(entry)
                fullMask = np.maximum(fullMask, mask2)
            j = 0
            for entry in anns:
                mask = coco.annToMask(entry)
                imageProcess(img, np.array(mask), cat + '_' + dataset + '/' + filename.split("_", 1)[0] + '_' + str(j), fullMask)
                j+=1


def main():
    readFolder('catdog/train', 'cat', 'catdog')

if __name__ == "__main__":
    main()