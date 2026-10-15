from pycocotools.coco import COCO
import os
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from torchvision.datasets import CIFAR10

coco = COCO("annotations/instances_train2017.json")

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
        while dim % 32 != 0:
            dim -= 1
    if dim > y:
        dim = y
        while dim % 32 != 0:
            dim -= 1

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

def imageProcess(arr, mask, path, folderName, ):
    arrCrop, maskCrop = crop(arr, mask)
    maskImg = Image.fromarray(np.round(maskCrop).astype(np.uint8))
    newMask = maskImg.resize((32, 32), Image.NEAREST)
    finalMask = np.round(np.array(newMask)).astype(np.uint8)
        
    arrImg = Image.fromarray(arrCrop.astype(np.uint8))
    newImg = arrImg.resize((32, 32), Image.BILINEAR)
    finalImg = np.round(np.array(newImg.convert("RGB"))).astype(np.uint8)

    arr = np.concatenate([finalImg, finalMask[..., None]], axis = -1)

    np.save(path, arr)

    #for testing purposes:
    #print(f"{path}:")
    #print(arr)
    #testMask = Image.fromarray(finalMask*255)
    #os.makedirs('verify/' + folderName, exist_ok=True)
    #newImg.save('verify/' + path[:-3] + 'png')
    #testMask.save('verify/' + path[:-4] + 'mask.png')


def populateFolder(folderName, imgLoad, startAnnIndex, numImgs, catid):
    count = startAnnIndex
    i = 0
    while i < numImgs and count < len(imgLoad):
        img = imgLoad[count]
        url = img["coco_url"]
        r = requests.get(url, timeout=10)
        image = Image.open(BytesIO(r.content))
        ann_ids = coco.getAnnIds(imgIds=img["id"], catIds=[catid], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        
        j = 0
        for ann in anns:
            if i < numImgs:
                mask = coco.annToMask(ann)
                imageProcess(np.array(image), np.array(mask), folderName + "/" + str(img["id"]) + f"_{j}.npy", folderName)
                j+=1
                i+=1
        count+=1
        if i % 100 <= 3:
            print(f"Downloaded image {i}")
            print(f"Annotation index: {count}")
    if count == len(imgLoad):
        print('Reached end of available data')

# please don't give it a negative start index or invalid catName or dumb folderSuffix
def loadCocoCatTrain(catName, startAnnIndex=0, numImgs=0, folderSuffix="Train"):
    catid = coco.getCatIds(catNms=[catName])[0]
    imgIds = coco.getImgIds(catIds=[catid])
    imgLoad = coco.loadImgs(imgIds)
    if numImgs <= 0:
        numImgs = 6000
    os.makedirs(catName+folderSuffix, exist_ok=True)
    populateFolder(catName+folderSuffix, imgLoad, startAnnIndex, numImgs, catid)

def main():
    loadCocoCatTrain("dog", startAnnIndex=3404, numImgs=2000) #include full or indiv? dog face vs legs in background

if __name__ == "__main__":
    main()