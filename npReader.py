import os
import numpy as np
from PIL import Image
import shutil

#sanity checks that the mask has both black and white. the image shouldn't be only mask or no mask
def check_unique(folder):
    for filename in os.listdir(folder):
        arr = np.load(folder + '/' + filename)
        if len(np.unique(arr[..., 3])) < 2:
            print(filename)

def delete_img(path):
    os.remove(path)

def load_imgMask(fileList, folder):
    os.makedirs('test', exist_ok=True)
    for filename in fileList:
        arr = np.load(folder + '/' + filename)
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                if arr[i][j][3] == 1:
                    arr[i][j][3] = 255
        mask = Image.fromarray(arr[...,3])
        mask.save(f'test/{filename}mask.png')
        img = Image.fromarray(arr[..., :3])
        img.save(f'test/{filename}.png')

def clear_imgMask():
    shutil.rmtree('test')

def main():
    check_unique('coco/automobileTrain')

if __name__ == "__main__":
    main()