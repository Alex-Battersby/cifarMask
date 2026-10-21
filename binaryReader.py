import os
import numpy as np
from PIL import Image
import shutil

def readImg(folder, fileList):
    for filename in fileList:
        path = os.path.join(folder, filename)
        with open(path, 'rb') as f:
            data = f.read()

        os.makedirs('test', exist_ok=True)
        label = data[0]
        red = np.frombuffer(data[1:1025], dtype=np.uint8).reshape(32,32)
        green = np.frombuffer(data[1025:2049], dtype=np.uint8).reshape(32,32)
        blue = np.frombuffer(data[2049:3073], dtype=np.uint8).reshape(32,32)
        mask = np.frombuffer(data[3073:4097], dtype=np.uint8).reshape(32,32)
        img_arr = np.stack((red,green,blue), axis=2)
        img = Image.fromarray(img_arr)
        img.save(f'test/class{label}_{filename[:-3]}.png')
        mask_arr = mask*255
        maskImg = Image.fromarray(mask_arr)
        maskImg.save(f'test/class{label}_{filename[:-4]}mask.png')

def make_zip(path, class_name):
    shutil.make_archive(os.path.join('zips', class_name), 'zip', root_dir=path)

def clear_imgMask():
    make_zip('binary/automobileBinary', 'automobile')

def main():
    clear_imgMask()

if __name__ == '__main__':
    main()