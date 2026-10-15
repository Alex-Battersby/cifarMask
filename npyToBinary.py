import os
import numpy as np

def makeBinary(folder, classid):
    os.makedirs(folder + 'Binary', exist_ok=True)
    for filename in os.listdir(folder):
        arr = np.load(folder + '/' + filename)
        red = arr[...,0].astype(np.uint8).reshape(-1)
        green = arr[...,1].astype(np.uint8).reshape(-1)
        blue = arr[...,2].astype(np.uint8).reshape(-1)
        mask = arr[...,3].astype(np.uint8).reshape(-1)
        label = np.uint8(classid)
        with open(folder + 'Binary/' + filename[:-3] + 'bin', 'wb') as f:
            f.write(bytes([label]))
            f.write(red.tobytes())
            f.write(green.tobytes())
            f.write(blue.tobytes())
            f.write(mask.tobytes())

def main():
    makeBinary('coco/birdTrain', 2)
    
if __name__ == '__main__':
    main()