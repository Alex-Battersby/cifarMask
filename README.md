# cifarMask
## [WIP]

For use, you must download COCO annotations and put them into a folder named coco in the same directory as the dataloader script

## For use

### Overview

This project downloads images from any MS-COCO class or Oxford-IIIT Pet class, then formats them to fit CIFAR-10 with an extra channel beyond RGB representing human-verified instance segmentation masks. It is modular and first outputs data as .npy files, after which the user can choose whether you want to convert them to .bin. The user also may specify the number of images they want, whether they would like to include background instances of objects of interest in their masks, and whether they would like to accept multiple instances cropped from the same base image.

[random image plus mask in coco, followed by my dataset]

### Prerequisites
The user must download ONLY the annotations folder for COCO 2017. Specifically, `coco/annotations/instances_train2017.json` and `coco/annotations/instances_val2017.json` are required to already be downloaded and in the parent folder of the project to run `cocoLoadFull.py` as is.

THE USER DOES NOT NEED TO DOWNLOAD ALL OF COCO! This program automatically loads images during runtime and processes them without downloading first. This avoids unnece

### Image format

`cocoLoadFull.py` processes an input image and annotation, then crops down to a square around an instance of interest such that a bounding box around the instance would take up 62.5% of the square, unless the instance is too large for that, in which case it is simply centered in the square frame and remains the same size. After centering and adjusting instance size to greater than or equal to 62.5% of the square frame, the image and mask are downsampled to 32x32 pixels, at which point they are converted to a single .npy array with four channels per pixel (R, G, B, mask). These .npy arrays are saved automatically and can be converted to .bin with `npyToBinary.py` and deleted, or used immediately. Each file is named after its image ID in COCO/IIIT so the original image can be easily referenced at their respective websites for troubleshooting, and the COCO images include an underscore followed by the highlighted instance's index within its original image; if there are two instances of `cat` within a single COCO image and the user opts to include multiple instances per image, they will be named `[imgID]_0` and `[imgID]_1`.

Following the same conventions, if `npyToBinary.py` is utilized, 