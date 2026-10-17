# cifarMask
## [WIP]

I apologize for the missing images in my README! I'm in the process of finalizing everything for this project, and once I verify that my full CIFAR-10 sized example dataset is functional, I will upload it here for easy download, add example images to my README, and sprinkle additional ease-of-use comments throughout my code. I also will add a license soon for this to be usable, once it's more finalized.

## For Use

### Overview

This project downloads images from any MS-COCO class or Oxford-IIIT Pet class, then formats them to fit CIFAR-10 with an extra channel beyond RGB representing human-verified instance segmentation masks. It is modular and first outputs data as .npy files, after which the user can choose whether you want to convert them to .bin. The user also may specify the number of images they want, whether they would like to include background instances of objects of interest in their masks, and whether they would like to accept multiple instances cropped from the same base image, among other variables.

[random image plus mask in coco, followed by my dataset]

### Prerequisites

The user must download ONLY the annotations folder for COCO 2017. Specifically, `coco/annotations/instances_train2017.json` and `coco/annotations/instances_val2017.json` are required to already be downloaded and in the repo to run `cocoLoadFull.py` as is.

THE USER DOES NOT NEED TO DOWNLOAD ALL OF COCO! This program automatically loads images from their annotation JSON's during runtime and processes them without downloading first. This avoids unnecessary downloads and saves time and storage, but note that running the program on a full class of 6000 images can take upwards of one hour due to COCO's slow servers.

The user also needs NumPy, PyCocoTools, PIL, and Requests installed to their environment, all of which are listed in `requirements.txt`.

If the user wants to include full cat and dog classes with 6000 images each, Oxford-IIIT Pet dataset is necessary to add approximately 700 images. To use it, the full dataset must first be downloaded and put into the repo, which can be done via Torrent.

### Variability

The user can specify many parameters for their dataset. Overarchingly, `cocoLoadFull.py` and `cocoLoadIndiv.py` accomplish the same thing, but the Full version compiles an amalgamation of every instance mask in an image to be used as each instance's mask, while the Indiv version only considers the instance being cropped around as part of its mask. In images with multiple instances of the same object in close proximity, there can be instances of your class of interest with separate masks from the object of interest. The Full version considers these background masks whose full objects are likely not in frame, while the Indiv version only counts the primary object which is almost guaranteed to be fully in frame. I find the Full version to be more useful for most situations, but it started as the Indiv version and perhaps someone needs that for some reason, so the Indiv script remains in the repo.

[Images of an object with vs without full masks]

Including multiple cropped objects of the same class from the same image may result in some data leakage. Because of this, the `single_img` variable in the `loadCocoCat` function in both COCO loader scripts can be set to True to only allow one output image per input image (within the same class). Repeated instances of the same class within the same image will be ignored, and only one object will be cropped around and downloaded.

[Images of multiple objects from the same image]

All other variable explanations necessary to implement functions should be listed in comments.

### Troubleshooting

This repo includes two scripts designed to assist troubleshooting: `npReader.py` and `binaryReader.py`. Despite only drawing from human-annotated datasets, as with any data, there are mistakes and corruption in both datasets as well as unfortunate segmentation mask tracing in some areas.

[if i can find it, image of boat mask being fully white due to foreground boat cables]

`npReader.py` features `check_unique()` and `load_imgMask()`, where the former scans a folder of npy files formatted as below and prints the names of any files whose masks are only one value (all 1's or all 0's) for the user's investigation. `load_imgMask()` creates a `test/` folder and populates it with a visible image and mask from the npy file passed to it. These two, together with searching the image's ID (included in file name as explained below) on the COCO website, should help the user identify and correct any obvious issues with their dataset by deleting problematic images and replacing them.

`binaryReader.py` is still being written, but I'll explain it here once I'm done

### Errors

If the user tries to download many thousands of images at once, there is always a chance of error due to network connectivity, computer idling, etc. For this purpose, the program prints its progress roughly every 100 annotations indexed. If an error occurs, to save the user from starting all the way back at image zero, they can make the `startAnnIndex` in `loadCocoCat()` equivalent to the most recent annotation index printed to terminal and adjust `numImgs` (number of images to be downloaded. By instance, not by annotation index) accordingly to pick up downloading and processing exactly where the error occurred.

Note that the IIIT dataset also includes a few files which are corrupted. If an error occurs caused by a certain image or `check_unique()` flags one of them, simply delete it and try downloading a replacement image from a different category.

## Image Formats

### Quick Description

Images will first be downloaded as 32x32x4 .npy files where the 4 channels are R, G, B, and mask (1 being object, 0 being background). A bounding box around the object of interest would take up at least 640 of the 1024 pixels and be as near to the center of the image as the initial image allows (an object near the edge of an image might not be possible to perfectly center).

[centered image with bounding box drawn]

### Specifics

`cocoLoadFull.py` processes an input image and annotation, then crops down to a square around an instance of interest such that a bounding box around the instance would take up 62.5% of the square, unless the instance is too large for that, in which case it is simply centered in the square frame and remains the same size. After centering and adjusting instance size to greater than or equal to 62.5% of the square frame, the image and mask are downsampled to 32x32 pixels, at which point they are converted to a single .npy array with four channels per pixel (R, G, B, mask). These .npy arrays are saved automatically and can be converted to .bin with `npyToBinary.py` and deleted, or used immediately. Each file is named after its image ID in COCO/IIIT so the original image can be easily referenced at their respective websites for troubleshooting, and the COCO images include an underscore followed by the highlighted instance's index within its original image; if there are two instances of `cat` within a single COCO image and the user opts to include multiple instances per image, they will be named `[imgID]_0` and `[imgID]_1`. Note that the IIIT images always include "_n" at the end of their filenames, but this does not mean there are multiple instances in a single image (there never are multiple object masks in a single IIIT image).

[labeled img filenames with multiple instances per id]

### Binary Format

Upon running `npyToBinary.py`, images & masks are written into binary files starting with one byte representing their class number and then followed by 1024 bytes of red ((x1,y1),(x2,y1),(x3,y1),...(x1,y2,),(x2,y2),(x3,y2),...(x30,y32),(x31,y32),(x32,y32)), 1024 bytes of green, 1024 bytes of blue, and then 1024 bytes of mask data (representing 1 for object or 0 for background, again) for a total of 4097 bytes per object. This exactly matches the official CIFAR-10 format of 3073 bytes per object, plus the mask channel. Following the same naming conventions, files will retain their names and simply end with `.bin` instead of `.npy`.

## The Future

I plan to next create an STL-10 style dataset building off of this, perhaps employing ImageNet to gather mass unlabeled data guaranteed to contain at least one class of interest per image. The addition of an unlabeled data module will make this code an essential upgrade to any binary or multilabel classification research, as it will enable easy creation of custom datasets to test hypotheses.