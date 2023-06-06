from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
from carvekit.api.interface import Interface
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from carvekit.pipelines.postprocessing import MattingMethod
from carvekit.pipelines.preprocessing import PreprocessingStub
from carvekit.trimap.generator import TrimapGenerator
# opencv
import cv2 as cv

seg_net = TracerUniversalB7(device='cuda', batch_size=1)
fba = FBAMatting(device='cuda', input_tensor_size=2048, batch_size=1)
trimap = TrimapGenerator()
preprocessing = PreprocessingStub()
postprocessing = MattingMethod(matting_module=fba, trimap_generator=trimap, device='cuda')
interface = Interface(pre_pipe=preprocessing, post_pipe=postprocessing, seg_pipe=seg_net)

# loop over the images in outputSeg folder
segmented_images = []
og_images = []
for filename in os.listdir('Try-On/our parse/outputSeg'):
    segmented_images.append(filename)
for filename in os.listdir('Try-On/our parse/inputSeg'):
    og_images.append(filename)

# loop over the segmented images
for filename in segmented_images:
    seg_image = Image.open('Try-On/our parse/outputSeg/' + filename)
    # get the original image
    og_image = None
    for og_filename in og_images:
        if og_filename[:-4] == filename[:-4]:
            og_image = Image.open('Try-On/our parse/inputSeg/' + og_filename)
            break
    # resize both images to 1024x768
    seg_image = seg_image.resize((1024, 768))
    og_image = og_image.resize((1024, 768))

    # Apply image matting and get the alpha matte
    og_image = interface([og_image])[0]
    # convert the alpha matte to a numpy array
    og_image = np.array(og_image)
    # convert the segmented image to a numpy array
    seg_image = np.array(seg_image)

    # show the top garment part of the segmented image where segmentation is 1
    # and the rest of the image is black
    top_garment = np.where(seg_image == 1, seg_image, 0)
    bottom_garment = np.where(seg_image == 2, seg_image, 0)

    # Convert og_image to grayscale
    og_image_gray = cv.cvtColor(og_image, cv.COLOR_RGB2GRAY)

    # get the difference between the top garment and the original image
    mask_one = np.where((top_garment == 1) | (bottom_garment == 2), og_image_gray, 0)
    mask_two = np.where((top_garment == 0) | (bottom_garment == 0), og_image_gray, 0)

    # get the difference between the two masks
    mask = mask_two - mask_one
    # make anything equal to 130 to be 0
    mask = np.where(mask == 130, 0, mask)

    # color any non-zero value to be red
    mask = np.where(mask != 0, 3, mask)

    # add the mask to the segmented image
    seg_image = np.where(mask == 3, mask, seg_image)

    # create empty image with the same size as the segmented image
    empty_image = np.zeros((768, 1024, 3), dtype=np.uint8)
    # color the empty image to be pink where the segmented image is 3

    # color the empty image to be black where the segmented image is 0
    # color the empty image to be purple where the segmented image is 1
    # color the empty image to be red where the segmented image is 2
    for i in range(768):
        for j in range(1024):
            if seg_image[i][j] == 0:
                empty_image[i][j] = [0, 0, 0]
            elif seg_image[i][j] == 1:
                empty_image[i][j] = [128, 0, 128]
            elif seg_image[i][j] == 2:
                empty_image[i][j] = [192, 0, 0]
            elif seg_image[i][j] == 3:
                empty_image[i][j] = [192, 0, 128]



    # save the image
    output_path = 'Try-On/our parse/outputParse/' + filename
    output_image = Image.fromarray(empty_image)
    output_image.save(output_path)
