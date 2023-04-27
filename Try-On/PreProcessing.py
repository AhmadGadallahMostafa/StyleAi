# In this file we will do the preprocessing of the input image in order to make them ready for our models 
# Our input is two images: 1) The image of the person 2) The image of the clothes
# our output is : 1) DensePose of the person 2) Parse of person 3) Parse of Person without upper body 4) cloth mask 5) Agnostic image 
import os 
import PIL.Image
from carvekit.api.interface import Interface
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from carvekit.pipelines.postprocessing import MattingMethod
from carvekit.pipelines.preprocessing import PreprocessingStub
from carvekit.trimap.generator import TrimapGenerator
import cv2
import numpy as np 
from PIL import ImageOps
from carvekit.api.high import HiInterface
import torch



    
class PreProcessing:

    def get_densepose(self):
        # DensePose parameters
        apply_net_path_densepose = "Try-On/DensePose/apply_net.py"
        config_path_densepose = "Try-On/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"
        images_path_densepose = "Try-On/InputImages"
        model_path_densepose = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
        # This will save the images with the same name as the input image at PreproceessedImages folder in densepose folder 
        command = "python " + apply_net_path_densepose + " show " + config_path_densepose + " " + model_path_densepose + " " + images_path_densepose + " dp_segm -v"
        os.system(command)  

    def get_mask(self):
        seg_net = TracerUniversalB7(device='cuda',
              batch_size=1)

        fba = FBAMatting(device='cuda',
                        input_tensor_size=2048,
                        batch_size=1)

        trimap = TrimapGenerator()

        preprocessing = PreprocessingStub()

        postprocessing = MattingMethod(matting_module=fba,
                                    trimap_generator=trimap,
                                    device='cuda')

        interface = Interface(pre_pipe=preprocessing,
                            post_pipe=postprocessing,
                            seg_pipe=seg_net)
        

        for filename in os.listdir("Try-On/InputClothesImages"):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image = PIL.Image.open("Try-On/InputClothesImages/" + filename)
                # remove the extension from the filename
                filename = filename.split(".")[0]
                # this just removes the background from the image
                image = interface([image])[0]
                # thresholding 
                channels = image.split()
                alpha = channels[-1].convert('L')
                threshold = 5
                mask = alpha.point(lambda p: p > threshold and 255)
                mask.save("Try-On/PreprocessedImages/ClothMask/" + filename + ".png")
            else:
                continue

    def run(self):
        self.get_densepose()
        self.get_mask()
        

def main():
    preProcessing = PreProcessing()
    preProcessing.run()


if __name__ == "__main__":
    main()