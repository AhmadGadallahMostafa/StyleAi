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
import subprocess



    
class PreProcessing:
    def __init__(self, input_path_image = "Try-On/InputImages", input_path_cloth = "Try-On/InputClothesImages"):
        self.input_path_image = input_path_image
        self.input_path_cloth = input_path_cloth
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

        self.interface = Interface(pre_pipe=preprocessing,
                            post_pipe=postprocessing,
                            seg_pipe=seg_net)


    def get_densepose(self):
        # DensePose parameters
        apply_net_path_densepose = "Try-On/DensePose/apply_net.py"
        config_path_densepose = "Try-On/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"
        images_path_densepose = self.input_path_image
        model_path_densepose = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
        # This will save the images with the same name as the input image at PreproceessedImages folder in densepose folder 
        command = "python " + apply_net_path_densepose + " show " + config_path_densepose + " " + model_path_densepose + " " + images_path_densepose + " dp_segm -v"
        os.system(command)  

    def supress_background(self):
        # this function will supress the background of the person image and make it white
        for filename in os.listdir(self.input_path_image):
            image = PIL.Image.open(self.input_path_image + '/' + filename)
            # this just removes the background from the image
            image = self.interface([image])[0]
            # add white background
            background = PIL.Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image)
            # Delete the old image
            os.remove(self.input_path_image + '/' + filename)
            # save the new image
            background.save(self.input_path_image + '/' + filename)

            



    def get_mask(self):
        for filename in os.listdir("Try-On/InputClothesImages"):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image = PIL.Image.open(self.input_path_cloth + '/' + filename)
                # remove the extension from the filename
                filename = filename.split(".")[0]
                # this just removes the background from the image
                image = self.interface([image])[0]
                # thresholding 
                channels = image.split()
                alpha = channels[-1].convert('L')
                threshold = 5
                mask = alpha.point(lambda p: p > threshold and 255)
                mask.save("Try-On/PreprocessedImages/ClothMask/" + filename + ".png")
            else:
                continue

    def get_openpose(self):
        # switch directory to openpose
        old_dir = os.getcwd()
        os.chdir("Try-On/openPose")
        exe_path = "bin/OpenPoseDemo.exe"
        input_path =  "../../" + self.input_path_image
        output_path = "../../" + "Try-On\PreprocessedImages\OpenPose"
        output_json_path =  "../../" + "Try-On\PreprocessedImages\OpenPosejson"
        command = exe_path + " --image_dir " + input_path + " --disable_blending --write_json " + output_json_path + " --display 0 -write_images " + output_path + " --num_gpu 1 --hand --num_gpu_start 0"
        subprocess.run([exe_path, "--image_dir", input_path, "--disable_blending", "--write_json", output_json_path, "--display", "0", "--write_images", output_path, "--num_gpu", "1", "--hand", "--num_gpu_start", "0"])
        # switch directory back to Try-On
        os.chdir(old_dir)

    def get_parse(self):
        parser_path = "Try-On/HumanParse/simple_extractor.py"
        input_path = self.input_path_image
        output_path = "Try-On/PreprocessedImages/Parse"
        command = "python " + parser_path + " --dataset lip --model-restore Try-On/HumanParse/checkpoints/final.pth --input-dir " + input_path + " --output-dir " + output_path
        os.system(command)

    def get_parse_agnostic(self):
        parse_agnostic_path = "Try-On/ParseAgnostic/parse_agnostic.py"
        input_path = "Try-On/PreprocessedImages/Parse/"
        json_path = "Try-On/PreprocessedImages/OpenPosejson"
        output_path = "Try-On/PreprocessedImages/ParseWithoutUpper"
        command = "python " + parse_agnostic_path + " --image_path " + input_path + " --json_path " + json_path + " --output_path " + output_path
        os.system(command)

    def get_agnostic(self):
        agnostic_path = "Try-On/HumanAgnostic/HumanAgnostic.py"
        input_path = self.input_path_image
        output_path = "Try-On/PreprocessedImages/Agnostic"
        json_path = "Try-On/PreprocessedImages/OpenPosejson"
        parse_path = "Try-On/PreprocessedImages/Convertedparse"
        command = "python " + agnostic_path + " --image_path " + input_path + " --output_path " + output_path + " --json_path " + json_path + " --parse_path " + parse_path
        os.system(command)

    def convert_parse(self):
        convert_parse_path = "Try-On/ConvertParse.py"
        parse_path = "Try-On/PreprocessedImages/Parse"
        command = "python " + convert_parse_path + " --parse_path " + parse_path
        os.system(command)

    def run(self):
        self.supress_background()
        # self.get_densepose()
        # self.get_mask()
        # self.get_openpose()
        # self.get_parse()
        # self.convert_parse()
        # self.get_parse_agnostic()
        # self.get_agnostic()

def main():
    preProcessing = PreProcessing()
    preProcessing.run()


if __name__ == "__main__":
    main()