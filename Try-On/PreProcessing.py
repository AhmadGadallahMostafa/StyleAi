# In this file we will do the preprocessing of the input image in order to make them ready for our models 
# Our input is two images: 1) The image of the person 2) The image of the clothes
# our output is : 1) DensePose of the person 2) Parse of person 3) Parse of Person without upper body 4) cloth mask 5) Agnostic image 
import os 

def get_densepose():
    apply_net_path = "Try-On/DensePose/apply_net.py"
    config_path = "Try-On/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"
    images_path = "Try-On/InputImages"
    model_path = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
    command = "python " + apply_net_path + " show " + config_path + " " + model_path + " " + images_path + " dp_segm -v"
    os.system(command)

def main():
    get_densepose()

if __name__ == "__main__":
    main()