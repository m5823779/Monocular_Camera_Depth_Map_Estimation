import argparse
import torch
import torch.nn.parallel
import cv2
import numpy as np
import pdb
import freenect
from torchvision import transforms
from PIL import Image
from models import modules, net, resnet, densenet, senet

# function to get RGB image from kinect
def get_video():
    array, _ = freenect.sync_get_video()
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return array

# function to get depth image from kinect
def get_depth():
    array, _ = freenect.sync_get_depth()
    array = array.astype(np.float)
    return array

def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained=True)
        Encoder = modules.E_resnet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel=[192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])

    return model


def main():
    image_pre_process_par = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    cam = cv2.VideoCapture(0)
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('pretrained_model/model_senet'))
    model.eval()
    while True:
        _, cam_img = cam.read()
        cam_img = get_video()
        img = cv2.resize(cam_img, (320, 240), interpolation=Image.BILINEAR)
        w1, h1 = 320, 240
        tw, th = 304, 228
        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))
        img = img[y1:th + y1, x1:tw + x1]
        img = torch.tensor(img / 255.)
        img = transforms.Normalize(image_pre_process_par['mean'], image_pre_process_par['std'])(img)
        img = img.permute(2, 0, 1).unsqueeze(0).float().cuda()
        pred = model(img).data[0].cpu().numpy() / 10
        show_output = cv2.resize(np.transpose(pred, [1, 2, 0]), (640, 480), interpolation=Image.BILINEAR)
        print(get_depth())
        raise
        ground_truth = get_depth() / 1000


        cv2.imshow('Raw_Image', cv2.resize(cam_img, (640, 480), interpolation=Image.BILINEAR))
        cv2.imshow('Predict_Depth_image', show_output)
        cv2.imshow('Ground_Truth', ground_truth)
        cv2.waitKey(10)

if __name__ == '__main__':
    main()
