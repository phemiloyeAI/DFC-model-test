import cv2
import torch
import argparse
import numpy as np
from torch.cuda import is_available
from yaml import load
from network import MyNet
import os
from collections import Counter

def choose_weights( crop_stage, weight_folder ):
    if crop_stage == 1:
        # model for stage 1 
        weights_path = os.path.join(weight_folder,'stage1_weights_c_100_conv_2_minlabel_6.pt')  
    else:
        # model for stage 2-3
        weights_path = os.path.join(weight_folder,'stage2-3_weights_c_100_conv_2_minlabel_6.pt')  
    return weights_path

def load_model( weight, nChannel, nConv):
    '''
    load DFC model based on crop stage.
        Args: 
            weight - weight path
            nChannel - number of channel
            nConv - number of convolution layer
        return:
            model - DFC model
            device - selected device to perform calculation
    '''
    model = MyNet(3, nChannel, nConv)  # nChannel and nConv used in training
    model.load_state_dict(torch.load(weight, map_location=torch.device('cpu'))) # loads the weigths of the trained model
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return model, device

def DFC_inference(input, weight_path):
    '''
    DFC model inference.
        Args:
            input - input image
            weight_path - weight path
        Return:
            DFC_mask - DFC inferenced mask
    '''
    np.random.seed(0)

    nChannel = 100
    nConv = 2

    model, device = load_model( weight_path, nChannel, nConv)
    resize_img = cv2.resize(input, dsize=(224, 224),
                            interpolation=cv2.INTER_AREA)

    label_colours = np.random.randint(255, size=(100, 3))

    input = torch.from_numpy(resize_img.transpose(
        (2, 0, 1)).astype("float32")) / 255.

    input = input.unsqueeze(0)
    input = input.to(device=device)
    output = model(input)  # predictions

    output = output.data.squeeze(0)
    input = input.squeeze(0)
    input = input.permute((2, 1, 0))

    output = output.permute(1, 2, 0).contiguous().view(-1, nChannel)
    _, target = torch.max(output, 1)
    im_target = target.data.cpu().numpy()

    im_target_rgb = np.array([label_colours[c % nChannel] for c in im_target])
    im_target_rgb = im_target_rgb.reshape(input.shape).astype(np.uint8)

    DFC_mask = im_target_rgb

    return DFC_mask

def color_seg( input_image, hsv_seg_img, HSV_MIN, HSV_MAX):
    '''
    Creates a HSV mask based lower and upper color limits.
        Args:
            input_image - input image
            hsv_seg_img - HSV segmented image
            HSV_MIN - HSV lower limit 
            HSV_MAX - HSV upper limit
        Returns:
            HSV_masked_img - HSV masked image 
    ''' 
    mask = cv2.inRange(hsv_seg_img, HSV_MIN, HSV_MAX)
    HSV_masked_img = cv2.bitwise_and(input_image, input_image, mask = mask)
    return HSV_masked_img

def extract_HSV_mask(input_image):
    '''
    Creates a HSV green or yellow mask.
        Args:
            input_image - input image

        Returns:
            masked_img - HSV masked image
    '''    
    # Set green HSV limits 
    HSV_MIN = np.array([39, 26, 105],np.uint8)
    HSV_MAX = np.array([88, 237, 223],np.uint8)

    # Transform image to hsv format
    hsv_seg_img = cv2.cvtColor(input_image,cv2.COLOR_BGR2HSV)
    # Extract green mask 
    masked_img = color_seg( input_image, hsv_seg_img, HSV_MIN, HSV_MAX)

    black_px = np.where(masked_img[:,:,1] == 0)
    total_px = input_image.shape[0] * input_image.shape[0]
    green_px_count = total_px - len(black_px[0])

    # Extract yellow mask if green mask fails
    if green_px_count < 10:
        
        # Set yellow HSV limits
        HSV_MIN = np.array([15, 80, 80],np.uint8)
        HSV_MAX = np.array([30, 255, 225],np.uint8)
        masked_img = color_seg( input_image, hsv_seg_img, HSV_MIN, HSV_MAX)

    return masked_img

def most_frequent(List):
    '''
        Find most frequent item in a list.
            Args:
                List - list to be searched

            Returns:
                num - the most frequent item 
    '''  
    num = [Counter(col).most_common(1)[0][0] for col in zip(*List)]
    return num

def get_dfc_crop_label(dfc_res_img, HSV_masked_img):
    '''
    Finds the color value in the DFC result image that correspond to green or yellow in the HSV masked image
        Args:
            dfc_res_img - the resulting mask imag from DFC model (np.array)
            HSV_masked_img - the resulting image from the color segemened image (np.array) 

        Returns:
            crop_pixel - crop label color
            dfc_color_ex - DFC mask color of interest ( numpy array )
    '''
    #Pick green pixels
    green_px = np.where(HSV_masked_img[:,:,1] > 0)

    # Picks all pixels
    upper = len(green_px[0])
    dfc_px = []
    for i in range( upper ):
    
        px_x = green_px[0][i]
        px_y = green_px[1][i] 
        dfc_px.append(list(dfc_res_img[px_x,px_y,:]))
    
    # Find the most occurance pixel / set as crop label
    crop_pixel = np.asarray(most_frequent(dfc_px))

    dfc_mask_val = crop_pixel.reshape(1,1,3)    
    
    #Fill a color example with the "plant" color 
    dfc_color_ex_x = np.copy(dfc_mask_val)
    
    for n_x in range(dfc_res_img.shape[0]-1):
        dfc_color_ex_x = np.append(dfc_color_ex_x,dfc_mask_val, axis = 0)
    dfc_color_ex = np.copy(dfc_color_ex_x)
    for n_y in range(dfc_res_img.shape[1]-1):
        dfc_color_ex = np.append(dfc_color_ex,dfc_color_ex_x, axis = 1)
    
    return crop_pixel, dfc_color_ex

def CropCoverageArea( plant_label, dfc_image ):
    '''
    Calculate Crop Coverage Area in dfc output. Divide total number of plant pixel by total pixels.
        Args: 
            plant_label - pixel color represent plant in DFC image
            dfc_image - DFC model output image
        Returns:
            crop_area - percentage of plant_label in DFC image 
    '''
    crop_px_count = np.count_nonzero(np.all(dfc_image==plant_label,axis=2))
    crop_area = round((crop_px_count / (dfc_image.shape[0]*dfc_image.shape[1]))*100,2)
    return crop_area