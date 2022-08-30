import cv2
import torch
import argparse
import numpy as np
from torch.cuda import is_available
from network import MyNet

""" This is the inference script for testing the trained model """

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", help="trained model weights")
    parser.add_argument("--input", help="input image")
    parser.add_argument("--output", type=str, help="path to save output")
    parser.add_argument('--nChannel', metavar='N', default=100, type=int, 
                        help='number of channels')
    parser.add_argument('--nConv', metavar='M', default=2, type=int, 
                        help='number of convolutional layers')
    args = parser.parse_args()
    return args

args = args_parser()

model = MyNet(3, args.nChannel, args.nConv) # nChannel and nConv as used for training

model.load_state_dict(torch.load(args.weights)) # loads the weigths of the trained model

device = ("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def viz():
    img = cv2.imread(args.input) #reads your image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(214, 214), interpolation=cv2.INTER_AREA)

    label_colours = np.random.randint(255,size=(100,3))

    img = torch.from_numpy(img.transpose((2, 0, 1)).astype("float32")) / 255.

    img = img.unsqueeze(0)
    img = img.to(device=device)
    output = model( img ) # predictions

    output = output.data.squeeze(0)
    img = img.squeeze(0)
    img = img.permute((2, 1, 0))
    
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )

    _, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()

    im_target_rgb = np.array([label_colours[ c % args.nChannel ] for c in im_target])

    im_target_rgb = im_target_rgb.reshape( img.shape ).astype( np.uint8 )

    
    cv2.imwrite(args.output, im_target_rgb) # write the segemented image to disk

if __name__ == "__main__":
    viz()

