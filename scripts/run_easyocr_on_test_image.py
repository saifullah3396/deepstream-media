import argparse
import ast
from collections import OrderedDict
import os
import torch
from torch.autograd import Variable
import cv2
import numpy as np

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def main():
    parser = argparse.ArgumentParser(
        description='Convert pytorch models to onnx model.')
    parser.add_argument('--weights', help='Model import path.', required=True)
    parser.add_argument('--model', help='Path to input model weights.', required=True)
    parser.add_argument('--input_shape', help='Shape of the input to model [w,h].', required=True)
    args = parser.parse_args()
    weights_file, weights_type = os.path.splitext(args.weights)
    weights_file_dir = os.path.dirname(weights_file)
    weights_file = os.path.basename(weights_file)
    
    if weights_type != '.pth':
        print("{} is not pytorch file.", args.weights)
    
    try:
        input_shape = args.input_shape
        input_shape = input_shape.replace('[','')
        input_shape = input_shape.replace(']','')
        input_shape = tuple([int(x) for x in input_shape.split(',')])
    except:
        print("Unable to create shape from {}. Please give input shape in correct format [w,h]".format(args.input_shape))
        exit(1)

    if args.model == 'craft':
        from models.craft import easyocr
        model = easyocr.craft.CRAFT()
        from easyocr.craft_utils import getDetBoxes, adjustResultCoordinates
    else:
        print ('{} model is not supported.'.format(args.model))
    print('{}/{}.onnx'.format(weights_file_dir, weights_file))
    
    # load the image
    img = cv2.imread('test_image.png')
    img = cv2.resize(img, input_shape).astype(np.float32)
    draw_image = img.copy()

    # normalize the image as done in deepstream
    net_scale_factor = 0.01735207357279195 # see pgie config for craft
    mean = (123.675, 116.28, 103.53) # see pgie config for craft
    img -= np.array([mean[0], mean[1], mean[2]], dtype=np.float32)
    img /= net_scale_factor

    # convert image to torch tensor
    x = torch.from_numpy(img).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    model.load_state_dict(copyStateDict(torch.load(args.weights)))
    y, features = model.forward(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    boxes, polys = getDetBoxes(score_text, score_link, 0.7, 0.4, 0.4, False)
    boxes = adjustResultCoordinates(boxes, 1, 1)
    polys = adjustResultCoordinates(polys, 1, 1)

    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]
    
    result = []
    for i, box in enumerate(polys):
        poly = np.array(box).astype(np.int32).reshape((-1))
        result.append(poly)

    for box in boxes:
        cv2.rectangle(draw_image, tuple(box[0]), tuple(box[2]), (255,0,0))

    reader = easyocr.easyocr.Reader(lang_list=['en'])
    boxes, frees = reader.detect(draw_image)

    for box in boxes:
        print(box)
        cv2.rectangle(draw_image, (box[0], box[2]), (box[1], box[3]), (0,255,0))
    
    # for free in frees:
    #     print(free)
    #     cv2.rectangle(draw_image, (free[0], free[2]), (free[1], free[3]), (255,0,0))
    # print(draw_image.shape)
    # print(img.shape)

    cv2.imwrite("image.png", draw_image)

if __name__ == "__main__":
    main()