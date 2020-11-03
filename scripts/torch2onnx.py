import argparse
import ast
from collections import OrderedDict
import os
import torch


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
    parser.add_argument('--input_shape', help='Shape of the input to model [b,c,h,w].', required=True)
    parser.add_argument('--verbose', help='Show verbose output.', nargs='?', const='')
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
        print("Unable to create shape from {}. Please give input shape in correct format [b,c,w,h]".format(args.input_shape))
        exit(1)

    if args.model == 'craft':
        from models.craft import easyocr
        model = easyocr.craft.CRAFT()
    else:
        print ('{} model is not supported.'.format(args.model))
    print('{}/{}.onnx'.format(weights_file_dir, weights_file))
    model.load_state_dict(copyStateDict(torch.load(args.weights)))
    x = torch.ones(input_shape, dtype=torch.float)
    verbose = True if args.verbose else False
    torch.onnx.export(
        model, 
        x, 
        './{}/{}.onnx'.format(weights_file_dir, weights_file),
        output_names=['scores', 'features'],
        verbose=verbose, 
        opset_version=11)


if __name__ == "__main__":
    main()