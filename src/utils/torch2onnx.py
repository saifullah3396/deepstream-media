import argparse
import ast
from collections import OrderedDict
import os
import torch
import sys


if not os.environ['MEDIA_APP_ROOT']:
    print(
        "Please add the environment variable MEDIA_APP_ROOT as path to the "
        "application root.")
    exit(1)

sys.path.append(os.environ['MEDIA_APP_ROOT'])


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
    parser.add_argument(
        '--model', help='Path to input model weights.', required=True)
    parser.add_argument(
        '--input_shape', help='Shape of the input to model [b,c,h,w].', required=True)
    parser.add_argument(
        '--verbose', help='Show verbose output.', nargs='?', const='')
    args = parser.parse_args()
    weights_file, weights_type = os.path.splitext(args.weights)
    weights_file_dir = os.path.dirname(weights_file)
    weights_file = os.path.basename(weights_file)

    if weights_type != '.pth':
        print("{} is not pytorch file.", args.weights)

    try:
        input_shape = args.input_shape
        input_shape = input_shape.replace('[', '')
        input_shape = input_shape.replace(']', '')
        input_shape = tuple([int(x) for x in input_shape.split(',')])
    except:
        print(
            "Unable to create shape from {}. Please give input shape in "
            "correct format [b,c,h,w]".format(args.input_shape))
        exit(1)

    dynamic_axis = None
    model_name = './{}/{}.onnx'.format(weights_file_dir, weights_file)
    if args.model == 'text_detection/craft':
        from models.text_detection.craft.craft import CRAFT
        model = CRAFT(y_permute=False)
        output_names = ['scores', 'features']
        dynamic_axis = {
            'input': {0: 'batch'},
        }
    elif args.model == 'text_recognition/latin':
        from models.text_recognition.model import Model
        N_LATIN_CHARS = 168
        model = Model(1, 512, 512, N_LATIN_CHARS)
        output_names = ['scores']
        # batch and width can be dynamic
        dynamic_axis = {
            'input': {0: 'batch'},
        }
        model_name = './{}/{}-{}.onnx'.format(
            weights_file_dir, weights_file, input_shape[3])
    elif args.model == 'text_recognition/arabic':
        from models.text_recognition.model import Model
        N_ARABIC_CHARS = 185
        model = Model(1, 512, 512, N_ARABIC_CHARS)
        output_names = ['scores']
        # batch and width can be dynamic
        dynamic_axis = {
            'input': {0: 'batch'},
        }
        model_name = './{}/{}-{}.onnx'.format(
            weights_file_dir, weights_file, input_shape[3])
    else:
        print('{} model is not supported.'.format(args.model))
        exit(1)
    model.load_state_dict(copyStateDict(torch.load(args.weights)))
    x = torch.ones(input_shape, dtype=torch.float)

    verbose = True if args.verbose else False
    torch.onnx.export(
        model,
        x,
        model_name,
        input_names=['input'],
        output_names=output_names,
        verbose=verbose,
        opset_version=11,
        dynamic_axes=dynamic_axis)  # batch and width can be dynamic


if __name__ == "__main__":
    main()
