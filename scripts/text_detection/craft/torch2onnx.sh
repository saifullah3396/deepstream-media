PATH_TO_FILE=../../../models/text_detection/craft/craft.pth
DIR="$(dirname $PATH_TO_FILE)"
FILE="$(basename $PATH_TO_FILE)"
# in our case, text_detection model craft takes input of Nx3x360x640
python ../../../src/utils/torch2onnx.py --weights "${DIR}/${FILE}" \
    --model text_detection/craft --input_shape [1,3,360,640]
