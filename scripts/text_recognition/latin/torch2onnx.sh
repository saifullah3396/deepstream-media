SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd ${SCRIPTPATH}

PATH_TO_FILE=../../../models/text_recognition/latin/latin.pth
DIR="$(dirname $PATH_TO_FILE)"
FILE="$(basename $PATH_TO_FILE)"

# generate 3 different models for 3 different sized inputs for text recognition
python ../../../src/utils/torch2onnx.py --weights "${DIR}/${FILE}" \
    --model text_recognition/latin --input_shape "[1,1,32,248]"
python ../../../src/utils/torch2onnx.py --weights "${DIR}/${FILE}" \
    --model text_recognition/latin --input_shape "[1,1,32,600]"
python ../../../src/utils/torch2onnx.py --weights "${DIR}/${FILE}" \
    --model text_recognition/latin --input_shape "[1,1,32,1280]"