DIR="$(dirname $1)"
FILE="$(basename $1)"
# in our case, craft takes input of Nx3x360x640
python torch2onnx.py --weights "${DIR}/${FILE}" \
    --model text_recognizer_latin --input_shape "[1,1,64,$2]"
