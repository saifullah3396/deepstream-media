DIR="$(dirname $1)"
FILE="$(basename $1)"
# in our case, craft takes input of Nx3x360x640
python torch2onnx.py --weights "${DIR}/${FILE}" \
    --model craft --input_shape [1,3,360,640]
