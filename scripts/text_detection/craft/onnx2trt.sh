SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd ${SCRIPTPATH}

PATH_TO_FILE=../../../models/text_detection/craft/craft.onnx
DIR="$(dirname $PATH_TO_FILE)"
FILE="$(basename $PATH_TO_FILE)"
# in our case, text_detection model craft takes input of Nx3x360x640
$TENSOR_RT/bin/trtexec --fp16 --explicitBatch \
    --workspace=256 \
    --onnx="${DIR}/${FILE}" --saveEngine="${DIR}/${FILE}.trt" \
    --minShapes=\'input\':1x3x360x640 \
    --optShapes=\'input\':8x3x360x640 \
    --maxShapes=\'input\':16x3x360x640
