SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd ${SCRIPTPATH}

PATH_TO_FILE=../../models/face_recognition/face_recognition.pb.onnx
DIR="$(dirname $PATH_TO_FILE)"
FILE="$(basename $PATH_TO_FILE)"
# in our case, craft takes input of Nx3x360x640
$TENSOR_RT/bin/trtexec --fp16 --explicitBatch \
    --workspace=256 \
    --onnx="${DIR}/${FILE}" --saveEngine="${DIR}/${FILE}.trt" \
    --minShapes=\'input_1:0\':1x3x224x224 \
    --optShapes=\'input_1:0\':8x3x224x224 \
    --maxShapes=\'input_1:0\':16x3x224x224
