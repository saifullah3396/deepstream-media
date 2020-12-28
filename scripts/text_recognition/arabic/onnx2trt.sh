SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd ${SCRIPTPATH}

PATH_TO_FILE=../../../models/text_recognition/arabic/arabic.pth
DIR="$(dirname $PATH_TO_FILE)"
FILE="$(basename $PATH_TO_FILE)"

# generate 3 different models for 3 different sized inputs for text recognition
$TENSOR_RT/bin/trtexec --fp16 --explicitBatch \
    --workspace=256 \
    --onnx="${DIR}/${FILE%.*}-248.onnx" --saveEngine="${DIR}/${FILE%.*}-248.trt" \
    --minShapes=\'input\':"1x1x32x248" \
    --optShapes=\'input\':"4x1x32x248" \
    --maxShapes=\'input\':"8x1x32x248"

$TENSOR_RT/bin/trtexec --fp16 --explicitBatch \
    --workspace=256 \
    --onnx="${DIR}/${FILE%.*}-600.onnx" --saveEngine="${DIR}/${FILE%.*}-600.trt" \
    --minShapes=\'input\':"1x1x32x600" \
    --optShapes=\'input\':"4x1x32x600" \
    --maxShapes=\'input\':"8x1x32x600"

$TENSOR_RT/bin/trtexec --fp16 --explicitBatch \
    --workspace=256 \
    --onnx="${DIR}/${FILE%.*}-1280.onnx" --saveEngine="${DIR}/${FILE%.*}-1280.trt" \
    --minShapes=\'input\':"1x1x32x1280" \
    --optShapes=\'input\':"4x1x32x1280" \
    --maxShapes=\'input\':"8x1x32x1280"