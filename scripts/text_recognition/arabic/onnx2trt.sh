SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd ${SCRIPTPATH}

PATH_TO_FILE=../../../models/text_recognition/arabic/arabic.pth
DIR="$(dirname $PATH_TO_FILE)"
FILE="$(basename $PATH_TO_FILE)"

# generate 3 different models for 3 different sized inputs for text recognition
$TENSOR_RT/bin/trtexec --fp16 --explicitBatch \
    --workspace=256 \
    --onnx="${DIR}/${FILE%.*}-496.onnx" --saveEngine="${DIR}/${FILE%.*}-496.trt" \
    --minShapes=\'input\':"1x1x64x496" \
    --optShapes=\'input\':"4x1x64x496" \
    --maxShapes=\'input\':"8x1x64x496"

$TENSOR_RT/bin/trtexec --fp16 --explicitBatch \
    --workspace=256 \
    --onnx="${DIR}/${FILE%.*}-1200.onnx" --saveEngine="${DIR}/${FILE%.*}-1200.trt" \
    --minShapes=\'input\':"1x1x64x1200" \
    --optShapes=\'input\':"4x1x64x1200" \
    --maxShapes=\'input\':"8x1x64x1200"

$TENSOR_RT/bin/trtexec --fp16 --explicitBatch \
    --workspace=256 \
    --onnx="${DIR}/${FILE%.*}-2560.onnx" --saveEngine="${DIR}/${FILE%.*}-2560.trt" \
    --minShapes=\'input\':"1x1x64x2560" \
    --optShapes=\'input\':"4x1x64x2560" \
    --maxShapes=\'input\':"8x1x64x2560"