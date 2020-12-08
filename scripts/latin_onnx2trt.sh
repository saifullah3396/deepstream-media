DIR="$(dirname $1)"
FILE="$(basename $1)"
# in our case, craft takes input of Nx3x360x640
$TENSOR_RT/bin/trtexec --fp16 --explicitBatch \
    --workspace=256 \
    --onnx="${DIR}/${FILE}" --saveEngine="${DIR}/${FILE}.trt" \
    --minShapes=\'input\':"1x1x64x$2" \
    --optShapes=\'input\':"4x1x64x$2" \
    --maxShapes=\'input\':"8x1x64x$2"
