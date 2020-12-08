DIR="$(dirname $1)"
FILE="$(basename $1)"
echo "$DIR"
echo "$FILE"
# in our case, craft takes input of Nx3x360x640
$TENSOR_RT/bin/trtexec --fp16 --explicitBatch \
    --workspace=256 \
    --onnx="${DIR}/${FILE}" --saveEngine="${DIR}/${FILE}.trt" \
    --minShapes=\'input\':1x3x360x640 \
    --optShapes=\'input\':8x3x360x640 \
    --maxShapes=\'input\':16x3x360x640
