DIR="$(dirname $1)"
FILE="$(basename $1)"
echo "$DIR"
echo "$FILE"
# in our case, craft takes input of Nx3x360x640
$TENSOR_RT/bin/trtexec --fp16 --explicitBatch \
    --workspace=256 \
    --onnx="${DIR}/${FILE}" --saveEngine="${DIR}/${FILE}.trt" \
    --minShapes=\'input_1:0\':1x3x224x224 \
    --optShapes=\'input_1:0\':8x3x224x224 \
    --maxShapes=\'input_1:0\':16x3x224x224
