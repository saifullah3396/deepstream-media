DIR="$(dirname $1)"
FILE="$(basename $1)"
echo "$DIR"
echo "$FILE"
/home/sai/ncai/TensorRT-7.0.0.11/bin/trtexec --fp16 --explicitBatch \
    --onnx="${DIR}/${FILE}" --saveEngine="${DIR}/${FILE}.trt" \
    --minShapes=\'input_1:0\':1x3x224x224 \
    --optShapes=\'input_1:0\':8x3x224x224 \
    --maxShapes=\'input_1:0\':32x3x224x224
