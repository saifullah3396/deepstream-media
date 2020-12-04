DIR="$(dirname $1)"
FILE="$(basename $1)"
echo "$DIR"
echo "$FILE"
/home/sai/ncai/TensorRT-7.0.0.11/bin/trtexec --fp16 --explicitBatch \
    --workspace=256 \
    --onnx="${DIR}/${FILE}" --saveEngine="${DIR}/${FILE}.trt" \
    --minShapes=\'input\':1x1x64x16,\'scores\':1x1x168 \
    --optShapes=\'input\':1x1x64x320,\'scores\':1x81x168 \
    --maxShapes=\'input\':1x1x64x3200,\'scores\':1x801x168