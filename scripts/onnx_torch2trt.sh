DIR="$(dirname $1)"
FILE="$(basename $1)"
echo "$DIR"
echo "$FILE"
/home/sai/ncai/TensorRT-7.0.0.11/bin/trtexec --fp16 --explicitBatch \
    --workspace=6000 \
    --onnx="${DIR}/${FILE}" --saveEngine="${DIR}/${FILE}.trt" \
    --minShapes=\'input\':1x1x64x16,\'input\':1x1x64x16 \
    --optShapes=\'input\':1x1x64x320,\'input\':1x1x64x320 \
    --maxShapes=\'input\':1x1x64x3200,\'input\':2x1x64x3200 --verbose 
