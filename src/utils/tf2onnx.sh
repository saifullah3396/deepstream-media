DIR="$(dirname $1)"
FILE="$(basename $1)"
echo "$DIR"
echo "$FILE"
python -m tf2onnx.convert --graphdef ${DIR}/${FILE} \
    --output ${DIR}/${FILE}.onnx --inputs input_1:0 \
    --outputs global_average_pooling2d_1/Mean:0 \
    --inputs-as-nchw input_1:0