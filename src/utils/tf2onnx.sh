if [ $# -lt 1 ]; then
    echo 1>&2 \
        "$0: Please provide the path to the frozen model file."
    exit 2
elif [ $# -gt 1 ]; then
    echo 1>&2 "$0: The script only accepts the path to the frozen model file" \
        "as an argument."
    exit 2
fi

if [[ $1 != *.pb ]]; then
    echo 1>&2 "$0: The model must be a .pb file."
    exit 2
fi

DIR="$(dirname $1)"
FILE="$(basename $1)"
python -m tf2onnx.convert --graphdef ${DIR}/${FILE} \
    --output ${DIR}/${FILE}.onnx --inputs input_1:0 \
    --outputs global_average_pooling2d_1/Mean:0 \
    --inputs-as-nchw input_1:0