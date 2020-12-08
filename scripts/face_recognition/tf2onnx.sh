SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd ${SCRIPTPATH}

PATH_TO_FILE=$(realpath ../../models/face_recognition/face_recognition.pb)

if [[ $PATH_TO_FILE != *.pb ]]; then
    echo 1>&2 "$0: The model must be a .pb file."
    exit 2
fi

DIR="$(dirname $PATH_TO_FILE)"
FILE="$(basename $PATH_TO_FILE)"
python -m tf2onnx.convert --graphdef ${DIR}/${FILE} \
    --opset 11 \
    --output ${DIR}/${FILE}.onnx --inputs input_1:0 \
    --outputs global_average_pooling2d_1/Mean:0 \
    --inputs-as-nchw input_1:0