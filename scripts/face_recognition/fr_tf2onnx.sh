SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd ${SCRIPTPATH}
../../src/utils/tf2onnx.sh $(realpath ../../models/face_recognition/face_recognition.pb)
