SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd ${SCRIPTPATH}

# make -C ${SCRIPTPATH}/deepstream/gst-nvinfer/ clean
make -C ${SCRIPTPATH}/deepstream_base/gst-nvinfer/
sudo CUDA_VER=10.2 make -C ./deepstream_base/gst-nvinfer/ install

# make -C ${SCRIPTPATH}/deepstream/nvdsinfer/ clean
make -C ${SCRIPTPATH}/deepstream_base/nvdsinfer/
sudo CUDA_VER=10.2 make -C ${SCRIPTPATH}/deepstream_base/nvdsinfer/ install

make -C ${SCRIPTPATH}/nvdsinfer_parsers/text_detection
make -C ${SCRIPTPATH}/nvdsinfer_parsers/face_recognition
make -C ${SCRIPTPATH}/nvdsinfer_parsers/text_recognition
make -C ${SCRIPTPATH}/media_analytics_app