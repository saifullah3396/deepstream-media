SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd ${SCRIPTPATH}

# install gst-nvinfer
make -C ${SCRIPTPATH}/deepstream/gst-nvinfer/ clean
make -C ${SCRIPTPATH}/deepstream_base/gst-nvinfer/
sudo CUDA_VER=10.2 make -C ./deepstream_base/gst-nvinfer/ install

# install nvdsinfer
make -C ${SCRIPTPATH}/deepstream/nvdsinfer/ clean
make -C ${SCRIPTPATH}/deepstream_base/nvdsinfer/
sudo CUDA_VER=10.2 make -C ${SCRIPTPATH}/deepstream_base/nvdsinfer/ install

# install gst-nvmsgconv
make -C ${SCRIPTPATH}/deepstream_base/gst-nvmsgconv/ clean
make -C ${SCRIPTPATH}/deepstream_base/gst-nvmsgconv/
sudo CUDA_VER=10.2 make -C ${SCRIPTPATH}/deepstream_base/gst-nvmsgconv/ install

# install nvmsgconv
make -C ${SCRIPTPATH}/deepstream_base/nvmsgconv/ clean
make -C ${SCRIPTPATH}/deepstream_base/nvmsgconv/
sudo CUDA_VER=10.2 make -C ${SCRIPTPATH}/deepstream_base/nvmsgconv/ install

# install text detection parser
make -C ${SCRIPTPATH}/nvdsinfer_parsers/text_detection/craft/ clean
make -C ${SCRIPTPATH}/nvdsinfer_parsers/text_detection/craft/

# install face detection parser
make -C ${SCRIPTPATH}/nvdsinfer_parsers/face_recognition clean
make -C ${SCRIPTPATH}/nvdsinfer_parsers/face_recognition

# install text recognition parser
make -C ${SCRIPTPATH}/nvdsinfer_parsers/text_recognition clean
make -C ${SCRIPTPATH}/nvdsinfer_parsers/text_recognition

# install app
make -C ${SCRIPTPATH}/media_analytics_app