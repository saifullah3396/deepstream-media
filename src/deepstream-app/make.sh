# make -C ./deepstream/gst-nvinfer/ clean
make -C ./deepstream/gst-nvinfer/
sudo CUDA_VER=10.2 make -C ./deepstream/gst-nvinfer/ install

# make -C ./deepstream/nvdsinfer/ clean
make -C ./deepstream/nvdsinfer/
sudo CUDA_VER=10.2 make -C ./deepstream/nvdsinfer/ install

# make clean
make