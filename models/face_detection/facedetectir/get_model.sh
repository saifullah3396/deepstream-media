mkdir -p facedetectir && \
    wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_facedetectir/versions/pruned_v1.0/files/resnet18_facedetectir_pruned.etlt \
    -O facedetectir/resnet18_facedetectir_pruned.etlt && \
    wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_facedetectir/versions/pruned_v1.0/files/facedetectir_int8.txt \
    -O facedetectir/facedetectir_int8.txt