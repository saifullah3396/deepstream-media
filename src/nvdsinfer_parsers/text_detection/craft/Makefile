DEEPSTREAM_PATH=/opt/nvidia/deepstream/deepstream
CC:= g++

CFLAGS:= -Wall -std=c++11 -O2 -s -DNDEBUG

CFLAGS+= -shared -fPIC # -DDEBUG

CFLAGS+= \
	-I$(DEEPSTREAM_PATH)/sources/includes \
	-I$(TENSOR_RT)/include \
	-I/usr/local/include/opencv4/opencv2

LIBS:= \
	-lnvinfer \
	-lnvparsers \
	-L/usr/local/lib \
	/usr/local/lib/libopencv_core.so \
	/usr/local/lib/libopencv_imgproc.so \
	/usr/local/lib/libopencv_calib3d.so \
	/usr/local/lib/libopencv_highgui.so \
	/usr/local/lib/libopencv_imgcodecs.so \
	-L$(TENSOR_RT)/lib
LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES:= nvdsinfer_bboxparser.cpp
TARGET_LIB:= libnvdsinfer_bboxparser.so

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

install: $(TARGET_LIB)
	cp $(TARGET_LIB) ../../../lib

clean:
	rm -rf $(TARGET_LIB)
