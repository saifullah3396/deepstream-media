DEEPSTREAM_PATH=/opt/nvidia/deepstream/deepstream
CC:= g++

CFLAGS:= -Wall -std=c++11

CFLAGS+= -shared -fPIC # -DDEBUG

CFLAGS+= -I${DEEPSTREAM_PATH}/sources/includes -I${TENSOR_RT}/include

LIBS:= -lnvinfer -lnvparsers -lopencv_core -lopencv_imgproc -lopencv_calib3d \
	-L${TENSOR_RT}/lib -ljsoncpp
LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES:= nvdsinfer_textrecognitionparser.cpp
TARGET_LIB:= libnvdsinfer_textrecognitionparser.so

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

install: $(TARGET_LIB)
	cp $(TARGET_LIB) ../../../lib

clean:
	rm -rf $(TARGET_LIB)
