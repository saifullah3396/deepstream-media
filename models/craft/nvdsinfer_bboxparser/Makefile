DEEPSTREAM_PATH=/opt/nvidia/deepstream/deepstream
CC:= g++

CFLAGS:= -Wall -std=c++11

CFLAGS+= -shared -fPIC

CFLAGS+= -I${DEEPSTREAM_PATH}/sources/includes

LIBS:= -lnvinfer -lnvparsers -lopencv_core -lopencv_imgproc -lopencv_calib3d
LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES:= nvdsinfer_craftbboxparser.cpp
TARGET_LIB:= libnvds_infercraftparser.so

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

install: $(TARGET_LIB)
	cp $(TARGET_LIB) ../../../lib

clean:
	rm -rf $(TARGET_LIB)
