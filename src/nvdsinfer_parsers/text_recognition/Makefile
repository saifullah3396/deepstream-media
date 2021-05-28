DEEPSTREAM_PATH=/opt/nvidia/deepstream/deepstream
CC:= g++

CFLAGS:= -std=c++11 -g -Ofast

CFLAGS+= -shared -fPIC

CFLAGS+= -I${DEEPSTREAM_PATH}/sources/includes \
			-I${TENSOR_RT}/include \
			-I./sym_spell_cpp/include \
			-I/usr/local/include/kenlm

LIBS:= \
	-L/usr/lib/x86_64-linux-gnu/ -L${TENSOR_RT}/lib -L/usr/local/lib/ \
	-lnvinfer -lnvparsers -lopencv_core -lopencv_imgproc -lopencv_calib3d \
	-ljsoncpp -lfst -lkenlm -lkenlm_util -lbz2 -llzma -lpthread -lz \
	-lsym_spell_cpp -lctc_decoder

LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group -rdynamic

SRCFILES:= ./*.cpp
TARGET_LIB:= libnvdsinfer_classifier.so

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

install: $(TARGET_LIB)
	cp $(TARGET_LIB) ../../../lib

clean:
	rm -rf $(TARGET_LIB)
