DEEPSTREAM_PATH=/opt/nvidia/deepstream/deepstream
CC:= g++
CFLAGS:= -std=c++11 -g -Ofast
CFLAGS+= -shared -fPIC
CFLAGS+= -I/usr/local/include/kenlm -I/usr/local/include
LIBS:= -L/usr/lib/x86_64-linux-gnu/ \
	-lfst -lkenlm -lkenlm_util -lbz2 -llzma -lpthread -lz
LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group -rdynamic
SRCFILES:= ./*.cpp
TARGET_LIB:= libctc_decoder.so
all: $(TARGET_LIB)
$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)
install: $(TARGET_LIB)
	cp $(TARGET_LIB) /usr/local/lib
clean:
	rm -rf $(TARGET_LIB)
