################################################################################
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

APP:= media_analytics_app

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

NVDS_VERSION:=5.0

DEEPSTREAM_SOURCES=../deepstream_base
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/

ifeq ($(TARGET_DEVICE),aarch64)
	CFLAGS:= -DPLATFORM_TEGRA
endif

SRCS:= $(wildcard *.c)
SRCS+= $(wildcard ../apps_common/src/*.c)
CXX_SRCS+= $(wildcard *.cpp)

INCS:= $(wildcard *.h)

PKGS:= \
		gstreamer-1.0 gstreamer-video-1.0 glib-2.0 gobject-2.0 json-glib-1.0 \
		uuid x11

OBJS:= $(SRCS:.c=.o)
CXX_OBJS:= $(CXX_SRCS:.cpp=.o)

CFLAGS+= \
		-I../apps_common/includes \
		-I$(DEEPSTREAM_SOURCES)/includes \
		-DDS_VERSION_MINOR=0 \
		-DDS_VERSION_MAJOR=5
CXXFLAGS+= \
		-I../apps_common/includes \
		-I$(DEEPSTREAM_SOURCES)/includes \
		-DDS_VERSION_MINOR=0 \
		-DDS_VERSION_MAJOR=5

LIBS+= -L$(LIB_INSTALL_DIR) \
		-lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvdsgst_smartrecord \
		-lnvds_utils -lm -lgstrtspserver-1.0 -ljsoncpp -lopencv_core \
		-lopencv_imgproc -lopencv_calib3d -ldl -Wl,-rpath,$(LIB_INSTALL_DIR)

CFLAGS+= `pkg-config --cflags $(PKGS)`
CXXFLAGS+= `pkg-config --cflags $(PKGS)`

LIBS+= `pkg-config --libs $(PKGS)`

all: $(APP)

%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CXXFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CC) -o $(APP) $(OBJS) $(LIBS)

$(APP): $(CXX_OBJS) Makefile
	$(CXX) -o $(APP) $(OBJS) $(CXX_OBJS) $(LIBS)

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(APP)
