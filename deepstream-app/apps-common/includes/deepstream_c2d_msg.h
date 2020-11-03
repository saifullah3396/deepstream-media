/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */


#ifndef __NVGSTDS_C2D_MSG_H__
#define __NVGSTDS_C2D_MSG_H__

#include <gst/gst.h>
#include "nvds_msgapi.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef NvDsMsgApiHandle (*nvds_msgapi_connect_ptr)(const char *connection_str,
    nvds_msgapi_connect_cb_t connect_cb, const char *config_path);

typedef NvDsMsgApiErrorType (*nvds_msgapi_disconnect_ptr)(NvDsMsgApiHandle conn);

typedef NvDsMsgApiErrorType (*nvds_msgapi_subscribe_ptr)(NvDsMsgApiHandle conn,
    char **topics, int num_topics, nvds_msgapi_subscribe_request_cb_t  cb,
    void *user_ctx);

typedef struct NvDsC2DContext {
  gpointer libHandle;
  gchar *protoLib;
  gchar *connStr;
  gchar *configFile;
  NvDsMsgApiHandle connHandle;
  nvds_msgapi_connect_ptr nvds_msgapi_connect;
  nvds_msgapi_disconnect_ptr nvds_msgapi_disconnect;
  nvds_msgapi_subscribe_ptr nvds_msgapi_subscribe;
} NvDsC2DContext;

typedef struct NvDsMsgConsumerConfig {
  gboolean enable;
  gchar *proto_lib;
  gchar *conn_str;
  gchar *config_file_path;
  GPtrArray *topicList;
} NvDsMsgConsumerConfig;

NvDsC2DContext*
start_cloud_to_device_messaging (NvDsMsgConsumerConfig *config, void *uData);
gboolean stop_cloud_to_device_messaging (NvDsC2DContext* uCtx);

#ifdef __cplusplus
}
#endif
#endif