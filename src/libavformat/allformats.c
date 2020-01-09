/*
 * Register all the formats and protocols
 * Copyright (c) 2000, 2001, 2002 Fabrice Bellard
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */
#include "avformat.h"
#include "rtp_internal.h"
#include "rdt.h"

#define REGISTER_MUXER(X,x) { \
          extern AVOutputFormat x##_muxer; \
          if(ENABLE_##X##_MUXER)   av_register_output_format(&x##_muxer); }
#define REGISTER_DEMUXER(X,x) { \
          extern AVInputFormat x##_demuxer; \
          if(ENABLE_##X##_DEMUXER) av_register_input_format(&x##_demuxer); }
#define REGISTER_MUXDEMUX(X,x)  REGISTER_MUXER(X,x); REGISTER_DEMUXER(X,x)
#define REGISTER_PROTOCOL(X,x) { \
          extern URLProtocol x##_protocol; \
          if(ENABLE_##X##_PROTOCOL) register_protocol(&x##_protocol); }

/* If you do not call this function, then you can select exactly which
   formats you want to support */

/**
 * Initialize libavformat and register all the (de)muxers and protocols.
 */
void av_register_all(void)
{
    static int initialized;

    if (initialized)
        return;
    initialized = 1;

    avcodec_init();
    avcodec_register_all();

    /* (de)muxers */
    REGISTER_MUXDEMUX (MJPEG, mjpeg);
    REGISTER_MUXER    (NULL, null);
    REGISTER_MUXDEMUX (RAWVIDEO, rawvideo);
    REGISTER_MUXDEMUX (YUV4MPEGPIPE, yuv4mpegpipe);

    /* protocols */
    REGISTER_PROTOCOL (FILE, file);
}
