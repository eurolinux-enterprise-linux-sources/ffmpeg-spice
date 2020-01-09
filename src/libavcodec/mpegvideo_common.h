/*
 * The simplest mpeg encoder (well, it was the simplest!)
 * Copyright (c) 2000,2001 Fabrice Bellard.
 * Copyright (c) 2002-2004 Michael Niedermayer <michaelni@gmx.at>
 *
 * 4MV & hq & B-frame encoding stuff by Michael Niedermayer <michaelni@gmx.at>
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

/**
 * @file mpegvideo_common.h
 * The simplest mpeg encoder (well, it was the simplest!).
 */

#ifndef AVCODEC_MPEGVIDEO_COMMON_H
#define AVCODEC_MPEGVIDEO_COMMON_H

#include "avcodec.h"
#include "dsputil.h"
#include "mpegvideo.h"
#include "mjpegenc.h"
#include "faandct.h"
#include <limits.h>

int dct_quantize_c(MpegEncContext *s, DCTELEM *block, int n, int qscale, int *overflow);
int dct_quantize_trellis_c(MpegEncContext *s, DCTELEM *block, int n, int qscale, int *overflow);
void  denoise_dct_c(MpegEncContext *s, DCTELEM *block);
void copy_picture(Picture *dst, Picture *src);

/**
 * allocates a Picture
 * The pixels are allocated/set by calling get_buffer() if shared=0
 */
int alloc_picture(MpegEncContext *s, Picture *pic, int shared);

/**
 * sets the given MpegEncContext to common defaults (same for encoding and decoding).
 * the changed fields will not depend upon the prior state of the MpegEncContext.
 */
void MPV_common_defaults(MpegEncContext *s);

#endif
