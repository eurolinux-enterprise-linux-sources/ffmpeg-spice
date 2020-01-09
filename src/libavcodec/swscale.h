/*
 * Copyright (C) 2001-2003 Michael Niedermayer <michaelni@gmx.at>
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

#ifndef QSWSCALE_QSWSCALE_H
#define QSWSCALE_QSWSCALE_H

/**
 * @file swscale.h
 * @brief
 *     external api for the swscale stuff
 */

#include "libavutil/avutil.h"

#define LIBQSWSCALE_VERSION_MAJOR 0
#define LIBQSWSCALE_VERSION_MINOR 6
#define LIBQSWSCALE_VERSION_MICRO 1

#define LIBQSWSCALE_VERSION_INT  AV_VERSION_INT(LIBQSWSCALE_VERSION_MAJOR, \
                                               LIBQSWSCALE_VERSION_MINOR, \
                                               LIBQSWSCALE_VERSION_MICRO)
#define LIBQSWSCALE_VERSION      AV_VERSION(LIBQSWSCALE_VERSION_MAJOR, \
                                           LIBQSWSCALE_VERSION_MINOR, \
                                           LIBQSWSCALE_VERSION_MICRO)
#define LIBQSWSCALE_BUILD        LIBQSWSCALE_VERSION_INT

#define LIBQSWSCALE_IDENT        "SwS" AV_STRINGIFY(LIBQSWSCALE_VERSION)

/**
 * Returns the LIBQSWSCALE_VERSION_INT constant.
 */
unsigned swscale_version(void);

/* values for the flags, the stuff on the command line is different */
#define QSWS_FAST_BILINEAR     1
#define QSWS_BILINEAR          2
#define QSWS_BICUBIC           4
#define QSWS_X                 8
#define QSWS_POINT          0x10
#define QSWS_AREA           0x20
#define QSWS_BICUBLIN       0x40
#define QSWS_GAUSS          0x80
#define QSWS_SINC          0x100
#define QSWS_LANCZOS       0x200
#define QSWS_SPLINE        0x400

#define QSWS_SRC_V_CHR_DROP_MASK     0x30000
#define QSWS_SRC_V_CHR_DROP_SHIFT    16

#define QSWS_PARAM_DEFAULT           123456

#define QSWS_PRINT_INFO              0x1000

//the following 3 flags are not completely implemented
//internal chrominace subsampling info
#define QSWS_FULL_CHR_H_INT    0x2000
//input subsampling info
#define QSWS_FULL_CHR_H_INP    0x4000
#define QSWS_DIRECT_BGR        0x8000
#define QSWS_ACCURATE_RND      0x40000
#define QSWS_BITEXACT          0x80000

#define QSWS_CPU_CAPS_MMX      0x80000000
#define QSWS_CPU_CAPS_MMX2     0x20000000
#define QSWS_CPU_CAPS_3DNOW    0x40000000
#define QSWS_CPU_CAPS_ALTIVEC  0x10000000
#define QSWS_CPU_CAPS_BFIN     0x01000000

#define QSWS_MAX_REDUCE_CUTOFF 0.002

#define QSWS_CS_ITU709         1
#define QSWS_CS_FCC            4
#define QSWS_CS_ITU601         5
#define QSWS_CS_ITU624         5
#define QSWS_CS_SMPTE170M      5
#define QSWS_CS_SMPTE240M      7
#define QSWS_CS_DEFAULT        5



// when used for filters they must have an odd number of elements
// coeffs cannot be shared between vectors
typedef struct {
    double *coeff;
    int length;
} SwsVector;

// vectors can be shared
typedef struct {
    SwsVector *lumH;
    SwsVector *lumV;
    SwsVector *chrH;
    SwsVector *chrV;
} SwsFilter;

struct SwsContext;

void sws_freeContext(struct SwsContext *swsContext);

struct SwsContext *sws_getContext(int srcW, int srcH, int srcFormat, int dstW, int dstH, int dstFormat, int flags,
                                  SwsFilter *srcFilter, SwsFilter *dstFilter, double *param);
int sws_scale(struct SwsContext *context, uint8_t* src[], int srcStride[], int srcSliceY,
              int srcSliceH, uint8_t* dst[], int dstStride[]);

struct SwsContext *sws_getCachedContext(struct SwsContext *context,
                                        int srcW, int srcH, int srcFormat,
                                        int dstW, int dstH, int dstFormat, int flags,
                                        SwsFilter *srcFilter, SwsFilter *dstFilter, double *param);

#endif /* QSWSCALE_QSWSCALE_H */
