/*
 * DSP utils
 * Copyright (c) 2000, 2001 Fabrice Bellard.
 * Copyright (c) 2002-2004 Michael Niedermayer <michaelni@gmx.at>
 *
 * gmc & q-pel & 32/64 bit based MC by Michael Niedermayer <michaelni@gmx.at>
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
 * @file dsputil.c
 * DSP utils
 */

#include "avcodec.h"
#include "dsputil.h"
#include "simple_idct.h"
#include "faandct.h"
#include "faanidct.h"
#include "mpegvideo.h"

/* vorbis.c */
void vorbis_inverse_coupling(float *mag, float *ang, int blocksize);

/* ac3dec.c */
void ff_ac3_downmix_c(float (*samples)[256], float (*matrix)[2], int out_ch, int in_ch, int len);

/* flacenc.c */
void ff_flac_compute_autocorr(const int32_t *data, int len, int lag, double *autoc);

/* pngdec.c */
void ff_add_png_paeth_prediction(uint8_t *dst, uint8_t *src, uint8_t *top, int w, int bpp);

uint8_t ff_cropTbl[256 + 2 * MAX_NEG_CROP] = {0, };
uint32_t ff_squareTbl[512] = {0, };

// 0x7f7f7f7f or 0x7f7f7f7f7f7f7f7f or whatever, depending on the cpu's native arithmetic size
#define pb_7f (~0UL/255 * 0x7f)
#define pb_80 (~0UL/255 * 0x80)

const uint8_t ff_zigzag_direct[64] = {
    0,   1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

/* Specific zigzag scan for 248 idct. NOTE that unlike the
   specification, we interleave the fields */
const uint8_t ff_zigzag248_direct[64] = {
     0,  8,  1,  9, 16, 24,  2, 10,
    17, 25, 32, 40, 48, 56, 33, 41,
    18, 26,  3, 11,  4, 12, 19, 27,
    34, 42, 49, 57, 50, 58, 35, 43,
    20, 28,  5, 13,  6, 14, 21, 29,
    36, 44, 51, 59, 52, 60, 37, 45,
    22, 30,  7, 15, 23, 31, 38, 46,
    53, 61, 54, 62, 39, 47, 55, 63,
};

/* not permutated inverse zigzag_direct + 1 for MMX quantizer */
DECLARE_ALIGNED_8(uint16_t, inv_zigzag_direct16[64]) = {0, };

const uint8_t ff_alternate_horizontal_scan[64] = {
    0,  1,   2,  3,  8,  9, 16, 17,
    10, 11,  4,  5,  6,  7, 15, 14,
    13, 12, 19, 18, 24, 25, 32, 33,
    26, 27, 20, 21, 22, 23, 28, 29,
    30, 31, 34, 35, 40, 41, 48, 49,
    42, 43, 36, 37, 38, 39, 44, 45,
    46, 47, 50, 51, 56, 57, 58, 59,
    52, 53, 54, 55, 60, 61, 62, 63,
};

const uint8_t ff_alternate_vertical_scan[64] = {
    0,  8,  16, 24,  1,  9,  2, 10,
    17, 25, 32, 40, 48, 56, 57, 49,
    41, 33, 26, 18,  3, 11,  4, 12,
    19, 27, 34, 42, 50, 58, 35, 43,
    51, 59, 20, 28,  5, 13,  6, 14,
    21, 29, 36, 44, 52, 60, 37, 45,
    53, 61, 22, 30,  7, 15, 23, 31,
    38, 46, 54, 62, 39, 47, 55, 63,
};

/* a*inverse[b]>>32 == a/b for all 0<=a<=65536 && 2<=b<=255 */
const uint32_t ff_inverse[256]={
         0, 4294967295U,2147483648U,1431655766, 1073741824,  858993460,  715827883,  613566757,
 536870912,  477218589,  429496730,  390451573,  357913942,  330382100,  306783379,  286331154,
 268435456,  252645136,  238609295,  226050911,  214748365,  204522253,  195225787,  186737709,
 178956971,  171798692,  165191050,  159072863,  153391690,  148102321,  143165577,  138547333,
 134217728,  130150525,  126322568,  122713352,  119304648,  116080198,  113025456,  110127367,
 107374183,  104755300,  102261127,   99882961,   97612894,   95443718,   93368855,   91382283,
  89478486,   87652394,   85899346,   84215046,   82595525,   81037119,   79536432,   78090315,
  76695845,   75350304,   74051161,   72796056,   71582789,   70409300,   69273667,   68174085,
  67108864,   66076420,   65075263,   64103990,   63161284,   62245903,   61356676,   60492498,
  59652324,   58835169,   58040099,   57266231,   56512728,   55778797,   55063684,   54366675,
  53687092,   53024288,   52377650,   51746594,   51130564,   50529028,   49941481,   49367441,
  48806447,   48258060,   47721859,   47197443,   46684428,   46182445,   45691142,   45210183,
  44739243,   44278014,   43826197,   43383509,   42949673,   42524429,   42107523,   41698712,
  41297763,   40904451,   40518560,   40139882,   39768216,   39403370,   39045158,   38693400,
  38347923,   38008561,   37675152,   37347542,   37025581,   36709123,   36398028,   36092163,
  35791395,   35495598,   35204650,   34918434,   34636834,   34359739,   34087043,   33818641,
  33554432,   33294321,   33038210,   32786010,   32537632,   32292988,   32051995,   31814573,
  31580642,   31350127,   31122952,   30899046,   30678338,   30460761,   30246249,   30034737,
  29826162,   29620465,   29417585,   29217465,   29020050,   28825284,   28633116,   28443493,
  28256364,   28071682,   27889399,   27709467,   27531842,   27356480,   27183338,   27012373,
  26843546,   26676816,   26512144,   26349493,   26188825,   26030105,   25873297,   25718368,
  25565282,   25414008,   25264514,   25116768,   24970741,   24826401,   24683721,   24542671,
  24403224,   24265352,   24129030,   23994231,   23860930,   23729102,   23598722,   23469767,
  23342214,   23216040,   23091223,   22967740,   22845571,   22724695,   22605092,   22486740,
  22369622,   22253717,   22139007,   22025474,   21913099,   21801865,   21691755,   21582751,
  21474837,   21367997,   21262215,   21157475,   21053762,   20951060,   20849356,   20748635,
  20648882,   20550083,   20452226,   20355296,   20259280,   20164166,   20069941,   19976593,
  19884108,   19792477,   19701685,   19611723,   19522579,   19434242,   19346700,   19259944,
  19173962,   19088744,   19004281,   18920561,   18837576,   18755316,   18673771,   18592933,
  18512791,   18433337,   18354562,   18276457,   18199014,   18122225,   18046082,   17970575,
  17895698,   17821442,   17747799,   17674763,   17602325,   17530479,   17459217,   17388532,
  17318417,   17248865,   17179870,   17111424,   17043522,   16976156,   16909321,   16843010,
};

/* Input permutation for the simple_idct_mmx */
static const uint8_t simple_mmx_permutation[64]={
        0x00, 0x08, 0x04, 0x09, 0x01, 0x0C, 0x05, 0x0D,
        0x10, 0x18, 0x14, 0x19, 0x11, 0x1C, 0x15, 0x1D,
        0x20, 0x28, 0x24, 0x29, 0x21, 0x2C, 0x25, 0x2D,
        0x12, 0x1A, 0x16, 0x1B, 0x13, 0x1E, 0x17, 0x1F,
        0x02, 0x0A, 0x06, 0x0B, 0x03, 0x0E, 0x07, 0x0F,
        0x30, 0x38, 0x34, 0x39, 0x31, 0x3C, 0x35, 0x3D,
        0x22, 0x2A, 0x26, 0x2B, 0x23, 0x2E, 0x27, 0x2F,
        0x32, 0x3A, 0x36, 0x3B, 0x33, 0x3E, 0x37, 0x3F,
};

static const uint8_t idct_sse2_row_perm[8] = {0, 4, 1, 5, 2, 6, 3, 7};

void ff_init_scantable(uint8_t *permutation, ScanTable *st, const uint8_t *src_scantable){
    int i;
    int end;

    st->scantable= src_scantable;

    for(i=0; i<64; i++){
        int j;
        j = src_scantable[i];
        st->permutated[i] = permutation[j];
#ifdef ARCH_POWERPC
        st->inverse[j] = i;
#endif
    }

    end=-1;
    for(i=0; i<64; i++){
        int j;
        j = st->permutated[i];
        if(j>end) end=j;
        st->raster_end[i]= end;
    }
}

static int pix_sum_c(uint8_t * pix, int line_size)
{
    int s, i, j;

    s = 0;
    for (i = 0; i < 16; i++) {
        for (j = 0; j < 16; j += 8) {
            s += pix[0];
            s += pix[1];
            s += pix[2];
            s += pix[3];
            s += pix[4];
            s += pix[5];
            s += pix[6];
            s += pix[7];
            pix += 8;
        }
        pix += line_size - 16;
    }
    return s;
}

static int pix_norm1_c(uint8_t * pix, int line_size)
{
    int s, i, j;
    uint32_t *sq = ff_squareTbl + 256;

    s = 0;
    for (i = 0; i < 16; i++) {
        for (j = 0; j < 16; j += 8) {
#if 0
            s += sq[pix[0]];
            s += sq[pix[1]];
            s += sq[pix[2]];
            s += sq[pix[3]];
            s += sq[pix[4]];
            s += sq[pix[5]];
            s += sq[pix[6]];
            s += sq[pix[7]];
#else
#if LONG_MAX > 2147483647
            register uint64_t x=*(uint64_t*)pix;
            s += sq[x&0xff];
            s += sq[(x>>8)&0xff];
            s += sq[(x>>16)&0xff];
            s += sq[(x>>24)&0xff];
            s += sq[(x>>32)&0xff];
            s += sq[(x>>40)&0xff];
            s += sq[(x>>48)&0xff];
            s += sq[(x>>56)&0xff];
#else
            register uint32_t x=*(uint32_t*)pix;
            s += sq[x&0xff];
            s += sq[(x>>8)&0xff];
            s += sq[(x>>16)&0xff];
            s += sq[(x>>24)&0xff];
            x=*(uint32_t*)(pix+4);
            s += sq[x&0xff];
            s += sq[(x>>8)&0xff];
            s += sq[(x>>16)&0xff];
            s += sq[(x>>24)&0xff];
#endif
#endif
            pix += 8;
        }
        pix += line_size - 16;
    }
    return s;
}

static void bswap_buf(uint32_t *dst, const uint32_t *src, int w){
    int i;

    for(i=0; i+8<=w; i+=8){
        dst[i+0]= bswap_32(src[i+0]);
        dst[i+1]= bswap_32(src[i+1]);
        dst[i+2]= bswap_32(src[i+2]);
        dst[i+3]= bswap_32(src[i+3]);
        dst[i+4]= bswap_32(src[i+4]);
        dst[i+5]= bswap_32(src[i+5]);
        dst[i+6]= bswap_32(src[i+6]);
        dst[i+7]= bswap_32(src[i+7]);
    }
    for(;i<w; i++){
        dst[i+0]= bswap_32(src[i+0]);
    }
}

static int sse4_c(void *v, uint8_t * pix1, uint8_t * pix2, int line_size, int h)
{
    int s, i;
    uint32_t *sq = ff_squareTbl + 256;

    s = 0;
    for (i = 0; i < h; i++) {
        s += sq[pix1[0] - pix2[0]];
        s += sq[pix1[1] - pix2[1]];
        s += sq[pix1[2] - pix2[2]];
        s += sq[pix1[3] - pix2[3]];
        pix1 += line_size;
        pix2 += line_size;
    }
    return s;
}

static int sse8_c(void *v, uint8_t * pix1, uint8_t * pix2, int line_size, int h)
{
    int s, i;
    uint32_t *sq = ff_squareTbl + 256;

    s = 0;
    for (i = 0; i < h; i++) {
        s += sq[pix1[0] - pix2[0]];
        s += sq[pix1[1] - pix2[1]];
        s += sq[pix1[2] - pix2[2]];
        s += sq[pix1[3] - pix2[3]];
        s += sq[pix1[4] - pix2[4]];
        s += sq[pix1[5] - pix2[5]];
        s += sq[pix1[6] - pix2[6]];
        s += sq[pix1[7] - pix2[7]];
        pix1 += line_size;
        pix2 += line_size;
    }
    return s;
}

static int sse16_c(void *v, uint8_t *pix1, uint8_t *pix2, int line_size, int h)
{
    int s, i;
    uint32_t *sq = ff_squareTbl + 256;

    s = 0;
    for (i = 0; i < h; i++) {
        s += sq[pix1[ 0] - pix2[ 0]];
        s += sq[pix1[ 1] - pix2[ 1]];
        s += sq[pix1[ 2] - pix2[ 2]];
        s += sq[pix1[ 3] - pix2[ 3]];
        s += sq[pix1[ 4] - pix2[ 4]];
        s += sq[pix1[ 5] - pix2[ 5]];
        s += sq[pix1[ 6] - pix2[ 6]];
        s += sq[pix1[ 7] - pix2[ 7]];
        s += sq[pix1[ 8] - pix2[ 8]];
        s += sq[pix1[ 9] - pix2[ 9]];
        s += sq[pix1[10] - pix2[10]];
        s += sq[pix1[11] - pix2[11]];
        s += sq[pix1[12] - pix2[12]];
        s += sq[pix1[13] - pix2[13]];
        s += sq[pix1[14] - pix2[14]];
        s += sq[pix1[15] - pix2[15]];

        pix1 += line_size;
        pix2 += line_size;
    }
    return s;
}


#ifdef CONFIG_SNOW_ENCODER //dwt is in snow.c
static inline int w_c(void *v, uint8_t * pix1, uint8_t * pix2, int line_size, int w, int h, int type){
    int s, i, j;
    const int dec_count= w==8 ? 3 : 4;
    int tmp[32*32];
    int level, ori;
    static const int scale[2][2][4][4]={
      {
        {
            // 9/7 8x8 dec=3
            {268, 239, 239, 213},
            {  0, 224, 224, 152},
            {  0, 135, 135, 110},
        },{
            // 9/7 16x16 or 32x32 dec=4
            {344, 310, 310, 280},
            {  0, 320, 320, 228},
            {  0, 175, 175, 136},
            {  0, 129, 129, 102},
        }
      },{
        {
            // 5/3 8x8 dec=3
            {275, 245, 245, 218},
            {  0, 230, 230, 156},
            {  0, 138, 138, 113},
        },{
            // 5/3 16x16 or 32x32 dec=4
            {352, 317, 317, 286},
            {  0, 328, 328, 233},
            {  0, 180, 180, 140},
            {  0, 132, 132, 105},
        }
      }
    };

    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j+=4) {
            tmp[32*i+j+0] = (pix1[j+0] - pix2[j+0])<<4;
            tmp[32*i+j+1] = (pix1[j+1] - pix2[j+1])<<4;
            tmp[32*i+j+2] = (pix1[j+2] - pix2[j+2])<<4;
            tmp[32*i+j+3] = (pix1[j+3] - pix2[j+3])<<4;
        }
        pix1 += line_size;
        pix2 += line_size;
    }

    ff_spatial_dwt(tmp, w, h, 32, type, dec_count);

    s=0;
    assert(w==h);
    for(level=0; level<dec_count; level++){
        for(ori= level ? 1 : 0; ori<4; ori++){
            int size= w>>(dec_count-level);
            int sx= (ori&1) ? size : 0;
            int stride= 32<<(dec_count-level);
            int sy= (ori&2) ? stride>>1 : 0;

            for(i=0; i<size; i++){
                for(j=0; j<size; j++){
                    int v= tmp[sx + sy + i*stride + j] * scale[type][dec_count-3][level][ori];
                    s += FFABS(v);
                }
            }
        }
    }
    assert(s>=0);
    return s>>9;
}

static int w53_8_c(void *v, uint8_t * pix1, uint8_t * pix2, int line_size, int h){
    return w_c(v, pix1, pix2, line_size,  8, h, 1);
}

static int w97_8_c(void *v, uint8_t * pix1, uint8_t * pix2, int line_size, int h){
    return w_c(v, pix1, pix2, line_size,  8, h, 0);
}

static int w53_16_c(void *v, uint8_t * pix1, uint8_t * pix2, int line_size, int h){
    return w_c(v, pix1, pix2, line_size, 16, h, 1);
}

static int w97_16_c(void *v, uint8_t * pix1, uint8_t * pix2, int line_size, int h){
    return w_c(v, pix1, pix2, line_size, 16, h, 0);
}

int w53_32_c(void *v, uint8_t * pix1, uint8_t * pix2, int line_size, int h){
    return w_c(v, pix1, pix2, line_size, 32, h, 1);
}

int w97_32_c(void *v, uint8_t * pix1, uint8_t * pix2, int line_size, int h){
    return w_c(v, pix1, pix2, line_size, 32, h, 0);
}
#endif

/* draw the edges of width 'w' of an image of size width, height */
//FIXME check that this is ok for mpeg4 interlaced
static void draw_edges_c(uint8_t *buf, int wrap, int width, int height, int w)
{
    uint8_t *ptr, *last_line;
    int i;

    last_line = buf + (height - 1) * wrap;
    for(i=0;i<w;i++) {
        /* top and bottom */
        memcpy(buf - (i + 1) * wrap, buf, width);
        memcpy(last_line + (i + 1) * wrap, last_line, width);
    }
    /* left and right */
    ptr = buf;
    for(i=0;i<height;i++) {
        memset(ptr - w, ptr[0], w);
        memset(ptr + width, ptr[width-1], w);
        ptr += wrap;
    }
    /* corners */
    for(i=0;i<w;i++) {
        memset(buf - (i + 1) * wrap - w, buf[0], w); /* top left */
        memset(buf - (i + 1) * wrap + width, buf[width-1], w); /* top right */
        memset(last_line + (i + 1) * wrap - w, last_line[0], w); /* top left */
        memset(last_line + (i + 1) * wrap + width, last_line[width-1], w); /* top right */
    }
}

/**
 * Copies a rectangular area of samples to a temporary buffer and replicates the boarder samples.
 * @param buf destination buffer
 * @param src source buffer
 * @param linesize number of bytes between 2 vertically adjacent samples in both the source and destination buffers
 * @param block_w width of block
 * @param block_h height of block
 * @param src_x x coordinate of the top left sample of the block in the source buffer
 * @param src_y y coordinate of the top left sample of the block in the source buffer
 * @param w width of the source buffer
 * @param h height of the source buffer
 */
void ff_emulated_edge_mc(uint8_t *buf, uint8_t *src, int linesize, int block_w, int block_h,
                                    int src_x, int src_y, int w, int h){
    int x, y;
    int start_y, start_x, end_y, end_x;

    if(src_y>= h){
        src+= (h-1-src_y)*linesize;
        src_y=h-1;
    }else if(src_y<=-block_h){
        src+= (1-block_h-src_y)*linesize;
        src_y=1-block_h;
    }
    if(src_x>= w){
        src+= (w-1-src_x);
        src_x=w-1;
    }else if(src_x<=-block_w){
        src+= (1-block_w-src_x);
        src_x=1-block_w;
    }

    start_y= FFMAX(0, -src_y);
    start_x= FFMAX(0, -src_x);
    end_y= FFMIN(block_h, h-src_y);
    end_x= FFMIN(block_w, w-src_x);

    // copy existing part
    for(y=start_y; y<end_y; y++){
        for(x=start_x; x<end_x; x++){
            buf[x + y*linesize]= src[x + y*linesize];
        }
    }

    //top
    for(y=0; y<start_y; y++){
        for(x=start_x; x<end_x; x++){
            buf[x + y*linesize]= buf[x + start_y*linesize];
        }
    }

    //bottom
    for(y=end_y; y<block_h; y++){
        for(x=start_x; x<end_x; x++){
            buf[x + y*linesize]= buf[x + (end_y-1)*linesize];
        }
    }

    for(y=0; y<block_h; y++){
       //left
        for(x=0; x<start_x; x++){
            buf[x + y*linesize]= buf[start_x + y*linesize];
        }

       //right
        for(x=end_x; x<block_w; x++){
            buf[x + y*linesize]= buf[end_x - 1 + y*linesize];
        }
    }
}

static void get_pixels_c(DCTELEM *restrict block, const uint8_t *pixels, int line_size)
{
    int i;

    /* read the pixels */
    for(i=0;i<8;i++) {
        block[0] = pixels[0];
        block[1] = pixels[1];
        block[2] = pixels[2];
        block[3] = pixels[3];
        block[4] = pixels[4];
        block[5] = pixels[5];
        block[6] = pixels[6];
        block[7] = pixels[7];
        pixels += line_size;
        block += 8;
    }
}

static void diff_pixels_c(DCTELEM *restrict block, const uint8_t *s1,
                          const uint8_t *s2, int stride){
    int i;

    /* read the pixels */
    for(i=0;i<8;i++) {
        block[0] = s1[0] - s2[0];
        block[1] = s1[1] - s2[1];
        block[2] = s1[2] - s2[2];
        block[3] = s1[3] - s2[3];
        block[4] = s1[4] - s2[4];
        block[5] = s1[5] - s2[5];
        block[6] = s1[6] - s2[6];
        block[7] = s1[7] - s2[7];
        s1 += stride;
        s2 += stride;
        block += 8;
    }
}


static void put_pixels_clamped_c(const DCTELEM *block, uint8_t *restrict pixels,
                                 int line_size)
{
    int i;
    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;

    /* read the pixels */
    for(i=0;i<8;i++) {
        pixels[0] = cm[block[0]];
        pixels[1] = cm[block[1]];
        pixels[2] = cm[block[2]];
        pixels[3] = cm[block[3]];
        pixels[4] = cm[block[4]];
        pixels[5] = cm[block[5]];
        pixels[6] = cm[block[6]];
        pixels[7] = cm[block[7]];

        pixels += line_size;
        block += 8;
    }
}

static void put_pixels_clamped4_c(const DCTELEM *block, uint8_t *restrict pixels,
                                 int line_size)
{
    int i;
    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;

    /* read the pixels */
    for(i=0;i<4;i++) {
        pixels[0] = cm[block[0]];
        pixels[1] = cm[block[1]];
        pixels[2] = cm[block[2]];
        pixels[3] = cm[block[3]];

        pixels += line_size;
        block += 8;
    }
}

static void put_pixels_clamped2_c(const DCTELEM *block, uint8_t *restrict pixels,
                                 int line_size)
{
    int i;
    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;

    /* read the pixels */
    for(i=0;i<2;i++) {
        pixels[0] = cm[block[0]];
        pixels[1] = cm[block[1]];

        pixels += line_size;
        block += 8;
    }
}

static void put_signed_pixels_clamped_c(const DCTELEM *block,
                                        uint8_t *restrict pixels,
                                        int line_size)
{
    int i, j;

    for (i = 0; i < 8; i++) {
        for (j = 0; j < 8; j++) {
            if (*block < -128)
                *pixels = 0;
            else if (*block > 127)
                *pixels = 255;
            else
                *pixels = (uint8_t)(*block + 128);
            block++;
            pixels++;
        }
        pixels += (line_size - 8);
    }
}

static void add_pixels_clamped_c(const DCTELEM *block, uint8_t *restrict pixels,
                          int line_size)
{
    int i;
    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;

    /* read the pixels */
    for(i=0;i<8;i++) {
        pixels[0] = cm[pixels[0] + block[0]];
        pixels[1] = cm[pixels[1] + block[1]];
        pixels[2] = cm[pixels[2] + block[2]];
        pixels[3] = cm[pixels[3] + block[3]];
        pixels[4] = cm[pixels[4] + block[4]];
        pixels[5] = cm[pixels[5] + block[5]];
        pixels[6] = cm[pixels[6] + block[6]];
        pixels[7] = cm[pixels[7] + block[7]];
        pixels += line_size;
        block += 8;
    }
}

static void add_pixels_clamped4_c(const DCTELEM *block, uint8_t *restrict pixels,
                          int line_size)
{
    int i;
    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;

    /* read the pixels */
    for(i=0;i<4;i++) {
        pixels[0] = cm[pixels[0] + block[0]];
        pixels[1] = cm[pixels[1] + block[1]];
        pixels[2] = cm[pixels[2] + block[2]];
        pixels[3] = cm[pixels[3] + block[3]];
        pixels += line_size;
        block += 8;
    }
}

static void add_pixels_clamped2_c(const DCTELEM *block, uint8_t *restrict pixels,
                          int line_size)
{
    int i;
    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;

    /* read the pixels */
    for(i=0;i<2;i++) {
        pixels[0] = cm[pixels[0] + block[0]];
        pixels[1] = cm[pixels[1] + block[1]];
        pixels += line_size;
        block += 8;
    }
}

static void add_pixels8_c(uint8_t *restrict pixels, DCTELEM *block, int line_size)
{
    int i;
    for(i=0;i<8;i++) {
        pixels[0] += block[0];
        pixels[1] += block[1];
        pixels[2] += block[2];
        pixels[3] += block[3];
        pixels[4] += block[4];
        pixels[5] += block[5];
        pixels[6] += block[6];
        pixels[7] += block[7];
        pixels += line_size;
        block += 8;
    }
}

static void add_pixels4_c(uint8_t *restrict pixels, DCTELEM *block, int line_size)
{
    int i;
    for(i=0;i<4;i++) {
        pixels[0] += block[0];
        pixels[1] += block[1];
        pixels[2] += block[2];
        pixels[3] += block[3];
        pixels += line_size;
        block += 4;
    }
}

static int sum_abs_dctelem_c(DCTELEM *block)
{
    int sum=0, i;
    for(i=0; i<64; i++)
        sum+= FFABS(block[i]);
    return sum;
}

#if 0

#define PIXOP2(OPNAME, OP) \
static void OPNAME ## _pixels(uint8_t *block, const uint8_t *pixels, int line_size, int h)\
{\
    int i;\
    for(i=0; i<h; i++){\
        OP(*((uint64_t*)block), AV_RN64(pixels));\
        pixels+=line_size;\
        block +=line_size;\
    }\
}\
\
static void OPNAME ## _no_rnd_pixels_x2_c(uint8_t *block, const uint8_t *pixels, int line_size, int h)\
{\
    int i;\
    for(i=0; i<h; i++){\
        const uint64_t a= AV_RN64(pixels  );\
        const uint64_t b= AV_RN64(pixels+1);\
        OP(*((uint64_t*)block), (a&b) + (((a^b)&0xFEFEFEFEFEFEFEFEULL)>>1));\
        pixels+=line_size;\
        block +=line_size;\
    }\
}\
\
static void OPNAME ## _pixels_x2_c(uint8_t *block, const uint8_t *pixels, int line_size, int h)\
{\
    int i;\
    for(i=0; i<h; i++){\
        const uint64_t a= AV_RN64(pixels  );\
        const uint64_t b= AV_RN64(pixels+1);\
        OP(*((uint64_t*)block), (a|b) - (((a^b)&0xFEFEFEFEFEFEFEFEULL)>>1));\
        pixels+=line_size;\
        block +=line_size;\
    }\
}\
\
static void OPNAME ## _no_rnd_pixels_y2_c(uint8_t *block, const uint8_t *pixels, int line_size, int h)\
{\
    int i;\
    for(i=0; i<h; i++){\
        const uint64_t a= AV_RN64(pixels          );\
        const uint64_t b= AV_RN64(pixels+line_size);\
        OP(*((uint64_t*)block), (a&b) + (((a^b)&0xFEFEFEFEFEFEFEFEULL)>>1));\
        pixels+=line_size;\
        block +=line_size;\
    }\
}\
\
static void OPNAME ## _pixels_y2_c(uint8_t *block, const uint8_t *pixels, int line_size, int h)\
{\
    int i;\
    for(i=0; i<h; i++){\
        const uint64_t a= AV_RN64(pixels          );\
        const uint64_t b= AV_RN64(pixels+line_size);\
        OP(*((uint64_t*)block), (a|b) - (((a^b)&0xFEFEFEFEFEFEFEFEULL)>>1));\
        pixels+=line_size;\
        block +=line_size;\
    }\
}\
\
static void OPNAME ## _pixels_xy2_c(uint8_t *block, const uint8_t *pixels, int line_size, int h)\
{\
        int i;\
        const uint64_t a= AV_RN64(pixels  );\
        const uint64_t b= AV_RN64(pixels+1);\
        uint64_t l0=  (a&0x0303030303030303ULL)\
                    + (b&0x0303030303030303ULL)\
                    + 0x0202020202020202ULL;\
        uint64_t h0= ((a&0xFCFCFCFCFCFCFCFCULL)>>2)\
                   + ((b&0xFCFCFCFCFCFCFCFCULL)>>2);\
        uint64_t l1,h1;\
\
        pixels+=line_size;\
        for(i=0; i<h; i+=2){\
            uint64_t a= AV_RN64(pixels  );\
            uint64_t b= AV_RN64(pixels+1);\
            l1=  (a&0x0303030303030303ULL)\
               + (b&0x0303030303030303ULL);\
            h1= ((a&0xFCFCFCFCFCFCFCFCULL)>>2)\
              + ((b&0xFCFCFCFCFCFCFCFCULL)>>2);\
            OP(*((uint64_t*)block), h0+h1+(((l0+l1)>>2)&0x0F0F0F0F0F0F0F0FULL));\
            pixels+=line_size;\
            block +=line_size;\
            a= AV_RN64(pixels  );\
            b= AV_RN64(pixels+1);\
            l0=  (a&0x0303030303030303ULL)\
               + (b&0x0303030303030303ULL)\
               + 0x0202020202020202ULL;\
            h0= ((a&0xFCFCFCFCFCFCFCFCULL)>>2)\
              + ((b&0xFCFCFCFCFCFCFCFCULL)>>2);\
            OP(*((uint64_t*)block), h0+h1+(((l0+l1)>>2)&0x0F0F0F0F0F0F0F0FULL));\
            pixels+=line_size;\
            block +=line_size;\
        }\
}\
\
static void OPNAME ## _no_rnd_pixels_xy2_c(uint8_t *block, const uint8_t *pixels, int line_size, int h)\
{\
        int i;\
        const uint64_t a= AV_RN64(pixels  );\
        const uint64_t b= AV_RN64(pixels+1);\
        uint64_t l0=  (a&0x0303030303030303ULL)\
                    + (b&0x0303030303030303ULL)\
                    + 0x0101010101010101ULL;\
        uint64_t h0= ((a&0xFCFCFCFCFCFCFCFCULL)>>2)\
                   + ((b&0xFCFCFCFCFCFCFCFCULL)>>2);\
        uint64_t l1,h1;\
\
        pixels+=line_size;\
        for(i=0; i<h; i+=2){\
            uint64_t a= AV_RN64(pixels  );\
            uint64_t b= AV_RN64(pixels+1);\
            l1=  (a&0x0303030303030303ULL)\
               + (b&0x0303030303030303ULL);\
            h1= ((a&0xFCFCFCFCFCFCFCFCULL)>>2)\
              + ((b&0xFCFCFCFCFCFCFCFCULL)>>2);\
            OP(*((uint64_t*)block), h0+h1+(((l0+l1)>>2)&0x0F0F0F0F0F0F0F0FULL));\
            pixels+=line_size;\
            block +=line_size;\
            a= AV_RN64(pixels  );\
            b= AV_RN64(pixels+1);\
            l0=  (a&0x0303030303030303ULL)\
               + (b&0x0303030303030303ULL)\
               + 0x0101010101010101ULL;\
            h0= ((a&0xFCFCFCFCFCFCFCFCULL)>>2)\
              + ((b&0xFCFCFCFCFCFCFCFCULL)>>2);\
            OP(*((uint64_t*)block), h0+h1+(((l0+l1)>>2)&0x0F0F0F0F0F0F0F0FULL));\
            pixels+=line_size;\
            block +=line_size;\
        }\
}\
\
CALL_2X_PIXELS(OPNAME ## _pixels16_c    , OPNAME ## _pixels_c    , 8)\
CALL_2X_PIXELS(OPNAME ## _pixels16_x2_c , OPNAME ## _pixels_x2_c , 8)\
CALL_2X_PIXELS(OPNAME ## _pixels16_y2_c , OPNAME ## _pixels_y2_c , 8)\
CALL_2X_PIXELS(OPNAME ## _pixels16_xy2_c, OPNAME ## _pixels_xy2_c, 8)\
CALL_2X_PIXELS(OPNAME ## _no_rnd_pixels16_x2_c , OPNAME ## _no_rnd_pixels_x2_c , 8)\
CALL_2X_PIXELS(OPNAME ## _no_rnd_pixels16_y2_c , OPNAME ## _no_rnd_pixels_y2_c , 8)\
CALL_2X_PIXELS(OPNAME ## _no_rnd_pixels16_xy2_c, OPNAME ## _no_rnd_pixels_xy2_c, 8)

#define op_avg(a, b) a = ( ((a)|(b)) - ((((a)^(b))&0xFEFEFEFEFEFEFEFEULL)>>1) )
#else // 64 bit variant

#define PIXOP2(OPNAME, OP) \
static void OPNAME ## _pixels2_c(uint8_t *block, const uint8_t *pixels, int line_size, int h){\
    int i;\
    for(i=0; i<h; i++){\
        OP(*((uint16_t*)(block  )), AV_RN16(pixels  ));\
        pixels+=line_size;\
        block +=line_size;\
    }\
}\
static void OPNAME ## _pixels4_c(uint8_t *block, const uint8_t *pixels, int line_size, int h){\
    int i;\
    for(i=0; i<h; i++){\
        OP(*((uint32_t*)(block  )), AV_RN32(pixels  ));\
        pixels+=line_size;\
        block +=line_size;\
    }\
}\
static void OPNAME ## _pixels8_c(uint8_t *block, const uint8_t *pixels, int line_size, int h){\
    int i;\
    for(i=0; i<h; i++){\
        OP(*((uint32_t*)(block  )), AV_RN32(pixels  ));\
        OP(*((uint32_t*)(block+4)), AV_RN32(pixels+4));\
        pixels+=line_size;\
        block +=line_size;\
    }\
}\
static inline void OPNAME ## _no_rnd_pixels8_c(uint8_t *block, const uint8_t *pixels, int line_size, int h){\
    OPNAME ## _pixels8_c(block, pixels, line_size, h);\
}\
\
static inline void OPNAME ## _no_rnd_pixels8_l2(uint8_t *dst, const uint8_t *src1, const uint8_t *src2, int dst_stride, \
                                                int src_stride1, int src_stride2, int h){\
    int i;\
    for(i=0; i<h; i++){\
        uint32_t a,b;\
        a= AV_RN32(&src1[i*src_stride1  ]);\
        b= AV_RN32(&src2[i*src_stride2  ]);\
        OP(*((uint32_t*)&dst[i*dst_stride  ]), no_rnd_avg32(a, b));\
        a= AV_RN32(&src1[i*src_stride1+4]);\
        b= AV_RN32(&src2[i*src_stride2+4]);\
        OP(*((uint32_t*)&dst[i*dst_stride+4]), no_rnd_avg32(a, b));\
    }\
}\
\
static inline void OPNAME ## _pixels8_l2(uint8_t *dst, const uint8_t *src1, const uint8_t *src2, int dst_stride, \
                                                int src_stride1, int src_stride2, int h){\
    int i;\
    for(i=0; i<h; i++){\
        uint32_t a,b;\
        a= AV_RN32(&src1[i*src_stride1  ]);\
        b= AV_RN32(&src2[i*src_stride2  ]);\
        OP(*((uint32_t*)&dst[i*dst_stride  ]), rnd_avg32(a, b));\
        a= AV_RN32(&src1[i*src_stride1+4]);\
        b= AV_RN32(&src2[i*src_stride2+4]);\
        OP(*((uint32_t*)&dst[i*dst_stride+4]), rnd_avg32(a, b));\
    }\
}\
\
static inline void OPNAME ## _pixels4_l2(uint8_t *dst, const uint8_t *src1, const uint8_t *src2, int dst_stride, \
                                                int src_stride1, int src_stride2, int h){\
    int i;\
    for(i=0; i<h; i++){\
        uint32_t a,b;\
        a= AV_RN32(&src1[i*src_stride1  ]);\
        b= AV_RN32(&src2[i*src_stride2  ]);\
        OP(*((uint32_t*)&dst[i*dst_stride  ]), rnd_avg32(a, b));\
    }\
}\
\
static inline void OPNAME ## _pixels2_l2(uint8_t *dst, const uint8_t *src1, const uint8_t *src2, int dst_stride, \
                                                int src_stride1, int src_stride2, int h){\
    int i;\
    for(i=0; i<h; i++){\
        uint32_t a,b;\
        a= AV_RN16(&src1[i*src_stride1  ]);\
        b= AV_RN16(&src2[i*src_stride2  ]);\
        OP(*((uint16_t*)&dst[i*dst_stride  ]), rnd_avg32(a, b));\
    }\
}\
\
static inline void OPNAME ## _pixels16_l2(uint8_t *dst, const uint8_t *src1, const uint8_t *src2, int dst_stride, \
                                                int src_stride1, int src_stride2, int h){\
    OPNAME ## _pixels8_l2(dst  , src1  , src2  , dst_stride, src_stride1, src_stride2, h);\
    OPNAME ## _pixels8_l2(dst+8, src1+8, src2+8, dst_stride, src_stride1, src_stride2, h);\
}\
\
static inline void OPNAME ## _no_rnd_pixels16_l2(uint8_t *dst, const uint8_t *src1, const uint8_t *src2, int dst_stride, \
                                                int src_stride1, int src_stride2, int h){\
    OPNAME ## _no_rnd_pixels8_l2(dst  , src1  , src2  , dst_stride, src_stride1, src_stride2, h);\
    OPNAME ## _no_rnd_pixels8_l2(dst+8, src1+8, src2+8, dst_stride, src_stride1, src_stride2, h);\
}\
\
static inline void OPNAME ## _no_rnd_pixels8_x2_c(uint8_t *block, const uint8_t *pixels, int line_size, int h){\
    OPNAME ## _no_rnd_pixels8_l2(block, pixels, pixels+1, line_size, line_size, line_size, h);\
}\
\
static inline void OPNAME ## _pixels8_x2_c(uint8_t *block, const uint8_t *pixels, int line_size, int h){\
    OPNAME ## _pixels8_l2(block, pixels, pixels+1, line_size, line_size, line_size, h);\
}\
\
static inline void OPNAME ## _no_rnd_pixels8_y2_c(uint8_t *block, const uint8_t *pixels, int line_size, int h){\
    OPNAME ## _no_rnd_pixels8_l2(block, pixels, pixels+line_size, line_size, line_size, line_size, h);\
}\
\
static inline void OPNAME ## _pixels8_y2_c(uint8_t *block, const uint8_t *pixels, int line_size, int h){\
    OPNAME ## _pixels8_l2(block, pixels, pixels+line_size, line_size, line_size, line_size, h);\
}\
\
static inline void OPNAME ## _pixels8_l4(uint8_t *dst, const uint8_t *src1, uint8_t *src2, uint8_t *src3, uint8_t *src4,\
                 int dst_stride, int src_stride1, int src_stride2,int src_stride3,int src_stride4, int h){\
    int i;\
    for(i=0; i<h; i++){\
        uint32_t a, b, c, d, l0, l1, h0, h1;\
        a= AV_RN32(&src1[i*src_stride1]);\
        b= AV_RN32(&src2[i*src_stride2]);\
        c= AV_RN32(&src3[i*src_stride3]);\
        d= AV_RN32(&src4[i*src_stride4]);\
        l0=  (a&0x03030303UL)\
           + (b&0x03030303UL)\
           + 0x02020202UL;\
        h0= ((a&0xFCFCFCFCUL)>>2)\
          + ((b&0xFCFCFCFCUL)>>2);\
        l1=  (c&0x03030303UL)\
           + (d&0x03030303UL);\
        h1= ((c&0xFCFCFCFCUL)>>2)\
          + ((d&0xFCFCFCFCUL)>>2);\
        OP(*((uint32_t*)&dst[i*dst_stride]), h0+h1+(((l0+l1)>>2)&0x0F0F0F0FUL));\
        a= AV_RN32(&src1[i*src_stride1+4]);\
        b= AV_RN32(&src2[i*src_stride2+4]);\
        c= AV_RN32(&src3[i*src_stride3+4]);\
        d= AV_RN32(&src4[i*src_stride4+4]);\
        l0=  (a&0x03030303UL)\
           + (b&0x03030303UL)\
           + 0x02020202UL;\
        h0= ((a&0xFCFCFCFCUL)>>2)\
          + ((b&0xFCFCFCFCUL)>>2);\
        l1=  (c&0x03030303UL)\
           + (d&0x03030303UL);\
        h1= ((c&0xFCFCFCFCUL)>>2)\
          + ((d&0xFCFCFCFCUL)>>2);\
        OP(*((uint32_t*)&dst[i*dst_stride+4]), h0+h1+(((l0+l1)>>2)&0x0F0F0F0FUL));\
    }\
}\
\
static inline void OPNAME ## _pixels4_x2_c(uint8_t *block, const uint8_t *pixels, int line_size, int h){\
    OPNAME ## _pixels4_l2(block, pixels, pixels+1, line_size, line_size, line_size, h);\
}\
\
static inline void OPNAME ## _pixels4_y2_c(uint8_t *block, const uint8_t *pixels, int line_size, int h){\
    OPNAME ## _pixels4_l2(block, pixels, pixels+line_size, line_size, line_size, line_size, h);\
}\
\
static inline void OPNAME ## _pixels2_x2_c(uint8_t *block, const uint8_t *pixels, int line_size, int h){\
    OPNAME ## _pixels2_l2(block, pixels, pixels+1, line_size, line_size, line_size, h);\
}\
\
static inline void OPNAME ## _pixels2_y2_c(uint8_t *block, const uint8_t *pixels, int line_size, int h){\
    OPNAME ## _pixels2_l2(block, pixels, pixels+line_size, line_size, line_size, line_size, h);\
}\
\
static inline void OPNAME ## _no_rnd_pixels8_l4(uint8_t *dst, const uint8_t *src1, uint8_t *src2, uint8_t *src3, uint8_t *src4,\
                 int dst_stride, int src_stride1, int src_stride2,int src_stride3,int src_stride4, int h){\
    int i;\
    for(i=0; i<h; i++){\
        uint32_t a, b, c, d, l0, l1, h0, h1;\
        a= AV_RN32(&src1[i*src_stride1]);\
        b= AV_RN32(&src2[i*src_stride2]);\
        c= AV_RN32(&src3[i*src_stride3]);\
        d= AV_RN32(&src4[i*src_stride4]);\
        l0=  (a&0x03030303UL)\
           + (b&0x03030303UL)\
           + 0x01010101UL;\
        h0= ((a&0xFCFCFCFCUL)>>2)\
          + ((b&0xFCFCFCFCUL)>>2);\
        l1=  (c&0x03030303UL)\
           + (d&0x03030303UL);\
        h1= ((c&0xFCFCFCFCUL)>>2)\
          + ((d&0xFCFCFCFCUL)>>2);\
        OP(*((uint32_t*)&dst[i*dst_stride]), h0+h1+(((l0+l1)>>2)&0x0F0F0F0FUL));\
        a= AV_RN32(&src1[i*src_stride1+4]);\
        b= AV_RN32(&src2[i*src_stride2+4]);\
        c= AV_RN32(&src3[i*src_stride3+4]);\
        d= AV_RN32(&src4[i*src_stride4+4]);\
        l0=  (a&0x03030303UL)\
           + (b&0x03030303UL)\
           + 0x01010101UL;\
        h0= ((a&0xFCFCFCFCUL)>>2)\
          + ((b&0xFCFCFCFCUL)>>2);\
        l1=  (c&0x03030303UL)\
           + (d&0x03030303UL);\
        h1= ((c&0xFCFCFCFCUL)>>2)\
          + ((d&0xFCFCFCFCUL)>>2);\
        OP(*((uint32_t*)&dst[i*dst_stride+4]), h0+h1+(((l0+l1)>>2)&0x0F0F0F0FUL));\
    }\
}\
static inline void OPNAME ## _pixels16_l4(uint8_t *dst, const uint8_t *src1, uint8_t *src2, uint8_t *src3, uint8_t *src4,\
                 int dst_stride, int src_stride1, int src_stride2,int src_stride3,int src_stride4, int h){\
    OPNAME ## _pixels8_l4(dst  , src1  , src2  , src3  , src4  , dst_stride, src_stride1, src_stride2, src_stride3, src_stride4, h);\
    OPNAME ## _pixels8_l4(dst+8, src1+8, src2+8, src3+8, src4+8, dst_stride, src_stride1, src_stride2, src_stride3, src_stride4, h);\
}\
static inline void OPNAME ## _no_rnd_pixels16_l4(uint8_t *dst, const uint8_t *src1, uint8_t *src2, uint8_t *src3, uint8_t *src4,\
                 int dst_stride, int src_stride1, int src_stride2,int src_stride3,int src_stride4, int h){\
    OPNAME ## _no_rnd_pixels8_l4(dst  , src1  , src2  , src3  , src4  , dst_stride, src_stride1, src_stride2, src_stride3, src_stride4, h);\
    OPNAME ## _no_rnd_pixels8_l4(dst+8, src1+8, src2+8, src3+8, src4+8, dst_stride, src_stride1, src_stride2, src_stride3, src_stride4, h);\
}\
\
static inline void OPNAME ## _pixels2_xy2_c(uint8_t *block, const uint8_t *pixels, int line_size, int h)\
{\
        int i, a0, b0, a1, b1;\
        a0= pixels[0];\
        b0= pixels[1] + 2;\
        a0 += b0;\
        b0 += pixels[2];\
\
        pixels+=line_size;\
        for(i=0; i<h; i+=2){\
            a1= pixels[0];\
            b1= pixels[1];\
            a1 += b1;\
            b1 += pixels[2];\
\
            block[0]= (a1+a0)>>2; /* FIXME non put */\
            block[1]= (b1+b0)>>2;\
\
            pixels+=line_size;\
            block +=line_size;\
\
            a0= pixels[0];\
            b0= pixels[1] + 2;\
            a0 += b0;\
            b0 += pixels[2];\
\
            block[0]= (a1+a0)>>2;\
            block[1]= (b1+b0)>>2;\
            pixels+=line_size;\
            block +=line_size;\
        }\
}\
\
static inline void OPNAME ## _pixels4_xy2_c(uint8_t *block, const uint8_t *pixels, int line_size, int h)\
{\
        int i;\
        const uint32_t a= AV_RN32(pixels  );\
        const uint32_t b= AV_RN32(pixels+1);\
        uint32_t l0=  (a&0x03030303UL)\
                    + (b&0x03030303UL)\
                    + 0x02020202UL;\
        uint32_t h0= ((a&0xFCFCFCFCUL)>>2)\
                   + ((b&0xFCFCFCFCUL)>>2);\
        uint32_t l1,h1;\
\
        pixels+=line_size;\
        for(i=0; i<h; i+=2){\
            uint32_t a= AV_RN32(pixels  );\
            uint32_t b= AV_RN32(pixels+1);\
            l1=  (a&0x03030303UL)\
               + (b&0x03030303UL);\
            h1= ((a&0xFCFCFCFCUL)>>2)\
              + ((b&0xFCFCFCFCUL)>>2);\
            OP(*((uint32_t*)block), h0+h1+(((l0+l1)>>2)&0x0F0F0F0FUL));\
            pixels+=line_size;\
            block +=line_size;\
            a= AV_RN32(pixels  );\
            b= AV_RN32(pixels+1);\
            l0=  (a&0x03030303UL)\
               + (b&0x03030303UL)\
               + 0x02020202UL;\
            h0= ((a&0xFCFCFCFCUL)>>2)\
              + ((b&0xFCFCFCFCUL)>>2);\
            OP(*((uint32_t*)block), h0+h1+(((l0+l1)>>2)&0x0F0F0F0FUL));\
            pixels+=line_size;\
            block +=line_size;\
        }\
}\
\
static inline void OPNAME ## _pixels8_xy2_c(uint8_t *block, const uint8_t *pixels, int line_size, int h)\
{\
    int j;\
    for(j=0; j<2; j++){\
        int i;\
        const uint32_t a= AV_RN32(pixels  );\
        const uint32_t b= AV_RN32(pixels+1);\
        uint32_t l0=  (a&0x03030303UL)\
                    + (b&0x03030303UL)\
                    + 0x02020202UL;\
        uint32_t h0= ((a&0xFCFCFCFCUL)>>2)\
                   + ((b&0xFCFCFCFCUL)>>2);\
        uint32_t l1,h1;\
\
        pixels+=line_size;\
        for(i=0; i<h; i+=2){\
            uint32_t a= AV_RN32(pixels  );\
            uint32_t b= AV_RN32(pixels+1);\
            l1=  (a&0x03030303UL)\
               + (b&0x03030303UL);\
            h1= ((a&0xFCFCFCFCUL)>>2)\
              + ((b&0xFCFCFCFCUL)>>2);\
            OP(*((uint32_t*)block), h0+h1+(((l0+l1)>>2)&0x0F0F0F0FUL));\
            pixels+=line_size;\
            block +=line_size;\
            a= AV_RN32(pixels  );\
            b= AV_RN32(pixels+1);\
            l0=  (a&0x03030303UL)\
               + (b&0x03030303UL)\
               + 0x02020202UL;\
            h0= ((a&0xFCFCFCFCUL)>>2)\
              + ((b&0xFCFCFCFCUL)>>2);\
            OP(*((uint32_t*)block), h0+h1+(((l0+l1)>>2)&0x0F0F0F0FUL));\
            pixels+=line_size;\
            block +=line_size;\
        }\
        pixels+=4-line_size*(h+1);\
        block +=4-line_size*h;\
    }\
}\
\
static inline void OPNAME ## _no_rnd_pixels8_xy2_c(uint8_t *block, const uint8_t *pixels, int line_size, int h)\
{\
    int j;\
    for(j=0; j<2; j++){\
        int i;\
        const uint32_t a= AV_RN32(pixels  );\
        const uint32_t b= AV_RN32(pixels+1);\
        uint32_t l0=  (a&0x03030303UL)\
                    + (b&0x03030303UL)\
                    + 0x01010101UL;\
        uint32_t h0= ((a&0xFCFCFCFCUL)>>2)\
                   + ((b&0xFCFCFCFCUL)>>2);\
        uint32_t l1,h1;\
\
        pixels+=line_size;\
        for(i=0; i<h; i+=2){\
            uint32_t a= AV_RN32(pixels  );\
            uint32_t b= AV_RN32(pixels+1);\
            l1=  (a&0x03030303UL)\
               + (b&0x03030303UL);\
            h1= ((a&0xFCFCFCFCUL)>>2)\
              + ((b&0xFCFCFCFCUL)>>2);\
            OP(*((uint32_t*)block), h0+h1+(((l0+l1)>>2)&0x0F0F0F0FUL));\
            pixels+=line_size;\
            block +=line_size;\
            a= AV_RN32(pixels  );\
            b= AV_RN32(pixels+1);\
            l0=  (a&0x03030303UL)\
               + (b&0x03030303UL)\
               + 0x01010101UL;\
            h0= ((a&0xFCFCFCFCUL)>>2)\
              + ((b&0xFCFCFCFCUL)>>2);\
            OP(*((uint32_t*)block), h0+h1+(((l0+l1)>>2)&0x0F0F0F0FUL));\
            pixels+=line_size;\
            block +=line_size;\
        }\
        pixels+=4-line_size*(h+1);\
        block +=4-line_size*h;\
    }\
}\
\
CALL_2X_PIXELS(OPNAME ## _pixels16_c  , OPNAME ## _pixels8_c  , 8)\
CALL_2X_PIXELS(OPNAME ## _pixels16_x2_c , OPNAME ## _pixels8_x2_c , 8)\
CALL_2X_PIXELS(OPNAME ## _pixels16_y2_c , OPNAME ## _pixels8_y2_c , 8)\
CALL_2X_PIXELS(OPNAME ## _pixels16_xy2_c, OPNAME ## _pixels8_xy2_c, 8)\
CALL_2X_PIXELS(OPNAME ## _no_rnd_pixels16_c  , OPNAME ## _pixels8_c         , 8)\
CALL_2X_PIXELS(OPNAME ## _no_rnd_pixels16_x2_c , OPNAME ## _no_rnd_pixels8_x2_c , 8)\
CALL_2X_PIXELS(OPNAME ## _no_rnd_pixels16_y2_c , OPNAME ## _no_rnd_pixels8_y2_c , 8)\
CALL_2X_PIXELS(OPNAME ## _no_rnd_pixels16_xy2_c, OPNAME ## _no_rnd_pixels8_xy2_c, 8)\

#define op_avg(a, b) a = rnd_avg32(a, b)
#endif
#define op_put(a, b) a = b

PIXOP2(avg, op_avg)
PIXOP2(put, op_put)
#undef op_avg
#undef op_put

#define avg2(a,b) ((a+b+1)>>1)
#define avg4(a,b,c,d) ((a+b+c+d+2)>>2)

static void put_no_rnd_pixels16_l2_c(uint8_t *dst, const uint8_t *a, const uint8_t *b, int stride, int h){
    put_no_rnd_pixels16_l2(dst, a, b, stride, stride, stride, h);
}

static void put_no_rnd_pixels8_l2_c(uint8_t *dst, const uint8_t *a, const uint8_t *b, int stride, int h){
    put_no_rnd_pixels8_l2(dst, a, b, stride, stride, stride, h);
}

static void gmc1_c(uint8_t *dst, uint8_t *src, int stride, int h, int x16, int y16, int rounder)
{
    const int A=(16-x16)*(16-y16);
    const int B=(   x16)*(16-y16);
    const int C=(16-x16)*(   y16);
    const int D=(   x16)*(   y16);
    int i;

    for(i=0; i<h; i++)
    {
        dst[0]= (A*src[0] + B*src[1] + C*src[stride+0] + D*src[stride+1] + rounder)>>8;
        dst[1]= (A*src[1] + B*src[2] + C*src[stride+1] + D*src[stride+2] + rounder)>>8;
        dst[2]= (A*src[2] + B*src[3] + C*src[stride+2] + D*src[stride+3] + rounder)>>8;
        dst[3]= (A*src[3] + B*src[4] + C*src[stride+3] + D*src[stride+4] + rounder)>>8;
        dst[4]= (A*src[4] + B*src[5] + C*src[stride+4] + D*src[stride+5] + rounder)>>8;
        dst[5]= (A*src[5] + B*src[6] + C*src[stride+5] + D*src[stride+6] + rounder)>>8;
        dst[6]= (A*src[6] + B*src[7] + C*src[stride+6] + D*src[stride+7] + rounder)>>8;
        dst[7]= (A*src[7] + B*src[8] + C*src[stride+7] + D*src[stride+8] + rounder)>>8;
        dst+= stride;
        src+= stride;
    }
}

void ff_gmc_c(uint8_t *dst, uint8_t *src, int stride, int h, int ox, int oy,
                  int dxx, int dxy, int dyx, int dyy, int shift, int r, int width, int height)
{
    int y, vx, vy;
    const int s= 1<<shift;

    width--;
    height--;

    for(y=0; y<h; y++){
        int x;

        vx= ox;
        vy= oy;
        for(x=0; x<8; x++){ //XXX FIXME optimize
            int src_x, src_y, frac_x, frac_y, index;

            src_x= vx>>16;
            src_y= vy>>16;
            frac_x= src_x&(s-1);
            frac_y= src_y&(s-1);
            src_x>>=shift;
            src_y>>=shift;

            if((unsigned)src_x < width){
                if((unsigned)src_y < height){
                    index= src_x + src_y*stride;
                    dst[y*stride + x]= (  (  src[index         ]*(s-frac_x)
                                           + src[index       +1]*   frac_x )*(s-frac_y)
                                        + (  src[index+stride  ]*(s-frac_x)
                                           + src[index+stride+1]*   frac_x )*   frac_y
                                        + r)>>(shift*2);
                }else{
                    index= src_x + av_clip(src_y, 0, height)*stride;
                    dst[y*stride + x]= ( (  src[index         ]*(s-frac_x)
                                          + src[index       +1]*   frac_x )*s
                                        + r)>>(shift*2);
                }
            }else{
                if((unsigned)src_y < height){
                    index= av_clip(src_x, 0, width) + src_y*stride;
                    dst[y*stride + x]= (  (  src[index         ]*(s-frac_y)
                                           + src[index+stride  ]*   frac_y )*s
                                        + r)>>(shift*2);
                }else{
                    index= av_clip(src_x, 0, width) + av_clip(src_y, 0, height)*stride;
                    dst[y*stride + x]=    src[index         ];
                }
            }

            vx+= dxx;
            vy+= dyx;
        }
        ox += dxy;
        oy += dyy;
    }
}

static inline void put_tpel_pixels_mc00_c(uint8_t *dst, const uint8_t *src, int stride, int width, int height){
    switch(width){
    case 2: put_pixels2_c (dst, src, stride, height); break;
    case 4: put_pixels4_c (dst, src, stride, height); break;
    case 8: put_pixels8_c (dst, src, stride, height); break;
    case 16:put_pixels16_c(dst, src, stride, height); break;
    }
}

static inline void put_tpel_pixels_mc10_c(uint8_t *dst, const uint8_t *src, int stride, int width, int height){
    int i,j;
    for (i=0; i < height; i++) {
      for (j=0; j < width; j++) {
        dst[j] = (683*(2*src[j] + src[j+1] + 1)) >> 11;
      }
      src += stride;
      dst += stride;
    }
}

static inline void put_tpel_pixels_mc20_c(uint8_t *dst, const uint8_t *src, int stride, int width, int height){
    int i,j;
    for (i=0; i < height; i++) {
      for (j=0; j < width; j++) {
        dst[j] = (683*(src[j] + 2*src[j+1] + 1)) >> 11;
      }
      src += stride;
      dst += stride;
    }
}

static inline void put_tpel_pixels_mc01_c(uint8_t *dst, const uint8_t *src, int stride, int width, int height){
    int i,j;
    for (i=0; i < height; i++) {
      for (j=0; j < width; j++) {
        dst[j] = (683*(2*src[j] + src[j+stride] + 1)) >> 11;
      }
      src += stride;
      dst += stride;
    }
}

static inline void put_tpel_pixels_mc11_c(uint8_t *dst, const uint8_t *src, int stride, int width, int height){
    int i,j;
    for (i=0; i < height; i++) {
      for (j=0; j < width; j++) {
        dst[j] = (2731*(4*src[j] + 3*src[j+1] + 3*src[j+stride] + 2*src[j+stride+1] + 6)) >> 15;
      }
      src += stride;
      dst += stride;
    }
}

static inline void put_tpel_pixels_mc12_c(uint8_t *dst, const uint8_t *src, int stride, int width, int height){
    int i,j;
    for (i=0; i < height; i++) {
      for (j=0; j < width; j++) {
        dst[j] = (2731*(3*src[j] + 2*src[j+1] + 4*src[j+stride] + 3*src[j+stride+1] + 6)) >> 15;
      }
      src += stride;
      dst += stride;
    }
}

static inline void put_tpel_pixels_mc02_c(uint8_t *dst, const uint8_t *src, int stride, int width, int height){
    int i,j;
    for (i=0; i < height; i++) {
      for (j=0; j < width; j++) {
        dst[j] = (683*(src[j] + 2*src[j+stride] + 1)) >> 11;
      }
      src += stride;
      dst += stride;
    }
}

static inline void put_tpel_pixels_mc21_c(uint8_t *dst, const uint8_t *src, int stride, int width, int height){
    int i,j;
    for (i=0; i < height; i++) {
      for (j=0; j < width; j++) {
        dst[j] = (2731*(3*src[j] + 4*src[j+1] + 2*src[j+stride] + 3*src[j+stride+1] + 6)) >> 15;
      }
      src += stride;
      dst += stride;
    }
}

static inline void put_tpel_pixels_mc22_c(uint8_t *dst, const uint8_t *src, int stride, int width, int height){
    int i,j;
    for (i=0; i < height; i++) {
      for (j=0; j < width; j++) {
        dst[j] = (2731*(2*src[j] + 3*src[j+1] + 3*src[j+stride] + 4*src[j+stride+1] + 6)) >> 15;
      }
      src += stride;
      dst += stride;
    }
}

static inline void avg_tpel_pixels_mc00_c(uint8_t *dst, const uint8_t *src, int stride, int width, int height){
    switch(width){
    case 2: avg_pixels2_c (dst, src, stride, height); break;
    case 4: avg_pixels4_c (dst, src, stride, height); break;
    case 8: avg_pixels8_c (dst, src, stride, height); break;
    case 16:avg_pixels16_c(dst, src, stride, height); break;
    }
}

static inline void avg_tpel_pixels_mc10_c(uint8_t *dst, const uint8_t *src, int stride, int width, int height){
    int i,j;
    for (i=0; i < height; i++) {
      for (j=0; j < width; j++) {
        dst[j] = (dst[j] + ((683*(2*src[j] + src[j+1] + 1)) >> 11) + 1) >> 1;
      }
      src += stride;
      dst += stride;
    }
}

static inline void avg_tpel_pixels_mc20_c(uint8_t *dst, const uint8_t *src, int stride, int width, int height){
    int i,j;
    for (i=0; i < height; i++) {
      for (j=0; j < width; j++) {
        dst[j] = (dst[j] + ((683*(src[j] + 2*src[j+1] + 1)) >> 11) + 1) >> 1;
      }
      src += stride;
      dst += stride;
    }
}

static inline void avg_tpel_pixels_mc01_c(uint8_t *dst, const uint8_t *src, int stride, int width, int height){
    int i,j;
    for (i=0; i < height; i++) {
      for (j=0; j < width; j++) {
        dst[j] = (dst[j] + ((683*(2*src[j] + src[j+stride] + 1)) >> 11) + 1) >> 1;
      }
      src += stride;
      dst += stride;
    }
}

static inline void avg_tpel_pixels_mc11_c(uint8_t *dst, const uint8_t *src, int stride, int width, int height){
    int i,j;
    for (i=0; i < height; i++) {
      for (j=0; j < width; j++) {
        dst[j] = (dst[j] + ((2731*(4*src[j] + 3*src[j+1] + 3*src[j+stride] + 2*src[j+stride+1] + 6)) >> 15) + 1) >> 1;
      }
      src += stride;
      dst += stride;
    }
}

static inline void avg_tpel_pixels_mc12_c(uint8_t *dst, const uint8_t *src, int stride, int width, int height){
    int i,j;
    for (i=0; i < height; i++) {
      for (j=0; j < width; j++) {
        dst[j] = (dst[j] + ((2731*(3*src[j] + 2*src[j+1] + 4*src[j+stride] + 3*src[j+stride+1] + 6)) >> 15) + 1) >> 1;
      }
      src += stride;
      dst += stride;
    }
}

static inline void avg_tpel_pixels_mc02_c(uint8_t *dst, const uint8_t *src, int stride, int width, int height){
    int i,j;
    for (i=0; i < height; i++) {
      for (j=0; j < width; j++) {
        dst[j] = (dst[j] + ((683*(src[j] + 2*src[j+stride] + 1)) >> 11) + 1) >> 1;
      }
      src += stride;
      dst += stride;
    }
}

static inline void avg_tpel_pixels_mc21_c(uint8_t *dst, const uint8_t *src, int stride, int width, int height){
    int i,j;
    for (i=0; i < height; i++) {
      for (j=0; j < width; j++) {
        dst[j] = (dst[j] + ((2731*(3*src[j] + 4*src[j+1] + 2*src[j+stride] + 3*src[j+stride+1] + 6)) >> 15) + 1) >> 1;
      }
      src += stride;
      dst += stride;
    }
}

static inline void avg_tpel_pixels_mc22_c(uint8_t *dst, const uint8_t *src, int stride, int width, int height){
    int i,j;
    for (i=0; i < height; i++) {
      for (j=0; j < width; j++) {
        dst[j] = (dst[j] + ((2731*(2*src[j] + 3*src[j+1] + 3*src[j+stride] + 4*src[j+stride+1] + 6)) >> 15) + 1) >> 1;
      }
      src += stride;
      dst += stride;
    }
}

void ff_intrax8dsp_init(DSPContext* c, AVCodecContext *avctx);


static inline int pix_abs16_c(void *v, uint8_t *pix1, uint8_t *pix2, int line_size, int h)
{
    int s, i;

    s = 0;
    for(i=0;i<h;i++) {
        s += abs(pix1[0] - pix2[0]);
        s += abs(pix1[1] - pix2[1]);
        s += abs(pix1[2] - pix2[2]);
        s += abs(pix1[3] - pix2[3]);
        s += abs(pix1[4] - pix2[4]);
        s += abs(pix1[5] - pix2[5]);
        s += abs(pix1[6] - pix2[6]);
        s += abs(pix1[7] - pix2[7]);
        s += abs(pix1[8] - pix2[8]);
        s += abs(pix1[9] - pix2[9]);
        s += abs(pix1[10] - pix2[10]);
        s += abs(pix1[11] - pix2[11]);
        s += abs(pix1[12] - pix2[12]);
        s += abs(pix1[13] - pix2[13]);
        s += abs(pix1[14] - pix2[14]);
        s += abs(pix1[15] - pix2[15]);
        pix1 += line_size;
        pix2 += line_size;
    }
    return s;
}

static int pix_abs16_x2_c(void *v, uint8_t *pix1, uint8_t *pix2, int line_size, int h)
{
    int s, i;

    s = 0;
    for(i=0;i<h;i++) {
        s += abs(pix1[0] - avg2(pix2[0], pix2[1]));
        s += abs(pix1[1] - avg2(pix2[1], pix2[2]));
        s += abs(pix1[2] - avg2(pix2[2], pix2[3]));
        s += abs(pix1[3] - avg2(pix2[3], pix2[4]));
        s += abs(pix1[4] - avg2(pix2[4], pix2[5]));
        s += abs(pix1[5] - avg2(pix2[5], pix2[6]));
        s += abs(pix1[6] - avg2(pix2[6], pix2[7]));
        s += abs(pix1[7] - avg2(pix2[7], pix2[8]));
        s += abs(pix1[8] - avg2(pix2[8], pix2[9]));
        s += abs(pix1[9] - avg2(pix2[9], pix2[10]));
        s += abs(pix1[10] - avg2(pix2[10], pix2[11]));
        s += abs(pix1[11] - avg2(pix2[11], pix2[12]));
        s += abs(pix1[12] - avg2(pix2[12], pix2[13]));
        s += abs(pix1[13] - avg2(pix2[13], pix2[14]));
        s += abs(pix1[14] - avg2(pix2[14], pix2[15]));
        s += abs(pix1[15] - avg2(pix2[15], pix2[16]));
        pix1 += line_size;
        pix2 += line_size;
    }
    return s;
}

static int pix_abs16_y2_c(void *v, uint8_t *pix1, uint8_t *pix2, int line_size, int h)
{
    int s, i;
    uint8_t *pix3 = pix2 + line_size;

    s = 0;
    for(i=0;i<h;i++) {
        s += abs(pix1[0] - avg2(pix2[0], pix3[0]));
        s += abs(pix1[1] - avg2(pix2[1], pix3[1]));
        s += abs(pix1[2] - avg2(pix2[2], pix3[2]));
        s += abs(pix1[3] - avg2(pix2[3], pix3[3]));
        s += abs(pix1[4] - avg2(pix2[4], pix3[4]));
        s += abs(pix1[5] - avg2(pix2[5], pix3[5]));
        s += abs(pix1[6] - avg2(pix2[6], pix3[6]));
        s += abs(pix1[7] - avg2(pix2[7], pix3[7]));
        s += abs(pix1[8] - avg2(pix2[8], pix3[8]));
        s += abs(pix1[9] - avg2(pix2[9], pix3[9]));
        s += abs(pix1[10] - avg2(pix2[10], pix3[10]));
        s += abs(pix1[11] - avg2(pix2[11], pix3[11]));
        s += abs(pix1[12] - avg2(pix2[12], pix3[12]));
        s += abs(pix1[13] - avg2(pix2[13], pix3[13]));
        s += abs(pix1[14] - avg2(pix2[14], pix3[14]));
        s += abs(pix1[15] - avg2(pix2[15], pix3[15]));
        pix1 += line_size;
        pix2 += line_size;
        pix3 += line_size;
    }
    return s;
}

static int pix_abs16_xy2_c(void *v, uint8_t *pix1, uint8_t *pix2, int line_size, int h)
{
    int s, i;
    uint8_t *pix3 = pix2 + line_size;

    s = 0;
    for(i=0;i<h;i++) {
        s += abs(pix1[0] - avg4(pix2[0], pix2[1], pix3[0], pix3[1]));
        s += abs(pix1[1] - avg4(pix2[1], pix2[2], pix3[1], pix3[2]));
        s += abs(pix1[2] - avg4(pix2[2], pix2[3], pix3[2], pix3[3]));
        s += abs(pix1[3] - avg4(pix2[3], pix2[4], pix3[3], pix3[4]));
        s += abs(pix1[4] - avg4(pix2[4], pix2[5], pix3[4], pix3[5]));
        s += abs(pix1[5] - avg4(pix2[5], pix2[6], pix3[5], pix3[6]));
        s += abs(pix1[6] - avg4(pix2[6], pix2[7], pix3[6], pix3[7]));
        s += abs(pix1[7] - avg4(pix2[7], pix2[8], pix3[7], pix3[8]));
        s += abs(pix1[8] - avg4(pix2[8], pix2[9], pix3[8], pix3[9]));
        s += abs(pix1[9] - avg4(pix2[9], pix2[10], pix3[9], pix3[10]));
        s += abs(pix1[10] - avg4(pix2[10], pix2[11], pix3[10], pix3[11]));
        s += abs(pix1[11] - avg4(pix2[11], pix2[12], pix3[11], pix3[12]));
        s += abs(pix1[12] - avg4(pix2[12], pix2[13], pix3[12], pix3[13]));
        s += abs(pix1[13] - avg4(pix2[13], pix2[14], pix3[13], pix3[14]));
        s += abs(pix1[14] - avg4(pix2[14], pix2[15], pix3[14], pix3[15]));
        s += abs(pix1[15] - avg4(pix2[15], pix2[16], pix3[15], pix3[16]));
        pix1 += line_size;
        pix2 += line_size;
        pix3 += line_size;
    }
    return s;
}

static inline int pix_abs8_c(void *v, uint8_t *pix1, uint8_t *pix2, int line_size, int h)
{
    int s, i;

    s = 0;
    for(i=0;i<h;i++) {
        s += abs(pix1[0] - pix2[0]);
        s += abs(pix1[1] - pix2[1]);
        s += abs(pix1[2] - pix2[2]);
        s += abs(pix1[3] - pix2[3]);
        s += abs(pix1[4] - pix2[4]);
        s += abs(pix1[5] - pix2[5]);
        s += abs(pix1[6] - pix2[6]);
        s += abs(pix1[7] - pix2[7]);
        pix1 += line_size;
        pix2 += line_size;
    }
    return s;
}

static int pix_abs8_x2_c(void *v, uint8_t *pix1, uint8_t *pix2, int line_size, int h)
{
    int s, i;

    s = 0;
    for(i=0;i<h;i++) {
        s += abs(pix1[0] - avg2(pix2[0], pix2[1]));
        s += abs(pix1[1] - avg2(pix2[1], pix2[2]));
        s += abs(pix1[2] - avg2(pix2[2], pix2[3]));
        s += abs(pix1[3] - avg2(pix2[3], pix2[4]));
        s += abs(pix1[4] - avg2(pix2[4], pix2[5]));
        s += abs(pix1[5] - avg2(pix2[5], pix2[6]));
        s += abs(pix1[6] - avg2(pix2[6], pix2[7]));
        s += abs(pix1[7] - avg2(pix2[7], pix2[8]));
        pix1 += line_size;
        pix2 += line_size;
    }
    return s;
}

static int pix_abs8_y2_c(void *v, uint8_t *pix1, uint8_t *pix2, int line_size, int h)
{
    int s, i;
    uint8_t *pix3 = pix2 + line_size;

    s = 0;
    for(i=0;i<h;i++) {
        s += abs(pix1[0] - avg2(pix2[0], pix3[0]));
        s += abs(pix1[1] - avg2(pix2[1], pix3[1]));
        s += abs(pix1[2] - avg2(pix2[2], pix3[2]));
        s += abs(pix1[3] - avg2(pix2[3], pix3[3]));
        s += abs(pix1[4] - avg2(pix2[4], pix3[4]));
        s += abs(pix1[5] - avg2(pix2[5], pix3[5]));
        s += abs(pix1[6] - avg2(pix2[6], pix3[6]));
        s += abs(pix1[7] - avg2(pix2[7], pix3[7]));
        pix1 += line_size;
        pix2 += line_size;
        pix3 += line_size;
    }
    return s;
}

static int pix_abs8_xy2_c(void *v, uint8_t *pix1, uint8_t *pix2, int line_size, int h)
{
    int s, i;
    uint8_t *pix3 = pix2 + line_size;

    s = 0;
    for(i=0;i<h;i++) {
        s += abs(pix1[0] - avg4(pix2[0], pix2[1], pix3[0], pix3[1]));
        s += abs(pix1[1] - avg4(pix2[1], pix2[2], pix3[1], pix3[2]));
        s += abs(pix1[2] - avg4(pix2[2], pix2[3], pix3[2], pix3[3]));
        s += abs(pix1[3] - avg4(pix2[3], pix2[4], pix3[3], pix3[4]));
        s += abs(pix1[4] - avg4(pix2[4], pix2[5], pix3[4], pix3[5]));
        s += abs(pix1[5] - avg4(pix2[5], pix2[6], pix3[5], pix3[6]));
        s += abs(pix1[6] - avg4(pix2[6], pix2[7], pix3[6], pix3[7]));
        s += abs(pix1[7] - avg4(pix2[7], pix2[8], pix3[7], pix3[8]));
        pix1 += line_size;
        pix2 += line_size;
        pix3 += line_size;
    }
    return s;
}

static int nsse16_c(void *v, uint8_t *s1, uint8_t *s2, int stride, int h){
    MpegEncContext *c = v;
    int score1=0;
    int score2=0;
    int x,y;

    for(y=0; y<h; y++){
        for(x=0; x<16; x++){
            score1+= (s1[x  ] - s2[x ])*(s1[x  ] - s2[x ]);
        }
        if(y+1<h){
            for(x=0; x<15; x++){
                score2+= FFABS(  s1[x  ] - s1[x  +stride]
                             - s1[x+1] + s1[x+1+stride])
                        -FFABS(  s2[x  ] - s2[x  +stride]
                             - s2[x+1] + s2[x+1+stride]);
            }
        }
        s1+= stride;
        s2+= stride;
    }

    if(c) return score1 + FFABS(score2)*c->avctx->nsse_weight;
    else  return score1 + FFABS(score2)*8;
}

static int nsse8_c(void *v, uint8_t *s1, uint8_t *s2, int stride, int h){
    MpegEncContext *c = v;
    int score1=0;
    int score2=0;
    int x,y;

    for(y=0; y<h; y++){
        for(x=0; x<8; x++){
            score1+= (s1[x  ] - s2[x ])*(s1[x  ] - s2[x ]);
        }
        if(y+1<h){
            for(x=0; x<7; x++){
                score2+= FFABS(  s1[x  ] - s1[x  +stride]
                             - s1[x+1] + s1[x+1+stride])
                        -FFABS(  s2[x  ] - s2[x  +stride]
                             - s2[x+1] + s2[x+1+stride]);
            }
        }
        s1+= stride;
        s2+= stride;
    }

    if(c) return score1 + FFABS(score2)*c->avctx->nsse_weight;
    else  return score1 + FFABS(score2)*8;
}

static int try_8x8basis_c(int16_t rem[64], int16_t weight[64], int16_t basis[64], int scale){
    int i;
    unsigned int sum=0;

    for(i=0; i<8*8; i++){
        int b= rem[i] + ((basis[i]*scale + (1<<(BASIS_SHIFT - RECON_SHIFT-1)))>>(BASIS_SHIFT - RECON_SHIFT));
        int w= weight[i];
        b>>= RECON_SHIFT;
        assert(-512<b && b<512);

        sum += (w*b)*(w*b)>>4;
    }
    return sum>>2;
}

static void add_8x8basis_c(int16_t rem[64], int16_t basis[64], int scale){
    int i;

    for(i=0; i<8*8; i++){
        rem[i] += (basis[i]*scale + (1<<(BASIS_SHIFT - RECON_SHIFT-1)))>>(BASIS_SHIFT - RECON_SHIFT);
    }
}

/**
 * permutes an 8x8 block.
 * @param block the block which will be permuted according to the given permutation vector
 * @param permutation the permutation vector
 * @param last the last non zero coefficient in scantable order, used to speed the permutation up
 * @param scantable the used scantable, this is only used to speed the permutation up, the block is not
 *                  (inverse) permutated to scantable order!
 */
void ff_block_permute(DCTELEM *block, uint8_t *permutation, const uint8_t *scantable, int last)
{
    int i;
    DCTELEM temp[64];

    if(last<=0) return;
    //if(permutation[1]==1) return; //FIXME it is ok but not clean and might fail for some permutations

    for(i=0; i<=last; i++){
        const int j= scantable[i];
        temp[j]= block[j];
        block[j]=0;
    }

    for(i=0; i<=last; i++){
        const int j= scantable[i];
        const int perm_j= permutation[j];
        block[perm_j]= temp[j];
    }
}

static int zero_cmp(void *s, uint8_t *a, uint8_t *b, int stride, int h){
    return 0;
}

void ff_set_cmp(DSPContext* c, me_cmp_func *cmp, int type){
    int i;

    memset(cmp, 0, sizeof(void*)*5);

    for(i=0; i<5; i++){
        switch(type&0xFF){
        case FF_CMP_SAD:
            cmp[i]= c->sad[i];
            break;
        case FF_CMP_SATD:
            cmp[i]= c->hadamard8_diff[i];
            break;
        case FF_CMP_SSE:
            cmp[i]= c->sse[i];
            break;
        case FF_CMP_DCT:
            cmp[i]= c->dct_sad[i];
            break;
        case FF_CMP_DCT264:
            cmp[i]= c->dct264_sad[i];
            break;
        case FF_CMP_DCTMAX:
            cmp[i]= c->dct_max[i];
            break;
        case FF_CMP_PSNR:
            cmp[i]= c->quant_psnr[i];
            break;
        case FF_CMP_BIT:
            cmp[i]= c->bit[i];
            break;
        case FF_CMP_RD:
            cmp[i]= c->rd[i];
            break;
        case FF_CMP_VSAD:
            cmp[i]= c->vsad[i];
            break;
        case FF_CMP_VSSE:
            cmp[i]= c->vsse[i];
            break;
        case FF_CMP_ZERO:
            cmp[i]= zero_cmp;
            break;
        case FF_CMP_NSSE:
            cmp[i]= c->nsse[i];
            break;
#ifdef CONFIG_SNOW_ENCODER
        case FF_CMP_W53:
            cmp[i]= c->w53[i];
            break;
        case FF_CMP_W97:
            cmp[i]= c->w97[i];
            break;
#endif
        default:
            av_log(NULL, AV_LOG_ERROR,"internal error in cmp function selection\n");
        }
    }
}

/**
 * memset(blocks, 0, sizeof(DCTELEM)*6*64)
 */
static void clear_blocks_c(DCTELEM *blocks)
{
    memset(blocks, 0, sizeof(DCTELEM)*6*64);
}

static void add_bytes_c(uint8_t *dst, uint8_t *src, int w){
    long i;
    for(i=0; i<=w-sizeof(long); i+=sizeof(long)){
        long a = *(long*)(src+i);
        long b = *(long*)(dst+i);
        *(long*)(dst+i) = ((a&pb_7f) + (b&pb_7f)) ^ ((a^b)&pb_80);
    }
    for(; i<w; i++)
        dst[i+0] += src[i+0];
}

static void add_bytes_l2_c(uint8_t *dst, uint8_t *src1, uint8_t *src2, int w){
    long i;
    for(i=0; i<=w-sizeof(long); i+=sizeof(long)){
        long a = *(long*)(src1+i);
        long b = *(long*)(src2+i);
        *(long*)(dst+i) = ((a&pb_7f) + (b&pb_7f)) ^ ((a^b)&pb_80);
    }
    for(; i<w; i++)
        dst[i] = src1[i]+src2[i];
}

static void diff_bytes_c(uint8_t *dst, uint8_t *src1, uint8_t *src2, int w){
    long i;
#ifndef HAVE_FAST_UNALIGNED
    if((long)src2 & (sizeof(long)-1)){
        for(i=0; i+7<w; i+=8){
            dst[i+0] = src1[i+0]-src2[i+0];
            dst[i+1] = src1[i+1]-src2[i+1];
            dst[i+2] = src1[i+2]-src2[i+2];
            dst[i+3] = src1[i+3]-src2[i+3];
            dst[i+4] = src1[i+4]-src2[i+4];
            dst[i+5] = src1[i+5]-src2[i+5];
            dst[i+6] = src1[i+6]-src2[i+6];
            dst[i+7] = src1[i+7]-src2[i+7];
        }
    }else
#endif
    for(i=0; i<=w-sizeof(long); i+=sizeof(long)){
        long a = *(long*)(src1+i);
        long b = *(long*)(src2+i);
        *(long*)(dst+i) = ((a|pb_80) - (b&pb_7f)) ^ ((a^b^pb_80)&pb_80);
    }
    for(; i<w; i++)
        dst[i+0] = src1[i+0]-src2[i+0];
}

static void sub_hfyu_median_prediction_c(uint8_t *dst, uint8_t *src1, uint8_t *src2, int w, int *left, int *left_top){
    int i;
    uint8_t l, lt;

    l= *left;
    lt= *left_top;

    for(i=0; i<w; i++){
        const int pred= mid_pred(l, src1[i], (l + src1[i] - lt)&0xFF);
        lt= src1[i];
        l= src2[i];
        dst[i]= l - pred;
    }

    *left= l;
    *left_top= lt;
}

#define BUTTERFLY2(o1,o2,i1,i2) \
o1= (i1)+(i2);\
o2= (i1)-(i2);

#define BUTTERFLY1(x,y) \
{\
    int a,b;\
    a= x;\
    b= y;\
    x= a+b;\
    y= a-b;\
}

#define BUTTERFLYA(x,y) (FFABS((x)+(y)) + FFABS((x)-(y)))

static int hadamard8_diff8x8_c(/*MpegEncContext*/ void *s, uint8_t *dst, uint8_t *src, int stride, int h){
    int i;
    int temp[64];
    int sum=0;

    assert(h==8);

    for(i=0; i<8; i++){
        //FIXME try pointer walks
        BUTTERFLY2(temp[8*i+0], temp[8*i+1], src[stride*i+0]-dst[stride*i+0],src[stride*i+1]-dst[stride*i+1]);
        BUTTERFLY2(temp[8*i+2], temp[8*i+3], src[stride*i+2]-dst[stride*i+2],src[stride*i+3]-dst[stride*i+3]);
        BUTTERFLY2(temp[8*i+4], temp[8*i+5], src[stride*i+4]-dst[stride*i+4],src[stride*i+5]-dst[stride*i+5]);
        BUTTERFLY2(temp[8*i+6], temp[8*i+7], src[stride*i+6]-dst[stride*i+6],src[stride*i+7]-dst[stride*i+7]);

        BUTTERFLY1(temp[8*i+0], temp[8*i+2]);
        BUTTERFLY1(temp[8*i+1], temp[8*i+3]);
        BUTTERFLY1(temp[8*i+4], temp[8*i+6]);
        BUTTERFLY1(temp[8*i+5], temp[8*i+7]);

        BUTTERFLY1(temp[8*i+0], temp[8*i+4]);
        BUTTERFLY1(temp[8*i+1], temp[8*i+5]);
        BUTTERFLY1(temp[8*i+2], temp[8*i+6]);
        BUTTERFLY1(temp[8*i+3], temp[8*i+7]);
    }

    for(i=0; i<8; i++){
        BUTTERFLY1(temp[8*0+i], temp[8*1+i]);
        BUTTERFLY1(temp[8*2+i], temp[8*3+i]);
        BUTTERFLY1(temp[8*4+i], temp[8*5+i]);
        BUTTERFLY1(temp[8*6+i], temp[8*7+i]);

        BUTTERFLY1(temp[8*0+i], temp[8*2+i]);
        BUTTERFLY1(temp[8*1+i], temp[8*3+i]);
        BUTTERFLY1(temp[8*4+i], temp[8*6+i]);
        BUTTERFLY1(temp[8*5+i], temp[8*7+i]);

        sum +=
             BUTTERFLYA(temp[8*0+i], temp[8*4+i])
            +BUTTERFLYA(temp[8*1+i], temp[8*5+i])
            +BUTTERFLYA(temp[8*2+i], temp[8*6+i])
            +BUTTERFLYA(temp[8*3+i], temp[8*7+i]);
    }
#if 0
static int maxi=0;
if(sum>maxi){
    maxi=sum;
    printf("MAX:%d\n", maxi);
}
#endif
    return sum;
}

static int hadamard8_intra8x8_c(/*MpegEncContext*/ void *s, uint8_t *src, uint8_t *dummy, int stride, int h){
    int i;
    int temp[64];
    int sum=0;

    assert(h==8);

    for(i=0; i<8; i++){
        //FIXME try pointer walks
        BUTTERFLY2(temp[8*i+0], temp[8*i+1], src[stride*i+0],src[stride*i+1]);
        BUTTERFLY2(temp[8*i+2], temp[8*i+3], src[stride*i+2],src[stride*i+3]);
        BUTTERFLY2(temp[8*i+4], temp[8*i+5], src[stride*i+4],src[stride*i+5]);
        BUTTERFLY2(temp[8*i+6], temp[8*i+7], src[stride*i+6],src[stride*i+7]);

        BUTTERFLY1(temp[8*i+0], temp[8*i+2]);
        BUTTERFLY1(temp[8*i+1], temp[8*i+3]);
        BUTTERFLY1(temp[8*i+4], temp[8*i+6]);
        BUTTERFLY1(temp[8*i+5], temp[8*i+7]);

        BUTTERFLY1(temp[8*i+0], temp[8*i+4]);
        BUTTERFLY1(temp[8*i+1], temp[8*i+5]);
        BUTTERFLY1(temp[8*i+2], temp[8*i+6]);
        BUTTERFLY1(temp[8*i+3], temp[8*i+7]);
    }

    for(i=0; i<8; i++){
        BUTTERFLY1(temp[8*0+i], temp[8*1+i]);
        BUTTERFLY1(temp[8*2+i], temp[8*3+i]);
        BUTTERFLY1(temp[8*4+i], temp[8*5+i]);
        BUTTERFLY1(temp[8*6+i], temp[8*7+i]);

        BUTTERFLY1(temp[8*0+i], temp[8*2+i]);
        BUTTERFLY1(temp[8*1+i], temp[8*3+i]);
        BUTTERFLY1(temp[8*4+i], temp[8*6+i]);
        BUTTERFLY1(temp[8*5+i], temp[8*7+i]);

        sum +=
             BUTTERFLYA(temp[8*0+i], temp[8*4+i])
            +BUTTERFLYA(temp[8*1+i], temp[8*5+i])
            +BUTTERFLYA(temp[8*2+i], temp[8*6+i])
            +BUTTERFLYA(temp[8*3+i], temp[8*7+i]);
    }

    sum -= FFABS(temp[8*0] + temp[8*4]); // -mean

    return sum;
}

static int dct_sad8x8_c(/*MpegEncContext*/ void *c, uint8_t *src1, uint8_t *src2, int stride, int h){
    MpegEncContext * const s= (MpegEncContext *)c;
    DECLARE_ALIGNED_16(uint64_t, aligned_temp[sizeof(DCTELEM)*64/8]);
    DCTELEM * const temp= (DCTELEM*)aligned_temp;

    assert(h==8);

    s->dsp.diff_pixels(temp, src1, src2, stride);
    s->dsp.fdct(temp);
    return s->dsp.sum_abs_dctelem(temp);
}

#ifdef CONFIG_GPL
#define DCT8_1D {\
    const int s07 = SRC(0) + SRC(7);\
    const int s16 = SRC(1) + SRC(6);\
    const int s25 = SRC(2) + SRC(5);\
    const int s34 = SRC(3) + SRC(4);\
    const int a0 = s07 + s34;\
    const int a1 = s16 + s25;\
    const int a2 = s07 - s34;\
    const int a3 = s16 - s25;\
    const int d07 = SRC(0) - SRC(7);\
    const int d16 = SRC(1) - SRC(6);\
    const int d25 = SRC(2) - SRC(5);\
    const int d34 = SRC(3) - SRC(4);\
    const int a4 = d16 + d25 + (d07 + (d07>>1));\
    const int a5 = d07 - d34 - (d25 + (d25>>1));\
    const int a6 = d07 + d34 - (d16 + (d16>>1));\
    const int a7 = d16 - d25 + (d34 + (d34>>1));\
    DST(0,  a0 + a1     ) ;\
    DST(1,  a4 + (a7>>2)) ;\
    DST(2,  a2 + (a3>>1)) ;\
    DST(3,  a5 + (a6>>2)) ;\
    DST(4,  a0 - a1     ) ;\
    DST(5,  a6 - (a5>>2)) ;\
    DST(6, (a2>>1) - a3 ) ;\
    DST(7, (a4>>2) - a7 ) ;\
}

static int dct264_sad8x8_c(/*MpegEncContext*/ void *c, uint8_t *src1, uint8_t *src2, int stride, int h){
    MpegEncContext * const s= (MpegEncContext *)c;
    DCTELEM dct[8][8];
    int i;
    int sum=0;

    s->dsp.diff_pixels(dct[0], src1, src2, stride);

#define SRC(x) dct[i][x]
#define DST(x,v) dct[i][x]= v
    for( i = 0; i < 8; i++ )
        DCT8_1D
#undef SRC
#undef DST

#define SRC(x) dct[x][i]
#define DST(x,v) sum += FFABS(v)
    for( i = 0; i < 8; i++ )
        DCT8_1D
#undef SRC
#undef DST
    return sum;
}
#endif

static int dct_max8x8_c(/*MpegEncContext*/ void *c, uint8_t *src1, uint8_t *src2, int stride, int h){
    MpegEncContext * const s= (MpegEncContext *)c;
    DECLARE_ALIGNED_8(uint64_t, aligned_temp[sizeof(DCTELEM)*64/8]);
    DCTELEM * const temp= (DCTELEM*)aligned_temp;
    int sum=0, i;

    assert(h==8);

    s->dsp.diff_pixels(temp, src1, src2, stride);
    s->dsp.fdct(temp);

    for(i=0; i<64; i++)
        sum= FFMAX(sum, FFABS(temp[i]));

    return sum;
}

static int quant_psnr8x8_c(/*MpegEncContext*/ void *c, uint8_t *src1, uint8_t *src2, int stride, int h){
    MpegEncContext * const s= (MpegEncContext *)c;
    DECLARE_ALIGNED_8 (uint64_t, aligned_temp[sizeof(DCTELEM)*64*2/8]);
    DCTELEM * const temp= (DCTELEM*)aligned_temp;
    DCTELEM * const bak = ((DCTELEM*)aligned_temp)+64;
    int sum=0, i;

    assert(h==8);
    s->mb_intra=0;

    s->dsp.diff_pixels(temp, src1, src2, stride);

    memcpy(bak, temp, 64*sizeof(DCTELEM));

    s->block_last_index[0/*FIXME*/]= s->fast_dct_quantize(s, temp, 0/*FIXME*/, s->qscale, &i);
    s->dct_unquantize_inter(s, temp, 0, s->qscale);
    ff_simple_idct(temp); //FIXME

    for(i=0; i<64; i++)
        sum+= (temp[i]-bak[i])*(temp[i]-bak[i]);

    return sum;
}

static int rd8x8_c(/*MpegEncContext*/ void *c, uint8_t *src1, uint8_t *src2, int stride, int h){
    MpegEncContext * const s= (MpegEncContext *)c;
    const uint8_t *scantable= s->intra_scantable.permutated;
    DECLARE_ALIGNED_8 (uint64_t, aligned_temp[sizeof(DCTELEM)*64/8]);
    DECLARE_ALIGNED_8 (uint64_t, aligned_bak[stride]);
    DCTELEM * const temp= (DCTELEM*)aligned_temp;
    uint8_t * const bak= (uint8_t*)aligned_bak;
    int i, last, run, bits, level, distortion, start_i;
    const int esc_length= s->ac_esc_length;
    uint8_t * length;
    uint8_t * last_length;

    assert(h==8);

    for(i=0; i<8; i++){
        ((uint32_t*)(bak + i*stride))[0]= ((uint32_t*)(src2 + i*stride))[0];
        ((uint32_t*)(bak + i*stride))[1]= ((uint32_t*)(src2 + i*stride))[1];
    }

    s->dsp.diff_pixels(temp, src1, src2, stride);

    s->block_last_index[0/*FIXME*/]= last= s->fast_dct_quantize(s, temp, 0/*FIXME*/, s->qscale, &i);

    bits=0;

    if (s->mb_intra) {
        start_i = 1;
        length     = s->intra_ac_vlc_length;
        last_length= s->intra_ac_vlc_last_length;
        bits+= s->luma_dc_vlc_length[temp[0] + 256]; //FIXME chroma
    } else {
        start_i = 0;
        length     = s->inter_ac_vlc_length;
        last_length= s->inter_ac_vlc_last_length;
    }

    if(last>=start_i){
        run=0;
        for(i=start_i; i<last; i++){
            int j= scantable[i];
            level= temp[j];

            if(level){
                level+=64;
                if((level&(~127)) == 0){
                    bits+= length[UNI_AC_ENC_INDEX(run, level)];
                }else
                    bits+= esc_length;
                run=0;
            }else
                run++;
        }
        i= scantable[last];

        level= temp[i] + 64;

        assert(level - 64);

        if((level&(~127)) == 0){
            bits+= last_length[UNI_AC_ENC_INDEX(run, level)];
        }else
            bits+= esc_length;

    }

    if(last>=0){
        if(s->mb_intra)
            s->dct_unquantize_intra(s, temp, 0, s->qscale);
        else
            s->dct_unquantize_inter(s, temp, 0, s->qscale);
    }

    s->dsp.idct_add(bak, stride, temp);

    distortion= s->dsp.sse[1](NULL, bak, src1, stride, 8);

    return distortion + ((bits*s->qscale*s->qscale*109 + 64)>>7);
}

static int bit8x8_c(/*MpegEncContext*/ void *c, uint8_t *src1, uint8_t *src2, int stride, int h){
    MpegEncContext * const s= (MpegEncContext *)c;
    const uint8_t *scantable= s->intra_scantable.permutated;
    DECLARE_ALIGNED_8 (uint64_t, aligned_temp[sizeof(DCTELEM)*64/8]);
    DCTELEM * const temp= (DCTELEM*)aligned_temp;
    int i, last, run, bits, level, start_i;
    const int esc_length= s->ac_esc_length;
    uint8_t * length;
    uint8_t * last_length;

    assert(h==8);

    s->dsp.diff_pixels(temp, src1, src2, stride);

    s->block_last_index[0/*FIXME*/]= last= s->fast_dct_quantize(s, temp, 0/*FIXME*/, s->qscale, &i);

    bits=0;

    if (s->mb_intra) {
        start_i = 1;
        length     = s->intra_ac_vlc_length;
        last_length= s->intra_ac_vlc_last_length;
        bits+= s->luma_dc_vlc_length[temp[0] + 256]; //FIXME chroma
    } else {
        start_i = 0;
        length     = s->inter_ac_vlc_length;
        last_length= s->inter_ac_vlc_last_length;
    }

    if(last>=start_i){
        run=0;
        for(i=start_i; i<last; i++){
            int j= scantable[i];
            level= temp[j];

            if(level){
                level+=64;
                if((level&(~127)) == 0){
                    bits+= length[UNI_AC_ENC_INDEX(run, level)];
                }else
                    bits+= esc_length;
                run=0;
            }else
                run++;
        }
        i= scantable[last];

        level= temp[i] + 64;

        assert(level - 64);

        if((level&(~127)) == 0){
            bits+= last_length[UNI_AC_ENC_INDEX(run, level)];
        }else
            bits+= esc_length;
    }

    return bits;
}

static int vsad_intra16_c(/*MpegEncContext*/ void *c, uint8_t *s, uint8_t *dummy, int stride, int h){
    int score=0;
    int x,y;

    for(y=1; y<h; y++){
        for(x=0; x<16; x+=4){
            score+= FFABS(s[x  ] - s[x  +stride]) + FFABS(s[x+1] - s[x+1+stride])
                   +FFABS(s[x+2] - s[x+2+stride]) + FFABS(s[x+3] - s[x+3+stride]);
        }
        s+= stride;
    }

    return score;
}

static int vsad16_c(/*MpegEncContext*/ void *c, uint8_t *s1, uint8_t *s2, int stride, int h){
    int score=0;
    int x,y;

    for(y=1; y<h; y++){
        for(x=0; x<16; x++){
            score+= FFABS(s1[x  ] - s2[x ] - s1[x  +stride] + s2[x +stride]);
        }
        s1+= stride;
        s2+= stride;
    }

    return score;
}

#define SQ(a) ((a)*(a))
static int vsse_intra16_c(/*MpegEncContext*/ void *c, uint8_t *s, uint8_t *dummy, int stride, int h){
    int score=0;
    int x,y;

    for(y=1; y<h; y++){
        for(x=0; x<16; x+=4){
            score+= SQ(s[x  ] - s[x  +stride]) + SQ(s[x+1] - s[x+1+stride])
                   +SQ(s[x+2] - s[x+2+stride]) + SQ(s[x+3] - s[x+3+stride]);
        }
        s+= stride;
    }

    return score;
}

static int vsse16_c(/*MpegEncContext*/ void *c, uint8_t *s1, uint8_t *s2, int stride, int h){
    int score=0;
    int x,y;

    for(y=1; y<h; y++){
        for(x=0; x<16; x++){
            score+= SQ(s1[x  ] - s2[x ] - s1[x  +stride] + s2[x +stride]);
        }
        s1+= stride;
        s2+= stride;
    }

    return score;
}

static int ssd_int8_vs_int16_c(const int8_t *pix1, const int16_t *pix2,
                               int size){
    int score=0;
    int i;
    for(i=0; i<size; i++)
        score += (pix1[i]-pix2[i])*(pix1[i]-pix2[i]);
    return score;
}

WRAPPER8_16_SQ(hadamard8_diff8x8_c, hadamard8_diff16_c)
WRAPPER8_16_SQ(hadamard8_intra8x8_c, hadamard8_intra16_c)
WRAPPER8_16_SQ(dct_sad8x8_c, dct_sad16_c)
#ifdef CONFIG_GPL
WRAPPER8_16_SQ(dct264_sad8x8_c, dct264_sad16_c)
#endif
WRAPPER8_16_SQ(dct_max8x8_c, dct_max16_c)
WRAPPER8_16_SQ(quant_psnr8x8_c, quant_psnr16_c)
WRAPPER8_16_SQ(rd8x8_c, rd16_c)
WRAPPER8_16_SQ(bit8x8_c, bit16_c)

static void vector_fmul_c(float *dst, const float *src, int len){
    int i;
    for(i=0; i<len; i++)
        dst[i] *= src[i];
}

static void vector_fmul_reverse_c(float *dst, const float *src0, const float *src1, int len){
    int i;
    src1 += len-1;
    for(i=0; i<len; i++)
        dst[i] = src0[i] * src1[-i];
}

void ff_vector_fmul_add_add_c(float *dst, const float *src0, const float *src1, const float *src2, int src3, int len, int step){
    int i;
    for(i=0; i<len; i++)
        dst[i*step] = src0[i] * src1[i] + src2[i] + src3;
}

void ff_vector_fmul_window_c(float *dst, const float *src0, const float *src1, const float *win, float add_bias, int len){
    int i,j;
    dst += len;
    win += len;
    src0+= len;
    for(i=-len, j=len-1; i<0; i++, j--) {
        float s0 = src0[i];
        float s1 = src1[j];
        float wi = win[i];
        float wj = win[j];
        dst[i] = s0*wj - s1*wi + add_bias;
        dst[j] = s0*wi + s1*wj + add_bias;
    }
}

static void int32_to_float_fmul_scalar_c(float *dst, const int *src, float mul, int len){
    int i;
    for(i=0; i<len; i++)
        dst[i] = src[i] * mul;
}

static av_always_inline int float_to_int16_one(const float *src){
    int_fast32_t tmp = *(const int32_t*)src;
    if(tmp & 0xf0000){
        tmp = (0x43c0ffff - tmp)>>31;
        // is this faster on some gcc/cpu combinations?
//      if(tmp > 0x43c0ffff) tmp = 0xFFFF;
//      else                 tmp = 0;
    }
    return tmp - 0x8000;
}

void ff_float_to_int16_c(int16_t *dst, const float *src, long len){
    int i;
    for(i=0; i<len; i++)
        dst[i] = float_to_int16_one(src+i);
}

void ff_float_to_int16_interleave_c(int16_t *dst, const float **src, long len, int channels){
    int i,j,c;
    if(channels==2){
        for(i=0; i<len; i++){
            dst[2*i]   = float_to_int16_one(src[0]+i);
            dst[2*i+1] = float_to_int16_one(src[1]+i);
        }
    }else{
        for(c=0; c<channels; c++)
            for(i=0, j=c; i<len; i++, j+=channels)
                dst[j] = float_to_int16_one(src[c]+i);
    }
}

static void add_int16_c(int16_t * v1, int16_t * v2, int order)
{
    while (order--)
       *v1++ += *v2++;
}

static void sub_int16_c(int16_t * v1, int16_t * v2, int order)
{
    while (order--)
        *v1++ -= *v2++;
}

static int32_t scalarproduct_int16_c(int16_t * v1, int16_t * v2, int order, int shift)
{
    int res = 0;

    while (order--)
        res += (*v1++ * *v2++) >> shift;

    return res;
}
static void ff_jref_idct_put(uint8_t *dest, int line_size, DCTELEM *block)
{
    j_rev_dct (block);
    put_pixels_clamped_c(block, dest, line_size);
}
static void ff_jref_idct_add(uint8_t *dest, int line_size, DCTELEM *block)
{
    j_rev_dct (block);
    add_pixels_clamped_c(block, dest, line_size);
}

static void ff_jref_idct4_put(uint8_t *dest, int line_size, DCTELEM *block)
{
    j_rev_dct4 (block);
    put_pixels_clamped4_c(block, dest, line_size);
}
static void ff_jref_idct4_add(uint8_t *dest, int line_size, DCTELEM *block)
{
    j_rev_dct4 (block);
    add_pixels_clamped4_c(block, dest, line_size);
}

static void ff_jref_idct2_put(uint8_t *dest, int line_size, DCTELEM *block)
{
    j_rev_dct2 (block);
    put_pixels_clamped2_c(block, dest, line_size);
}
static void ff_jref_idct2_add(uint8_t *dest, int line_size, DCTELEM *block)
{
    j_rev_dct2 (block);
    add_pixels_clamped2_c(block, dest, line_size);
}

static void ff_jref_idct1_put(uint8_t *dest, int line_size, DCTELEM *block)
{
    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;

    dest[0] = cm[(block[0] + 4)>>3];
}
static void ff_jref_idct1_add(uint8_t *dest, int line_size, DCTELEM *block)
{
    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;

    dest[0] = cm[dest[0] + ((block[0] + 4)>>3)];
}

static void just_return(void *mem av_unused, int stride av_unused, int h av_unused) { return; }

/* init static data */
void dsputil_static_init(void)
{
    int i;

    for(i=0;i<256;i++) ff_cropTbl[i + MAX_NEG_CROP] = i;
    for(i=0;i<MAX_NEG_CROP;i++) {
        ff_cropTbl[i] = 0;
        ff_cropTbl[i + MAX_NEG_CROP + 256] = 255;
    }

    for(i=0;i<512;i++) {
        ff_squareTbl[i] = (i - 256) * (i - 256);
    }

    for(i=0; i<64; i++) inv_zigzag_direct16[ff_zigzag_direct[i]]= i+1;
}

int ff_check_alignment(void){
    static int did_fail=0;
    DECLARE_ALIGNED_16(int, aligned);

    if((long)&aligned & 15){
        if(!did_fail){
#if defined(HAVE_MMX) || defined(HAVE_ALTIVEC)
            av_log(NULL, AV_LOG_ERROR,
                "Compiler did not align stack variables. Libavcodec has been miscompiled\n"
                "and may be very slow or crash. This is not a bug in libavcodec,\n"
                "but in the compiler. You may try recompiling using gcc >= 4.2.\n"
                "Do not report crashes to FFmpeg developers.\n");
#endif
            did_fail=1;
        }
        return -1;
    }
    return 0;
}

void dsputil_init(DSPContext* c, AVCodecContext *avctx)
{
    int i;

    ff_check_alignment();

#ifdef CONFIG_ENCODERS
    if(avctx->dct_algo==FF_DCT_FASTINT) {
        c->fdct = fdct_ifast;
        c->fdct248 = fdct_ifast248;
    }
    else if(avctx->dct_algo==FF_DCT_FAAN) {
        c->fdct = ff_faandct;
        c->fdct248 = ff_faandct248;
    }
    else {
        c->fdct = ff_jpeg_fdct_islow; //slow/accurate/default
        c->fdct248 = ff_fdct248_islow;
    }
#endif //CONFIG_ENCODERS

    if(avctx->lowres==1){
        c->idct_put= ff_jref_idct4_put;
        c->idct_add= ff_jref_idct4_add;
        c->idct    = j_rev_dct4;
        c->idct_permutation_type= FF_NO_IDCT_PERM;
    }else if(avctx->lowres==2){
        c->idct_put= ff_jref_idct2_put;
        c->idct_add= ff_jref_idct2_add;
        c->idct    = j_rev_dct2;
        c->idct_permutation_type= FF_NO_IDCT_PERM;
    }else if(avctx->lowres==3){
        c->idct_put= ff_jref_idct1_put;
        c->idct_add= ff_jref_idct1_add;
        c->idct    = j_rev_dct1;
        c->idct_permutation_type= FF_NO_IDCT_PERM;
    }else{
        if(avctx->idct_algo==FF_IDCT_INT){
            c->idct_put= ff_jref_idct_put;
            c->idct_add= ff_jref_idct_add;
            c->idct    = j_rev_dct;
            c->idct_permutation_type= FF_LIBMPEG2_IDCT_PERM;
        }else if(avctx->idct_algo==FF_IDCT_FAAN){
            c->idct_put= ff_faanidct_put;
            c->idct_add= ff_faanidct_add;
            c->idct    = ff_faanidct;
            c->idct_permutation_type= FF_NO_IDCT_PERM;
        }else{ //accurate/default
            c->idct_put= ff_simple_idct_put;
            c->idct_add= ff_simple_idct_add;
            c->idct    = ff_simple_idct;
            c->idct_permutation_type= FF_NO_IDCT_PERM;
        }
    }

    c->get_pixels = get_pixels_c;
    c->diff_pixels = diff_pixels_c;
    c->put_pixels_clamped = put_pixels_clamped_c;
    c->put_signed_pixels_clamped = put_signed_pixels_clamped_c;
    c->add_pixels_clamped = add_pixels_clamped_c;
    c->add_pixels8 = add_pixels8_c;
    c->add_pixels4 = add_pixels4_c;
    c->sum_abs_dctelem = sum_abs_dctelem_c;
    c->gmc1 = gmc1_c;
    c->gmc = ff_gmc_c;
    c->clear_blocks = clear_blocks_c;
    c->pix_sum = pix_sum_c;
    c->pix_norm1 = pix_norm1_c;

    /* TODO [0] 16  [1] 8 */
    c->pix_abs[0][0] = pix_abs16_c;
    c->pix_abs[0][1] = pix_abs16_x2_c;
    c->pix_abs[0][2] = pix_abs16_y2_c;
    c->pix_abs[0][3] = pix_abs16_xy2_c;
    c->pix_abs[1][0] = pix_abs8_c;
    c->pix_abs[1][1] = pix_abs8_x2_c;
    c->pix_abs[1][2] = pix_abs8_y2_c;
    c->pix_abs[1][3] = pix_abs8_xy2_c;

#define dspfunc(PFX, IDX, NUM) \
    c->PFX ## _pixels_tab[IDX][0] = PFX ## _pixels ## NUM ## _c;     \
    c->PFX ## _pixels_tab[IDX][1] = PFX ## _pixels ## NUM ## _x2_c;  \
    c->PFX ## _pixels_tab[IDX][2] = PFX ## _pixels ## NUM ## _y2_c;  \
    c->PFX ## _pixels_tab[IDX][3] = PFX ## _pixels ## NUM ## _xy2_c

    dspfunc(put, 0, 16);
    dspfunc(put_no_rnd, 0, 16);
    dspfunc(put, 1, 8);
    dspfunc(put_no_rnd, 1, 8);
    dspfunc(put, 2, 4);
    dspfunc(put, 3, 2);

    dspfunc(avg, 0, 16);
    dspfunc(avg_no_rnd, 0, 16);
    dspfunc(avg, 1, 8);
    dspfunc(avg_no_rnd, 1, 8);
    dspfunc(avg, 2, 4);
    dspfunc(avg, 3, 2);
#undef dspfunc

    c->put_no_rnd_pixels_l2[0]= put_no_rnd_pixels16_l2_c;
    c->put_no_rnd_pixels_l2[1]= put_no_rnd_pixels8_l2_c;

    c->put_tpel_pixels_tab[ 0] = put_tpel_pixels_mc00_c;
    c->put_tpel_pixels_tab[ 1] = put_tpel_pixels_mc10_c;
    c->put_tpel_pixels_tab[ 2] = put_tpel_pixels_mc20_c;
    c->put_tpel_pixels_tab[ 4] = put_tpel_pixels_mc01_c;
    c->put_tpel_pixels_tab[ 5] = put_tpel_pixels_mc11_c;
    c->put_tpel_pixels_tab[ 6] = put_tpel_pixels_mc21_c;
    c->put_tpel_pixels_tab[ 8] = put_tpel_pixels_mc02_c;
    c->put_tpel_pixels_tab[ 9] = put_tpel_pixels_mc12_c;
    c->put_tpel_pixels_tab[10] = put_tpel_pixels_mc22_c;

    c->avg_tpel_pixels_tab[ 0] = avg_tpel_pixels_mc00_c;
    c->avg_tpel_pixels_tab[ 1] = avg_tpel_pixels_mc10_c;
    c->avg_tpel_pixels_tab[ 2] = avg_tpel_pixels_mc20_c;
    c->avg_tpel_pixels_tab[ 4] = avg_tpel_pixels_mc01_c;
    c->avg_tpel_pixels_tab[ 5] = avg_tpel_pixels_mc11_c;
    c->avg_tpel_pixels_tab[ 6] = avg_tpel_pixels_mc21_c;
    c->avg_tpel_pixels_tab[ 8] = avg_tpel_pixels_mc02_c;
    c->avg_tpel_pixels_tab[ 9] = avg_tpel_pixels_mc12_c;
    c->avg_tpel_pixels_tab[10] = avg_tpel_pixels_mc22_c;

#define dspfunc(PFX, IDX, NUM) \
    c->PFX ## _pixels_tab[IDX][ 0] = PFX ## NUM ## _mc00_c; \
    c->PFX ## _pixels_tab[IDX][ 1] = PFX ## NUM ## _mc10_c; \
    c->PFX ## _pixels_tab[IDX][ 2] = PFX ## NUM ## _mc20_c; \
    c->PFX ## _pixels_tab[IDX][ 3] = PFX ## NUM ## _mc30_c; \
    c->PFX ## _pixels_tab[IDX][ 4] = PFX ## NUM ## _mc01_c; \
    c->PFX ## _pixels_tab[IDX][ 5] = PFX ## NUM ## _mc11_c; \
    c->PFX ## _pixels_tab[IDX][ 6] = PFX ## NUM ## _mc21_c; \
    c->PFX ## _pixels_tab[IDX][ 7] = PFX ## NUM ## _mc31_c; \
    c->PFX ## _pixels_tab[IDX][ 8] = PFX ## NUM ## _mc02_c; \
    c->PFX ## _pixels_tab[IDX][ 9] = PFX ## NUM ## _mc12_c; \
    c->PFX ## _pixels_tab[IDX][10] = PFX ## NUM ## _mc22_c; \
    c->PFX ## _pixels_tab[IDX][11] = PFX ## NUM ## _mc32_c; \
    c->PFX ## _pixels_tab[IDX][12] = PFX ## NUM ## _mc03_c; \
    c->PFX ## _pixels_tab[IDX][13] = PFX ## NUM ## _mc13_c; \
    c->PFX ## _pixels_tab[IDX][14] = PFX ## NUM ## _mc23_c; \
    c->PFX ## _pixels_tab[IDX][15] = PFX ## NUM ## _mc33_c

#undef dspfunc

    c->draw_edges = draw_edges_c;

#define SET_CMP_FUNC(name) \
    c->name[0]= name ## 16_c;\
    c->name[1]= name ## 8x8_c;

    SET_CMP_FUNC(hadamard8_diff)
    c->hadamard8_diff[4]= hadamard8_intra16_c;
    SET_CMP_FUNC(dct_sad)
    SET_CMP_FUNC(dct_max)
#ifdef CONFIG_GPL
    SET_CMP_FUNC(dct264_sad)
#endif
    c->sad[0]= pix_abs16_c;
    c->sad[1]= pix_abs8_c;
    c->sse[0]= sse16_c;
    c->sse[1]= sse8_c;
    c->sse[2]= sse4_c;
    SET_CMP_FUNC(quant_psnr)
    SET_CMP_FUNC(rd)
    SET_CMP_FUNC(bit)
    c->vsad[0]= vsad16_c;
    c->vsad[4]= vsad_intra16_c;
    c->vsse[0]= vsse16_c;
    c->vsse[4]= vsse_intra16_c;
    c->nsse[0]= nsse16_c;
    c->nsse[1]= nsse8_c;
#ifdef CONFIG_SNOW_ENCODER
    c->w53[0]= w53_16_c;
    c->w53[1]= w53_8_c;
    c->w97[0]= w97_16_c;
    c->w97[1]= w97_8_c;
#endif

    c->ssd_int8_vs_int16 = ssd_int8_vs_int16_c;

    c->add_bytes= add_bytes_c;
    c->add_bytes_l2= add_bytes_l2_c;
    c->diff_bytes= diff_bytes_c;
    c->sub_hfyu_median_prediction= sub_hfyu_median_prediction_c;
    c->bswap_buf= bswap_buf;
#ifdef CONFIG_PNG_DECODER
    c->add_png_paeth_prediction= ff_add_png_paeth_prediction;
#endif

    c->try_8x8basis= try_8x8basis_c;
    c->add_8x8basis= add_8x8basis_c;

#ifdef CONFIG_SNOW_DECODER
    c->vertical_compose97i = ff_snow_vertical_compose97i;
    c->horizontal_compose97i = ff_snow_horizontal_compose97i;
    c->inner_add_yblock = ff_snow_inner_add_yblock;
#endif

#ifdef CONFIG_VORBIS_DECODER
    c->vorbis_inverse_coupling = vorbis_inverse_coupling;
#endif
#ifdef CONFIG_AC3_DECODER
    c->ac3_downmix = ff_ac3_downmix_c;
#endif
#ifdef CONFIG_FLAC_ENCODER
    c->flac_compute_autocorr = ff_flac_compute_autocorr;
#endif
    c->vector_fmul = vector_fmul_c;
    c->vector_fmul_reverse = vector_fmul_reverse_c;
    c->vector_fmul_add_add = ff_vector_fmul_add_add_c;
    c->vector_fmul_window = ff_vector_fmul_window_c;
    c->int32_to_float_fmul_scalar = int32_to_float_fmul_scalar_c;
    c->float_to_int16 = ff_float_to_int16_c;
    c->float_to_int16_interleave = ff_float_to_int16_interleave_c;
    c->add_int16 = add_int16_c;
    c->sub_int16 = sub_int16_c;
    c->scalarproduct_int16 = scalarproduct_int16_c;

    c->shrink[0]= ff_img_copy_plane;
    c->shrink[1]= ff_shrink22;
    c->shrink[2]= ff_shrink44;
    c->shrink[3]= ff_shrink88;

    c->prefetch= just_return;

    memset(c->put_2tap_qpel_pixels_tab, 0, sizeof(c->put_2tap_qpel_pixels_tab));
    memset(c->avg_2tap_qpel_pixels_tab, 0, sizeof(c->avg_2tap_qpel_pixels_tab));

    if (ENABLE_MMX)      dsputil_init_mmx   (c, avctx);
    if (ENABLE_ARMV4L)   dsputil_init_armv4l(c, avctx);
    if (ENABLE_MLIB)     dsputil_init_mlib  (c, avctx);
    if (ENABLE_VIS)      dsputil_init_vis   (c, avctx);
    if (ENABLE_ALPHA)    dsputil_init_alpha (c, avctx);
    if (ENABLE_POWERPC)  dsputil_init_ppc   (c, avctx);
    if (ENABLE_MMI)      dsputil_init_mmi   (c, avctx);
    if (ENABLE_SH4)      dsputil_init_sh4   (c, avctx);
    if (ENABLE_BFIN)     dsputil_init_bfin  (c, avctx);

    switch(c->idct_permutation_type){
    case FF_NO_IDCT_PERM:
        for(i=0; i<64; i++)
            c->idct_permutation[i]= i;
        break;
    case FF_LIBMPEG2_IDCT_PERM:
        for(i=0; i<64; i++)
            c->idct_permutation[i]= (i & 0x38) | ((i & 6) >> 1) | ((i & 1) << 2);
        break;
    case FF_SIMPLE_IDCT_PERM:
        for(i=0; i<64; i++)
            c->idct_permutation[i]= simple_mmx_permutation[i];
        break;
    case FF_TRANSPOSE_IDCT_PERM:
        for(i=0; i<64; i++)
            c->idct_permutation[i]= ((i&7)<<3) | (i>>3);
        break;
    case FF_PARTTRANS_IDCT_PERM:
        for(i=0; i<64; i++)
            c->idct_permutation[i]= (i&0x24) | ((i&3)<<3) | ((i>>3)&3);
        break;
    case FF_SSE2_IDCT_PERM:
        for(i=0; i<64; i++)
            c->idct_permutation[i]= (i&0x38) | idct_sse2_row_perm[i&7];
        break;
    default:
        av_log(avctx, AV_LOG_ERROR, "Internal error, IDCT permutation not set\n");
    }
}

