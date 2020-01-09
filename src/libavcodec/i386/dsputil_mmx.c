/*
 * MMX optimized DSP utils
 * Copyright (c) 2000, 2001 Fabrice Bellard.
 * Copyright (c) 2002-2004 Michael Niedermayer <michaelni@gmx.at>
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
 *
 * MMX optimization by Nick Kurshev <nickols_k@mail.ru>
 */

#include "libavutil/x86_cpu.h"
#include "libavcodec/dsputil.h"
#include "libavcodec/mpegvideo.h"
#include "libavcodec/simple_idct.h"
#include "dsputil_mmx.h"
#include "mmx.h"
#include "vp3dsp_mmx.h"
#include "vp3dsp_sse2.h"

//#undef NDEBUG
//#include <assert.h>

int mm_flags; /* multimedia extension flags */

/* pixel operations */
DECLARE_ALIGNED_8 (const uint64_t, ff_bone) = 0x0101010101010101ULL;
DECLARE_ALIGNED_8 (const uint64_t, ff_wtwo) = 0x0002000200020002ULL;

DECLARE_ALIGNED_16(const uint64_t, ff_pdw_80000000[2]) =
{0x8000000080000000ULL, 0x8000000080000000ULL};

DECLARE_ALIGNED_8 (const uint64_t, ff_pw_3  ) = 0x0003000300030003ULL;
DECLARE_ALIGNED_8 (const uint64_t, ff_pw_4  ) = 0x0004000400040004ULL;
DECLARE_ALIGNED_16(const xmm_t,    ff_pw_5  ) = {0x0005000500050005ULL, 0x0005000500050005ULL};
DECLARE_ALIGNED_16(const xmm_t,    ff_pw_8  ) = {0x0008000800080008ULL, 0x0008000800080008ULL};
DECLARE_ALIGNED_8 (const uint64_t, ff_pw_15 ) = 0x000F000F000F000FULL;
DECLARE_ALIGNED_16(const xmm_t,    ff_pw_16 ) = {0x0010001000100010ULL, 0x0010001000100010ULL};
DECLARE_ALIGNED_8 (const uint64_t, ff_pw_20 ) = 0x0014001400140014ULL;
DECLARE_ALIGNED_16(const xmm_t,    ff_pw_28 ) = {0x001C001C001C001CULL, 0x001C001C001C001CULL};
DECLARE_ALIGNED_16(const xmm_t,    ff_pw_32 ) = {0x0020002000200020ULL, 0x0020002000200020ULL};
DECLARE_ALIGNED_8 (const uint64_t, ff_pw_42 ) = 0x002A002A002A002AULL;
DECLARE_ALIGNED_8 (const uint64_t, ff_pw_64 ) = 0x0040004000400040ULL;
DECLARE_ALIGNED_8 (const uint64_t, ff_pw_96 ) = 0x0060006000600060ULL;
DECLARE_ALIGNED_8 (const uint64_t, ff_pw_128) = 0x0080008000800080ULL;
DECLARE_ALIGNED_8 (const uint64_t, ff_pw_255) = 0x00ff00ff00ff00ffULL;

DECLARE_ALIGNED_8 (const uint64_t, ff_pb_1  ) = 0x0101010101010101ULL;
DECLARE_ALIGNED_8 (const uint64_t, ff_pb_3  ) = 0x0303030303030303ULL;
DECLARE_ALIGNED_8 (const uint64_t, ff_pb_7  ) = 0x0707070707070707ULL;
DECLARE_ALIGNED_8 (const uint64_t, ff_pb_3F ) = 0x3F3F3F3F3F3F3F3FULL;
DECLARE_ALIGNED_8 (const uint64_t, ff_pb_A1 ) = 0xA1A1A1A1A1A1A1A1ULL;
DECLARE_ALIGNED_8 (const uint64_t, ff_pb_FC ) = 0xFCFCFCFCFCFCFCFCULL;

DECLARE_ALIGNED_16(const double, ff_pd_1[2]) = { 1.0, 1.0 };
DECLARE_ALIGNED_16(const double, ff_pd_2[2]) = { 2.0, 2.0 };

#define JUMPALIGN() asm volatile (ASMALIGN(3)::)
#define MOVQ_ZERO(regd)  asm volatile ("pxor %%" #regd ", %%" #regd ::)

#define MOVQ_BFE(regd) \
    asm volatile ( \
    "pcmpeqd %%" #regd ", %%" #regd " \n\t"\
    "paddb %%" #regd ", %%" #regd " \n\t" ::)

#ifndef PIC
#define MOVQ_BONE(regd)  asm volatile ("movq %0, %%" #regd " \n\t" ::"m"(ff_bone))
#define MOVQ_WTWO(regd)  asm volatile ("movq %0, %%" #regd " \n\t" ::"m"(ff_wtwo))
#else
// for shared library it's better to use this way for accessing constants
// pcmpeqd -> -1
#define MOVQ_BONE(regd) \
    asm volatile ( \
    "pcmpeqd %%" #regd ", %%" #regd " \n\t" \
    "psrlw $15, %%" #regd " \n\t" \
    "packuswb %%" #regd ", %%" #regd " \n\t" ::)

#define MOVQ_WTWO(regd) \
    asm volatile ( \
    "pcmpeqd %%" #regd ", %%" #regd " \n\t" \
    "psrlw $15, %%" #regd " \n\t" \
    "psllw $1, %%" #regd " \n\t"::)

#endif

// using regr as temporary and for the output result
// first argument is unmodifed and second is trashed
// regfe is supposed to contain 0xfefefefefefefefe
#define PAVGB_MMX_NO_RND(rega, regb, regr, regfe) \
    "movq " #rega ", " #regr "  \n\t"\
    "pand " #regb ", " #regr "  \n\t"\
    "pxor " #rega ", " #regb "  \n\t"\
    "pand " #regfe "," #regb "  \n\t"\
    "psrlq $1, " #regb "        \n\t"\
    "paddb " #regb ", " #regr " \n\t"

#define PAVGB_MMX(rega, regb, regr, regfe) \
    "movq " #rega ", " #regr "  \n\t"\
    "por  " #regb ", " #regr "  \n\t"\
    "pxor " #rega ", " #regb "  \n\t"\
    "pand " #regfe "," #regb "  \n\t"\
    "psrlq $1, " #regb "        \n\t"\
    "psubb " #regb ", " #regr " \n\t"

// mm6 is supposed to contain 0xfefefefefefefefe
#define PAVGBP_MMX_NO_RND(rega, regb, regr,  regc, regd, regp) \
    "movq " #rega ", " #regr "  \n\t"\
    "movq " #regc ", " #regp "  \n\t"\
    "pand " #regb ", " #regr "  \n\t"\
    "pand " #regd ", " #regp "  \n\t"\
    "pxor " #rega ", " #regb "  \n\t"\
    "pxor " #regc ", " #regd "  \n\t"\
    "pand %%mm6, " #regb "      \n\t"\
    "pand %%mm6, " #regd "      \n\t"\
    "psrlq $1, " #regb "        \n\t"\
    "psrlq $1, " #regd "        \n\t"\
    "paddb " #regb ", " #regr " \n\t"\
    "paddb " #regd ", " #regp " \n\t"

#define PAVGBP_MMX(rega, regb, regr, regc, regd, regp) \
    "movq " #rega ", " #regr "  \n\t"\
    "movq " #regc ", " #regp "  \n\t"\
    "por  " #regb ", " #regr "  \n\t"\
    "por  " #regd ", " #regp "  \n\t"\
    "pxor " #rega ", " #regb "  \n\t"\
    "pxor " #regc ", " #regd "  \n\t"\
    "pand %%mm6, " #regb "      \n\t"\
    "pand %%mm6, " #regd "      \n\t"\
    "psrlq $1, " #regd "        \n\t"\
    "psrlq $1, " #regb "        \n\t"\
    "psubb " #regb ", " #regr " \n\t"\
    "psubb " #regd ", " #regp " \n\t"

/***********************************/
/* MMX no rounding */
#define DEF(x, y) x ## _no_rnd_ ## y ##_mmx
#define SET_RND  MOVQ_WONE
#define PAVGBP(a, b, c, d, e, f)        PAVGBP_MMX_NO_RND(a, b, c, d, e, f)
#define PAVGB(a, b, c, e)               PAVGB_MMX_NO_RND(a, b, c, e)

#include "dsputil_mmx_rnd.h"

#undef DEF
#undef SET_RND
#undef PAVGBP
#undef PAVGB
/***********************************/
/* MMX rounding */

#define DEF(x, y) x ## _ ## y ##_mmx
#define SET_RND  MOVQ_WTWO
#define PAVGBP(a, b, c, d, e, f)        PAVGBP_MMX(a, b, c, d, e, f)
#define PAVGB(a, b, c, e)               PAVGB_MMX(a, b, c, e)

#include "dsputil_mmx_rnd.h"

#undef DEF
#undef SET_RND
#undef PAVGBP
#undef PAVGB

/***********************************/
/* 3Dnow specific */

#define DEF(x) x ## _3dnow
#define PAVGB "pavgusb"

#include "dsputil_mmx_avg.h"

#undef DEF
#undef PAVGB

/***********************************/
/* MMX2 specific */

#define DEF(x) x ## _mmx2

/* Introduced only in MMX2 set */
#define PAVGB "pavgb"

#include "dsputil_mmx_avg.h"

#undef DEF
#undef PAVGB

#define put_no_rnd_pixels16_mmx put_pixels16_mmx
#define put_no_rnd_pixels8_mmx put_pixels8_mmx
#define put_pixels16_mmx2 put_pixels16_mmx
#define put_pixels8_mmx2 put_pixels8_mmx
#define put_pixels4_mmx2 put_pixels4_mmx
#define put_no_rnd_pixels16_mmx2 put_no_rnd_pixels16_mmx
#define put_no_rnd_pixels8_mmx2 put_no_rnd_pixels8_mmx
#define put_pixels16_3dnow put_pixels16_mmx
#define put_pixels8_3dnow put_pixels8_mmx
#define put_pixels4_3dnow put_pixels4_mmx
#define put_no_rnd_pixels16_3dnow put_no_rnd_pixels16_mmx
#define put_no_rnd_pixels8_3dnow put_no_rnd_pixels8_mmx

/***********************************/
/* standard MMX */

void put_pixels_clamped_mmx(const DCTELEM *block, uint8_t *pixels, int line_size)
{
    const DCTELEM *p;
    uint8_t *pix;

    /* read the pixels */
    p = block;
    pix = pixels;
    /* unrolled loop */
        asm volatile(
                "movq   %3, %%mm0               \n\t"
                "movq   8%3, %%mm1              \n\t"
                "movq   16%3, %%mm2             \n\t"
                "movq   24%3, %%mm3             \n\t"
                "movq   32%3, %%mm4             \n\t"
                "movq   40%3, %%mm5             \n\t"
                "movq   48%3, %%mm6             \n\t"
                "movq   56%3, %%mm7             \n\t"
                "packuswb %%mm1, %%mm0          \n\t"
                "packuswb %%mm3, %%mm2          \n\t"
                "packuswb %%mm5, %%mm4          \n\t"
                "packuswb %%mm7, %%mm6          \n\t"
                "movq   %%mm0, (%0)             \n\t"
                "movq   %%mm2, (%0, %1)         \n\t"
                "movq   %%mm4, (%0, %1, 2)      \n\t"
                "movq   %%mm6, (%0, %2)         \n\t"
                ::"r" (pix), "r" ((x86_reg)line_size), "r" ((x86_reg)line_size*3), "m"(*p)
                :"memory");
        pix += line_size*4;
        p += 32;

    // if here would be an exact copy of the code above
    // compiler would generate some very strange code
    // thus using "r"
    asm volatile(
            "movq       (%3), %%mm0             \n\t"
            "movq       8(%3), %%mm1            \n\t"
            "movq       16(%3), %%mm2           \n\t"
            "movq       24(%3), %%mm3           \n\t"
            "movq       32(%3), %%mm4           \n\t"
            "movq       40(%3), %%mm5           \n\t"
            "movq       48(%3), %%mm6           \n\t"
            "movq       56(%3), %%mm7           \n\t"
            "packuswb %%mm1, %%mm0              \n\t"
            "packuswb %%mm3, %%mm2              \n\t"
            "packuswb %%mm5, %%mm4              \n\t"
            "packuswb %%mm7, %%mm6              \n\t"
            "movq       %%mm0, (%0)             \n\t"
            "movq       %%mm2, (%0, %1)         \n\t"
            "movq       %%mm4, (%0, %1, 2)      \n\t"
            "movq       %%mm6, (%0, %2)         \n\t"
            ::"r" (pix), "r" ((x86_reg)line_size), "r" ((x86_reg)line_size*3), "r"(p)
            :"memory");
}

static DECLARE_ALIGNED_8(const unsigned char, vector128[8]) =
  { 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80 };

void put_signed_pixels_clamped_mmx(const DCTELEM *block, uint8_t *pixels, int line_size)
{
    int i;

    movq_m2r(*vector128, mm1);
    for (i = 0; i < 8; i++) {
        movq_m2r(*(block), mm0);
        packsswb_m2r(*(block + 4), mm0);
        block += 8;
        paddb_r2r(mm1, mm0);
        movq_r2m(mm0, *pixels);
        pixels += line_size;
    }
}

void add_pixels_clamped_mmx(const DCTELEM *block, uint8_t *pixels, int line_size)
{
    const DCTELEM *p;
    uint8_t *pix;
    int i;

    /* read the pixels */
    p = block;
    pix = pixels;
    MOVQ_ZERO(mm7);
    i = 4;
    do {
        asm volatile(
                "movq   (%2), %%mm0     \n\t"
                "movq   8(%2), %%mm1    \n\t"
                "movq   16(%2), %%mm2   \n\t"
                "movq   24(%2), %%mm3   \n\t"
                "movq   %0, %%mm4       \n\t"
                "movq   %1, %%mm6       \n\t"
                "movq   %%mm4, %%mm5    \n\t"
                "punpcklbw %%mm7, %%mm4 \n\t"
                "punpckhbw %%mm7, %%mm5 \n\t"
                "paddsw %%mm4, %%mm0    \n\t"
                "paddsw %%mm5, %%mm1    \n\t"
                "movq   %%mm6, %%mm5    \n\t"
                "punpcklbw %%mm7, %%mm6 \n\t"
                "punpckhbw %%mm7, %%mm5 \n\t"
                "paddsw %%mm6, %%mm2    \n\t"
                "paddsw %%mm5, %%mm3    \n\t"
                "packuswb %%mm1, %%mm0  \n\t"
                "packuswb %%mm3, %%mm2  \n\t"
                "movq   %%mm0, %0       \n\t"
                "movq   %%mm2, %1       \n\t"
                :"+m"(*pix), "+m"(*(pix+line_size))
                :"r"(p)
                :"memory");
        pix += line_size*2;
        p += 16;
    } while (--i);
}

static void put_pixels4_mmx(uint8_t *block, const uint8_t *pixels, int line_size, int h)
{
    asm volatile(
         "lea (%3, %3), %%"REG_a"       \n\t"
         ASMALIGN(3)
         "1:                            \n\t"
         "movd (%1), %%mm0              \n\t"
         "movd (%1, %3), %%mm1          \n\t"
         "movd %%mm0, (%2)              \n\t"
         "movd %%mm1, (%2, %3)          \n\t"
         "add %%"REG_a", %1             \n\t"
         "add %%"REG_a", %2             \n\t"
         "movd (%1), %%mm0              \n\t"
         "movd (%1, %3), %%mm1          \n\t"
         "movd %%mm0, (%2)              \n\t"
         "movd %%mm1, (%2, %3)          \n\t"
         "add %%"REG_a", %1             \n\t"
         "add %%"REG_a", %2             \n\t"
         "subl $4, %0                   \n\t"
         "jnz 1b                        \n\t"
         : "+g"(h), "+r" (pixels),  "+r" (block)
         : "r"((x86_reg)line_size)
         : "%"REG_a, "memory"
        );
}

static void put_pixels8_mmx(uint8_t *block, const uint8_t *pixels, int line_size, int h)
{
    asm volatile(
         "lea (%3, %3), %%"REG_a"       \n\t"
         ASMALIGN(3)
         "1:                            \n\t"
         "movq (%1), %%mm0              \n\t"
         "movq (%1, %3), %%mm1          \n\t"
         "movq %%mm0, (%2)              \n\t"
         "movq %%mm1, (%2, %3)          \n\t"
         "add %%"REG_a", %1             \n\t"
         "add %%"REG_a", %2             \n\t"
         "movq (%1), %%mm0              \n\t"
         "movq (%1, %3), %%mm1          \n\t"
         "movq %%mm0, (%2)              \n\t"
         "movq %%mm1, (%2, %3)          \n\t"
         "add %%"REG_a", %1             \n\t"
         "add %%"REG_a", %2             \n\t"
         "subl $4, %0                   \n\t"
         "jnz 1b                        \n\t"
         : "+g"(h), "+r" (pixels),  "+r" (block)
         : "r"((x86_reg)line_size)
         : "%"REG_a, "memory"
        );
}

static void put_pixels16_mmx(uint8_t *block, const uint8_t *pixels, int line_size, int h)
{
    asm volatile(
         "lea (%3, %3), %%"REG_a"       \n\t"
         ASMALIGN(3)
         "1:                            \n\t"
         "movq (%1), %%mm0              \n\t"
         "movq 8(%1), %%mm4             \n\t"
         "movq (%1, %3), %%mm1          \n\t"
         "movq 8(%1, %3), %%mm5         \n\t"
         "movq %%mm0, (%2)              \n\t"
         "movq %%mm4, 8(%2)             \n\t"
         "movq %%mm1, (%2, %3)          \n\t"
         "movq %%mm5, 8(%2, %3)         \n\t"
         "add %%"REG_a", %1             \n\t"
         "add %%"REG_a", %2             \n\t"
         "movq (%1), %%mm0              \n\t"
         "movq 8(%1), %%mm4             \n\t"
         "movq (%1, %3), %%mm1          \n\t"
         "movq 8(%1, %3), %%mm5         \n\t"
         "movq %%mm0, (%2)              \n\t"
         "movq %%mm4, 8(%2)             \n\t"
         "movq %%mm1, (%2, %3)          \n\t"
         "movq %%mm5, 8(%2, %3)         \n\t"
         "add %%"REG_a", %1             \n\t"
         "add %%"REG_a", %2             \n\t"
         "subl $4, %0                   \n\t"
         "jnz 1b                        \n\t"
         : "+g"(h), "+r" (pixels),  "+r" (block)
         : "r"((x86_reg)line_size)
         : "%"REG_a, "memory"
        );
}

static void put_pixels16_sse2(uint8_t *block, const uint8_t *pixels, int line_size, int h)
{
    asm volatile(
         "1:                            \n\t"
         "movdqu (%1), %%xmm0           \n\t"
         "movdqu (%1,%3), %%xmm1        \n\t"
         "movdqu (%1,%3,2), %%xmm2      \n\t"
         "movdqu (%1,%4), %%xmm3        \n\t"
         "movdqa %%xmm0, (%2)           \n\t"
         "movdqa %%xmm1, (%2,%3)        \n\t"
         "movdqa %%xmm2, (%2,%3,2)      \n\t"
         "movdqa %%xmm3, (%2,%4)        \n\t"
         "subl $4, %0                   \n\t"
         "lea (%1,%3,4), %1             \n\t"
         "lea (%2,%3,4), %2             \n\t"
         "jnz 1b                        \n\t"
         : "+g"(h), "+r" (pixels),  "+r" (block)
         : "r"((x86_reg)line_size), "r"((x86_reg)3L*line_size)
         : "memory"
        );
}

static void avg_pixels16_sse2(uint8_t *block, const uint8_t *pixels, int line_size, int h)
{
    asm volatile(
         "1:                            \n\t"
         "movdqu (%1), %%xmm0           \n\t"
         "movdqu (%1,%3), %%xmm1        \n\t"
         "movdqu (%1,%3,2), %%xmm2      \n\t"
         "movdqu (%1,%4), %%xmm3        \n\t"
         "pavgb  (%2), %%xmm0           \n\t"
         "pavgb  (%2,%3), %%xmm1        \n\t"
         "pavgb  (%2,%3,2), %%xmm2      \n\t"
         "pavgb  (%2,%4), %%xmm3        \n\t"
         "movdqa %%xmm0, (%2)           \n\t"
         "movdqa %%xmm1, (%2,%3)        \n\t"
         "movdqa %%xmm2, (%2,%3,2)      \n\t"
         "movdqa %%xmm3, (%2,%4)        \n\t"
         "subl $4, %0                   \n\t"
         "lea (%1,%3,4), %1             \n\t"
         "lea (%2,%3,4), %2             \n\t"
         "jnz 1b                        \n\t"
         : "+g"(h), "+r" (pixels),  "+r" (block)
         : "r"((x86_reg)line_size), "r"((x86_reg)3L*line_size)
         : "memory"
        );
}

static void clear_blocks_mmx(DCTELEM *blocks)
{
    asm volatile(
                "pxor %%mm7, %%mm7              \n\t"
                "mov $-128*6, %%"REG_a"         \n\t"
                "1:                             \n\t"
                "movq %%mm7, (%0, %%"REG_a")    \n\t"
                "movq %%mm7, 8(%0, %%"REG_a")   \n\t"
                "movq %%mm7, 16(%0, %%"REG_a")  \n\t"
                "movq %%mm7, 24(%0, %%"REG_a")  \n\t"
                "add $32, %%"REG_a"             \n\t"
                " js 1b                         \n\t"
                : : "r" (((uint8_t *)blocks)+128*6)
                : "%"REG_a
        );
}

static void add_bytes_mmx(uint8_t *dst, uint8_t *src, int w){
    x86_reg i=0;
    asm volatile(
        "jmp 2f                         \n\t"
        "1:                             \n\t"
        "movq  (%1, %0), %%mm0          \n\t"
        "movq  (%2, %0), %%mm1          \n\t"
        "paddb %%mm0, %%mm1             \n\t"
        "movq %%mm1, (%2, %0)           \n\t"
        "movq 8(%1, %0), %%mm0          \n\t"
        "movq 8(%2, %0), %%mm1          \n\t"
        "paddb %%mm0, %%mm1             \n\t"
        "movq %%mm1, 8(%2, %0)          \n\t"
        "add $16, %0                    \n\t"
        "2:                             \n\t"
        "cmp %3, %0                     \n\t"
        " js 1b                         \n\t"
        : "+r" (i)
        : "r"(src), "r"(dst), "r"((x86_reg)w-15)
    );
    for(; i<w; i++)
        dst[i+0] += src[i+0];
}

static void add_bytes_l2_mmx(uint8_t *dst, uint8_t *src1, uint8_t *src2, int w){
    x86_reg i=0;
    asm volatile(
        "jmp 2f                         \n\t"
        "1:                             \n\t"
        "movq   (%2, %0), %%mm0         \n\t"
        "movq  8(%2, %0), %%mm1         \n\t"
        "paddb  (%3, %0), %%mm0         \n\t"
        "paddb 8(%3, %0), %%mm1         \n\t"
        "movq %%mm0,  (%1, %0)          \n\t"
        "movq %%mm1, 8(%1, %0)          \n\t"
        "add $16, %0                    \n\t"
        "2:                             \n\t"
        "cmp %4, %0                     \n\t"
        " js 1b                         \n\t"
        : "+r" (i)
        : "r"(dst), "r"(src1), "r"(src2), "r"((x86_reg)w-15)
    );
    for(; i<w; i++)
        dst[i] = src1[i] + src2[i];
}

static inline void transpose4x4(uint8_t *dst, uint8_t *src, int dst_stride, int src_stride){
    asm volatile( //FIXME could save 1 instruction if done as 8x4 ...
        "movd  %4, %%mm0                \n\t"
        "movd  %5, %%mm1                \n\t"
        "movd  %6, %%mm2                \n\t"
        "movd  %7, %%mm3                \n\t"
        "punpcklbw %%mm1, %%mm0         \n\t"
        "punpcklbw %%mm3, %%mm2         \n\t"
        "movq %%mm0, %%mm1              \n\t"
        "punpcklwd %%mm2, %%mm0         \n\t"
        "punpckhwd %%mm2, %%mm1         \n\t"
        "movd  %%mm0, %0                \n\t"
        "punpckhdq %%mm0, %%mm0         \n\t"
        "movd  %%mm0, %1                \n\t"
        "movd  %%mm1, %2                \n\t"
        "punpckhdq %%mm1, %%mm1         \n\t"
        "movd  %%mm1, %3                \n\t"

        : "=m" (*(uint32_t*)(dst + 0*dst_stride)),
          "=m" (*(uint32_t*)(dst + 1*dst_stride)),
          "=m" (*(uint32_t*)(dst + 2*dst_stride)),
          "=m" (*(uint32_t*)(dst + 3*dst_stride))
        :  "m" (*(uint32_t*)(src + 0*src_stride)),
           "m" (*(uint32_t*)(src + 1*src_stride)),
           "m" (*(uint32_t*)(src + 2*src_stride)),
           "m" (*(uint32_t*)(src + 3*src_stride))
    );
}

/* draw the edges of width 'w' of an image of size width, height
   this mmx version can only handle w==8 || w==16 */
static void draw_edges_mmx(uint8_t *buf, int wrap, int width, int height, int w)
{
    uint8_t *ptr, *last_line;
    int i;

    last_line = buf + (height - 1) * wrap;
    /* left and right */
    ptr = buf;
    if(w==8)
    {
        asm volatile(
                "1:                             \n\t"
                "movd (%0), %%mm0               \n\t"
                "punpcklbw %%mm0, %%mm0         \n\t"
                "punpcklwd %%mm0, %%mm0         \n\t"
                "punpckldq %%mm0, %%mm0         \n\t"
                "movq %%mm0, -8(%0)             \n\t"
                "movq -8(%0, %2), %%mm1         \n\t"
                "punpckhbw %%mm1, %%mm1         \n\t"
                "punpckhwd %%mm1, %%mm1         \n\t"
                "punpckhdq %%mm1, %%mm1         \n\t"
                "movq %%mm1, (%0, %2)           \n\t"
                "add %1, %0                     \n\t"
                "cmp %3, %0                     \n\t"
                " jb 1b                         \n\t"
                : "+r" (ptr)
                : "r" ((x86_reg)wrap), "r" ((x86_reg)width), "r" (ptr + wrap*height)
        );
    }
    else
    {
        asm volatile(
                "1:                             \n\t"
                "movd (%0), %%mm0               \n\t"
                "punpcklbw %%mm0, %%mm0         \n\t"
                "punpcklwd %%mm0, %%mm0         \n\t"
                "punpckldq %%mm0, %%mm0         \n\t"
                "movq %%mm0, -8(%0)             \n\t"
                "movq %%mm0, -16(%0)            \n\t"
                "movq -8(%0, %2), %%mm1         \n\t"
                "punpckhbw %%mm1, %%mm1         \n\t"
                "punpckhwd %%mm1, %%mm1         \n\t"
                "punpckhdq %%mm1, %%mm1         \n\t"
                "movq %%mm1, (%0, %2)           \n\t"
                "movq %%mm1, 8(%0, %2)          \n\t"
                "add %1, %0                     \n\t"
                "cmp %3, %0                     \n\t"
                " jb 1b                         \n\t"
                : "+r" (ptr)
                : "r" ((x86_reg)wrap), "r" ((x86_reg)width), "r" (ptr + wrap*height)
        );
    }

    for(i=0;i<w;i+=4) {
        /* top and bottom (and hopefully also the corners) */
        ptr= buf - (i + 1) * wrap - w;
        asm volatile(
                "1:                             \n\t"
                "movq (%1, %0), %%mm0           \n\t"
                "movq %%mm0, (%0)               \n\t"
                "movq %%mm0, (%0, %2)           \n\t"
                "movq %%mm0, (%0, %2, 2)        \n\t"
                "movq %%mm0, (%0, %3)           \n\t"
                "add $8, %0                     \n\t"
                "cmp %4, %0                     \n\t"
                " jb 1b                         \n\t"
                : "+r" (ptr)
                : "r" ((x86_reg)buf - (x86_reg)ptr - w), "r" ((x86_reg)-wrap), "r" ((x86_reg)-wrap*3), "r" (ptr+width+2*w)
        );
        ptr= last_line + (i + 1) * wrap - w;
        asm volatile(
                "1:                             \n\t"
                "movq (%1, %0), %%mm0           \n\t"
                "movq %%mm0, (%0)               \n\t"
                "movq %%mm0, (%0, %2)           \n\t"
                "movq %%mm0, (%0, %2, 2)        \n\t"
                "movq %%mm0, (%0, %3)           \n\t"
                "add $8, %0                     \n\t"
                "cmp %4, %0                     \n\t"
                " jb 1b                         \n\t"
                : "+r" (ptr)
                : "r" ((x86_reg)last_line - (x86_reg)ptr - w), "r" ((x86_reg)wrap), "r" ((x86_reg)wrap*3), "r" (ptr+width+2*w)
        );
    }
}

#define PAETH(cpu, abs3)\
static void add_png_paeth_prediction_##cpu(uint8_t *dst, uint8_t *src, uint8_t *top, int w, int bpp)\
{\
    x86_reg i = -bpp;\
    x86_reg end = w-3;\
    asm volatile(\
        "pxor      %%mm7, %%mm7 \n"\
        "movd    (%1,%0), %%mm0 \n"\
        "movd    (%2,%0), %%mm1 \n"\
        "punpcklbw %%mm7, %%mm0 \n"\
        "punpcklbw %%mm7, %%mm1 \n"\
        "add       %4, %0 \n"\
        "1: \n"\
        "movq      %%mm1, %%mm2 \n"\
        "movd    (%2,%0), %%mm1 \n"\
        "movq      %%mm2, %%mm3 \n"\
        "punpcklbw %%mm7, %%mm1 \n"\
        "movq      %%mm2, %%mm4 \n"\
        "psubw     %%mm1, %%mm3 \n"\
        "psubw     %%mm0, %%mm4 \n"\
        "movq      %%mm3, %%mm5 \n"\
        "paddw     %%mm4, %%mm5 \n"\
        abs3\
        "movq      %%mm4, %%mm6 \n"\
        "pminsw    %%mm5, %%mm6 \n"\
        "pcmpgtw   %%mm6, %%mm3 \n"\
        "pcmpgtw   %%mm5, %%mm4 \n"\
        "movq      %%mm4, %%mm6 \n"\
        "pand      %%mm3, %%mm4 \n"\
        "pandn     %%mm3, %%mm6 \n"\
        "pandn     %%mm0, %%mm3 \n"\
        "movd    (%3,%0), %%mm0 \n"\
        "pand      %%mm1, %%mm6 \n"\
        "pand      %%mm4, %%mm2 \n"\
        "punpcklbw %%mm7, %%mm0 \n"\
        "movq      %6,    %%mm5 \n"\
        "paddw     %%mm6, %%mm0 \n"\
        "paddw     %%mm2, %%mm3 \n"\
        "paddw     %%mm3, %%mm0 \n"\
        "pand      %%mm5, %%mm0 \n"\
        "movq      %%mm0, %%mm3 \n"\
        "packuswb  %%mm3, %%mm3 \n"\
        "movd      %%mm3, (%1,%0) \n"\
        "add       %4, %0 \n"\
        "cmp       %5, %0 \n"\
        "jle 1b \n"\
        :"+r"(i)\
        :"r"(dst), "r"(top), "r"(src), "r"((x86_reg)bpp), "g"(end),\
         "m"(ff_pw_255)\
        :"memory"\
    );\
}

#define ABS3_MMX2\
        "psubw     %%mm5, %%mm7 \n"\
        "pmaxsw    %%mm7, %%mm5 \n"\
        "pxor      %%mm6, %%mm6 \n"\
        "pxor      %%mm7, %%mm7 \n"\
        "psubw     %%mm3, %%mm6 \n"\
        "psubw     %%mm4, %%mm7 \n"\
        "pmaxsw    %%mm6, %%mm3 \n"\
        "pmaxsw    %%mm7, %%mm4 \n"\
        "pxor      %%mm7, %%mm7 \n"

#define ABS3_SSSE3\
        "pabsw     %%mm3, %%mm3 \n"\
        "pabsw     %%mm4, %%mm4 \n"\
        "pabsw     %%mm5, %%mm5 \n"

PAETH(mmx2, ABS3_MMX2)
#ifdef HAVE_SSSE3
PAETH(ssse3, ABS3_SSSE3)
#endif

#define QPEL_V_LOW(m3,m4,m5,m6, pw_20, pw_3, rnd, in0, in1, in2, in7, out, OP)\
        "paddw " #m4 ", " #m3 "           \n\t" /* x1 */\
        "movq "MANGLE(ff_pw_20)", %%mm4   \n\t" /* 20 */\
        "pmullw " #m3 ", %%mm4            \n\t" /* 20x1 */\
        "movq "#in7", " #m3 "             \n\t" /* d */\
        "movq "#in0", %%mm5               \n\t" /* D */\
        "paddw " #m3 ", %%mm5             \n\t" /* x4 */\
        "psubw %%mm5, %%mm4               \n\t" /* 20x1 - x4 */\
        "movq "#in1", %%mm5               \n\t" /* C */\
        "movq "#in2", %%mm6               \n\t" /* B */\
        "paddw " #m6 ", %%mm5             \n\t" /* x3 */\
        "paddw " #m5 ", %%mm6             \n\t" /* x2 */\
        "paddw %%mm6, %%mm6               \n\t" /* 2x2 */\
        "psubw %%mm6, %%mm5               \n\t" /* -2x2 + x3 */\
        "pmullw "MANGLE(ff_pw_3)", %%mm5  \n\t" /* -6x2 + 3x3 */\
        "paddw " #rnd ", %%mm4            \n\t" /* x2 */\
        "paddw %%mm4, %%mm5               \n\t" /* 20x1 - 6x2 + 3x3 - x4 */\
        "psraw $5, %%mm5                  \n\t"\
        "packuswb %%mm5, %%mm5            \n\t"\
        OP(%%mm5, out, %%mm7, d)


#define PUT_OP(a,b,temp, size) "mov" #size " " #a ", " #b "        \n\t"
#define AVG_3DNOW_OP(a,b,temp, size) \
"mov" #size " " #b ", " #temp "   \n\t"\
"pavgusb " #temp ", " #a "        \n\t"\
"mov" #size " " #a ", " #b "      \n\t"
#define AVG_MMX2_OP(a,b,temp, size) \
"mov" #size " " #b ", " #temp "   \n\t"\
"pavgb " #temp ", " #a "          \n\t"\
"mov" #size " " #a ", " #b "      \n\t"


#if 0
static void just_return() { return; }
#endif

static void gmc_mmx(uint8_t *dst, uint8_t *src, int stride, int h, int ox, int oy,
                    int dxx, int dxy, int dyx, int dyy, int shift, int r, int width, int height){
    const int w = 8;
    const int ix = ox>>(16+shift);
    const int iy = oy>>(16+shift);
    const int oxs = ox>>4;
    const int oys = oy>>4;
    const int dxxs = dxx>>4;
    const int dxys = dxy>>4;
    const int dyxs = dyx>>4;
    const int dyys = dyy>>4;
    const uint16_t r4[4] = {r,r,r,r};
    const uint16_t dxy4[4] = {dxys,dxys,dxys,dxys};
    const uint16_t dyy4[4] = {dyys,dyys,dyys,dyys};
    const uint64_t shift2 = 2*shift;
    uint8_t edge_buf[(h+1)*stride];
    int x, y;

    const int dxw = (dxx-(1<<(16+shift)))*(w-1);
    const int dyh = (dyy-(1<<(16+shift)))*(h-1);
    const int dxh = dxy*(h-1);
    const int dyw = dyx*(w-1);
    if( // non-constant fullpel offset (3% of blocks)
        ((ox^(ox+dxw)) | (ox^(ox+dxh)) | (ox^(ox+dxw+dxh)) |
         (oy^(oy+dyw)) | (oy^(oy+dyh)) | (oy^(oy+dyw+dyh))) >> (16+shift)
        // uses more than 16 bits of subpel mv (only at huge resolution)
        || (dxx|dxy|dyx|dyy)&15 )
    {
        //FIXME could still use mmx for some of the rows
        ff_gmc_c(dst, src, stride, h, ox, oy, dxx, dxy, dyx, dyy, shift, r, width, height);
        return;
    }

    src += ix + iy*stride;
    if( (unsigned)ix >= width-w ||
        (unsigned)iy >= height-h )
    {
        ff_emulated_edge_mc(edge_buf, src, stride, w+1, h+1, ix, iy, width, height);
        src = edge_buf;
    }

    asm volatile(
        "movd         %0, %%mm6 \n\t"
        "pxor      %%mm7, %%mm7 \n\t"
        "punpcklwd %%mm6, %%mm6 \n\t"
        "punpcklwd %%mm6, %%mm6 \n\t"
        :: "r"(1<<shift)
    );

    for(x=0; x<w; x+=4){
        uint16_t dx4[4] = { oxs - dxys + dxxs*(x+0),
                            oxs - dxys + dxxs*(x+1),
                            oxs - dxys + dxxs*(x+2),
                            oxs - dxys + dxxs*(x+3) };
        uint16_t dy4[4] = { oys - dyys + dyxs*(x+0),
                            oys - dyys + dyxs*(x+1),
                            oys - dyys + dyxs*(x+2),
                            oys - dyys + dyxs*(x+3) };

        for(y=0; y<h; y++){
            asm volatile(
                "movq   %0,  %%mm4 \n\t"
                "movq   %1,  %%mm5 \n\t"
                "paddw  %2,  %%mm4 \n\t"
                "paddw  %3,  %%mm5 \n\t"
                "movq   %%mm4, %0  \n\t"
                "movq   %%mm5, %1  \n\t"
                "psrlw  $12, %%mm4 \n\t"
                "psrlw  $12, %%mm5 \n\t"
                : "+m"(*dx4), "+m"(*dy4)
                : "m"(*dxy4), "m"(*dyy4)
            );

            asm volatile(
                "movq   %%mm6, %%mm2 \n\t"
                "movq   %%mm6, %%mm1 \n\t"
                "psubw  %%mm4, %%mm2 \n\t"
                "psubw  %%mm5, %%mm1 \n\t"
                "movq   %%mm2, %%mm0 \n\t"
                "movq   %%mm4, %%mm3 \n\t"
                "pmullw %%mm1, %%mm0 \n\t" // (s-dx)*(s-dy)
                "pmullw %%mm5, %%mm3 \n\t" // dx*dy
                "pmullw %%mm5, %%mm2 \n\t" // (s-dx)*dy
                "pmullw %%mm4, %%mm1 \n\t" // dx*(s-dy)

                "movd   %4,    %%mm5 \n\t"
                "movd   %3,    %%mm4 \n\t"
                "punpcklbw %%mm7, %%mm5 \n\t"
                "punpcklbw %%mm7, %%mm4 \n\t"
                "pmullw %%mm5, %%mm3 \n\t" // src[1,1] * dx*dy
                "pmullw %%mm4, %%mm2 \n\t" // src[0,1] * (s-dx)*dy

                "movd   %2,    %%mm5 \n\t"
                "movd   %1,    %%mm4 \n\t"
                "punpcklbw %%mm7, %%mm5 \n\t"
                "punpcklbw %%mm7, %%mm4 \n\t"
                "pmullw %%mm5, %%mm1 \n\t" // src[1,0] * dx*(s-dy)
                "pmullw %%mm4, %%mm0 \n\t" // src[0,0] * (s-dx)*(s-dy)
                "paddw  %5,    %%mm1 \n\t"
                "paddw  %%mm3, %%mm2 \n\t"
                "paddw  %%mm1, %%mm0 \n\t"
                "paddw  %%mm2, %%mm0 \n\t"

                "psrlw    %6,    %%mm0 \n\t"
                "packuswb %%mm0, %%mm0 \n\t"
                "movd     %%mm0, %0    \n\t"

                : "=m"(dst[x+y*stride])
                : "m"(src[0]), "m"(src[1]),
                  "m"(src[stride]), "m"(src[stride+1]),
                  "m"(*r4), "m"(shift2)
            );
            src += stride;
        }
        src += 4-h*stride;
    }
}

#define PREFETCH(name, op) \
static void name(void *mem, int stride, int h){\
    const uint8_t *p= mem;\
    do{\
        asm volatile(#op" %0" :: "m"(*p));\
        p+= stride;\
    }while(--h);\
}
PREFETCH(prefetch_mmx2,  prefetcht0)
PREFETCH(prefetch_3dnow, prefetch)
#undef PREFETCH


/* external functions, from idct_mmx.c */
void ff_mmx_idct(DCTELEM *block);
void ff_mmxext_idct(DCTELEM *block);

/* XXX: those functions should be suppressed ASAP when all IDCTs are
   converted */
#ifdef CONFIG_GPL
static void ff_libmpeg2mmx_idct_put(uint8_t *dest, int line_size, DCTELEM *block)
{
    ff_mmx_idct (block);
    put_pixels_clamped_mmx(block, dest, line_size);
}
static void ff_libmpeg2mmx_idct_add(uint8_t *dest, int line_size, DCTELEM *block)
{
    ff_mmx_idct (block);
    add_pixels_clamped_mmx(block, dest, line_size);
}
static void ff_libmpeg2mmx2_idct_put(uint8_t *dest, int line_size, DCTELEM *block)
{
    ff_mmxext_idct (block);
    put_pixels_clamped_mmx(block, dest, line_size);
}
static void ff_libmpeg2mmx2_idct_add(uint8_t *dest, int line_size, DCTELEM *block)
{
    ff_mmxext_idct (block);
    add_pixels_clamped_mmx(block, dest, line_size);
}
#endif

static void vorbis_inverse_coupling_3dnow(float *mag, float *ang, int blocksize)
{
    int i;
    asm volatile("pxor %%mm7, %%mm7":);
    for(i=0; i<blocksize; i+=2) {
        asm volatile(
            "movq    %0,    %%mm0 \n\t"
            "movq    %1,    %%mm1 \n\t"
            "movq    %%mm0, %%mm2 \n\t"
            "movq    %%mm1, %%mm3 \n\t"
            "pfcmpge %%mm7, %%mm2 \n\t" // m <= 0.0
            "pfcmpge %%mm7, %%mm3 \n\t" // a <= 0.0
            "pslld   $31,   %%mm2 \n\t" // keep only the sign bit
            "pxor    %%mm2, %%mm1 \n\t"
            "movq    %%mm3, %%mm4 \n\t"
            "pand    %%mm1, %%mm3 \n\t"
            "pandn   %%mm1, %%mm4 \n\t"
            "pfadd   %%mm0, %%mm3 \n\t" // a = m + ((a<0) & (a ^ sign(m)))
            "pfsub   %%mm4, %%mm0 \n\t" // m = m + ((a>0) & (a ^ sign(m)))
            "movq    %%mm3, %1    \n\t"
            "movq    %%mm0, %0    \n\t"
            :"+m"(mag[i]), "+m"(ang[i])
            ::"memory"
        );
    }
    asm volatile("femms");
}
static void vorbis_inverse_coupling_sse(float *mag, float *ang, int blocksize)
{
    int i;

    asm volatile(
            "movaps  %0,     %%xmm5 \n\t"
        ::"m"(ff_pdw_80000000[0])
    );
    for(i=0; i<blocksize; i+=4) {
        asm volatile(
            "movaps  %0,     %%xmm0 \n\t"
            "movaps  %1,     %%xmm1 \n\t"
            "xorps   %%xmm2, %%xmm2 \n\t"
            "xorps   %%xmm3, %%xmm3 \n\t"
            "cmpleps %%xmm0, %%xmm2 \n\t" // m <= 0.0
            "cmpleps %%xmm1, %%xmm3 \n\t" // a <= 0.0
            "andps   %%xmm5, %%xmm2 \n\t" // keep only the sign bit
            "xorps   %%xmm2, %%xmm1 \n\t"
            "movaps  %%xmm3, %%xmm4 \n\t"
            "andps   %%xmm1, %%xmm3 \n\t"
            "andnps  %%xmm1, %%xmm4 \n\t"
            "addps   %%xmm0, %%xmm3 \n\t" // a = m + ((a<0) & (a ^ sign(m)))
            "subps   %%xmm4, %%xmm0 \n\t" // m = m + ((a>0) & (a ^ sign(m)))
            "movaps  %%xmm3, %1     \n\t"
            "movaps  %%xmm0, %0     \n\t"
            :"+m"(mag[i]), "+m"(ang[i])
            ::"memory"
        );
    }
}

#define IF1(x) x
#define IF0(x)

#define MIX5(mono,stereo)\
    asm volatile(\
        "movss          0(%2), %%xmm5 \n"\
        "movss          8(%2), %%xmm6 \n"\
        "movss         24(%2), %%xmm7 \n"\
        "shufps    $0, %%xmm5, %%xmm5 \n"\
        "shufps    $0, %%xmm6, %%xmm6 \n"\
        "shufps    $0, %%xmm7, %%xmm7 \n"\
        "1: \n"\
        "movaps       (%0,%1), %%xmm0 \n"\
        "movaps  0x400(%0,%1), %%xmm1 \n"\
        "movaps  0x800(%0,%1), %%xmm2 \n"\
        "movaps  0xc00(%0,%1), %%xmm3 \n"\
        "movaps 0x1000(%0,%1), %%xmm4 \n"\
        "mulps         %%xmm5, %%xmm0 \n"\
        "mulps         %%xmm6, %%xmm1 \n"\
        "mulps         %%xmm5, %%xmm2 \n"\
        "mulps         %%xmm7, %%xmm3 \n"\
        "mulps         %%xmm7, %%xmm4 \n"\
 stereo("addps         %%xmm1, %%xmm0 \n")\
        "addps         %%xmm1, %%xmm2 \n"\
        "addps         %%xmm3, %%xmm0 \n"\
        "addps         %%xmm4, %%xmm2 \n"\
   mono("addps         %%xmm2, %%xmm0 \n")\
        "movaps  %%xmm0,      (%0,%1) \n"\
 stereo("movaps  %%xmm2, 0x400(%0,%1) \n")\
        "add $16, %0 \n"\
        "jl 1b \n"\
        :"+&r"(i)\
        :"r"(samples[0]+len), "r"(matrix)\
        :"memory"\
    );

#define MIX_MISC(stereo)\
    asm volatile(\
        "1: \n"\
        "movaps  (%3,%0), %%xmm0 \n"\
 stereo("movaps   %%xmm0, %%xmm1 \n")\
        "mulps    %%xmm6, %%xmm0 \n"\
 stereo("mulps    %%xmm7, %%xmm1 \n")\
        "lea 1024(%3,%0), %1 \n"\
        "mov %5, %2 \n"\
        "2: \n"\
        "movaps   (%1),   %%xmm2 \n"\
 stereo("movaps   %%xmm2, %%xmm3 \n")\
        "mulps   (%4,%2), %%xmm2 \n"\
 stereo("mulps 16(%4,%2), %%xmm3 \n")\
        "addps    %%xmm2, %%xmm0 \n"\
 stereo("addps    %%xmm3, %%xmm1 \n")\
        "add $1024, %1 \n"\
        "add $32, %2 \n"\
        "jl 2b \n"\
        "movaps   %%xmm0,     (%3,%0) \n"\
 stereo("movaps   %%xmm1, 1024(%3,%0) \n")\
        "add $16, %0 \n"\
        "jl 1b \n"\
        :"+&r"(i), "=&r"(j), "=&r"(k)\
        :"r"(samples[0]+len), "r"(matrix_simd+in_ch), "g"((intptr_t)-32*(in_ch-1))\
        :"memory"\
    );

static void ac3_downmix_sse(float (*samples)[256], float (*matrix)[2], int out_ch, int in_ch, int len)
{
    int (*matrix_cmp)[2] = (int(*)[2])matrix;
    intptr_t i,j,k;

    i = -len*sizeof(float);
    if(in_ch == 5 && out_ch == 2 && !(matrix_cmp[0][1]|matrix_cmp[2][0]|matrix_cmp[3][1]|matrix_cmp[4][0]|(matrix_cmp[1][0]^matrix_cmp[1][1])|(matrix_cmp[0][0]^matrix_cmp[2][1]))) {
        MIX5(IF0,IF1);
    } else if(in_ch == 5 && out_ch == 1 && matrix_cmp[0][0]==matrix_cmp[2][0] && matrix_cmp[3][0]==matrix_cmp[4][0]) {
        MIX5(IF1,IF0);
    } else {
        DECLARE_ALIGNED_16(float, matrix_simd[in_ch][2][4]);
        j = 2*in_ch*sizeof(float);
        asm volatile(
            "1: \n"
            "sub $8, %0 \n"
            "movss     (%2,%0), %%xmm6 \n"
            "movss    4(%2,%0), %%xmm7 \n"
            "shufps $0, %%xmm6, %%xmm6 \n"
            "shufps $0, %%xmm7, %%xmm7 \n"
            "movaps %%xmm6,   (%1,%0,4) \n"
            "movaps %%xmm7, 16(%1,%0,4) \n"
            "jg 1b \n"
            :"+&r"(j)
            :"r"(matrix_simd), "r"(matrix)
            :"memory"
        );
        if(out_ch == 2) {
            MIX_MISC(IF1);
        } else {
            MIX_MISC(IF0);
        }
    }
}

static void vector_fmul_3dnow(float *dst, const float *src, int len){
    x86_reg i = (len-4)*4;
    asm volatile(
        "1: \n\t"
        "movq    (%1,%0), %%mm0 \n\t"
        "movq   8(%1,%0), %%mm1 \n\t"
        "pfmul   (%2,%0), %%mm0 \n\t"
        "pfmul  8(%2,%0), %%mm1 \n\t"
        "movq   %%mm0,  (%1,%0) \n\t"
        "movq   %%mm1, 8(%1,%0) \n\t"
        "sub  $16, %0 \n\t"
        "jge 1b \n\t"
        "femms  \n\t"
        :"+r"(i)
        :"r"(dst), "r"(src)
        :"memory"
    );
}
static void vector_fmul_sse(float *dst, const float *src, int len){
    x86_reg i = (len-8)*4;
    asm volatile(
        "1: \n\t"
        "movaps    (%1,%0), %%xmm0 \n\t"
        "movaps  16(%1,%0), %%xmm1 \n\t"
        "mulps     (%2,%0), %%xmm0 \n\t"
        "mulps   16(%2,%0), %%xmm1 \n\t"
        "movaps  %%xmm0,   (%1,%0) \n\t"
        "movaps  %%xmm1, 16(%1,%0) \n\t"
        "sub  $32, %0 \n\t"
        "jge 1b \n\t"
        :"+r"(i)
        :"r"(dst), "r"(src)
        :"memory"
    );
}

static void vector_fmul_reverse_3dnow2(float *dst, const float *src0, const float *src1, int len){
    x86_reg i = len*4-16;
    asm volatile(
        "1: \n\t"
        "pswapd   8(%1), %%mm0 \n\t"
        "pswapd    (%1), %%mm1 \n\t"
        "pfmul  (%3,%0), %%mm0 \n\t"
        "pfmul 8(%3,%0), %%mm1 \n\t"
        "movq  %%mm0,  (%2,%0) \n\t"
        "movq  %%mm1, 8(%2,%0) \n\t"
        "add   $16, %1 \n\t"
        "sub   $16, %0 \n\t"
        "jge   1b \n\t"
        :"+r"(i), "+r"(src1)
        :"r"(dst), "r"(src0)
    );
    asm volatile("femms");
}
static void vector_fmul_reverse_sse(float *dst, const float *src0, const float *src1, int len){
    x86_reg i = len*4-32;
    asm volatile(
        "1: \n\t"
        "movaps        16(%1), %%xmm0 \n\t"
        "movaps          (%1), %%xmm1 \n\t"
        "shufps $0x1b, %%xmm0, %%xmm0 \n\t"
        "shufps $0x1b, %%xmm1, %%xmm1 \n\t"
        "mulps        (%3,%0), %%xmm0 \n\t"
        "mulps      16(%3,%0), %%xmm1 \n\t"
        "movaps     %%xmm0,   (%2,%0) \n\t"
        "movaps     %%xmm1, 16(%2,%0) \n\t"
        "add    $32, %1 \n\t"
        "sub    $32, %0 \n\t"
        "jge    1b \n\t"
        :"+r"(i), "+r"(src1)
        :"r"(dst), "r"(src0)
    );
}

static void vector_fmul_add_add_3dnow(float *dst, const float *src0, const float *src1,
                                      const float *src2, int src3, int len, int step){
    x86_reg i = (len-4)*4;
    if(step == 2 && src3 == 0){
        dst += (len-4)*2;
        asm volatile(
            "1: \n\t"
            "movq   (%2,%0),  %%mm0 \n\t"
            "movq  8(%2,%0),  %%mm1 \n\t"
            "pfmul  (%3,%0),  %%mm0 \n\t"
            "pfmul 8(%3,%0),  %%mm1 \n\t"
            "pfadd  (%4,%0),  %%mm0 \n\t"
            "pfadd 8(%4,%0),  %%mm1 \n\t"
            "movd     %%mm0,   (%1) \n\t"
            "movd     %%mm1, 16(%1) \n\t"
            "psrlq      $32,  %%mm0 \n\t"
            "psrlq      $32,  %%mm1 \n\t"
            "movd     %%mm0,  8(%1) \n\t"
            "movd     %%mm1, 24(%1) \n\t"
            "sub  $32, %1 \n\t"
            "sub  $16, %0 \n\t"
            "jge  1b \n\t"
            :"+r"(i), "+r"(dst)
            :"r"(src0), "r"(src1), "r"(src2)
            :"memory"
        );
    }
    else if(step == 1 && src3 == 0){
        asm volatile(
            "1: \n\t"
            "movq    (%2,%0), %%mm0 \n\t"
            "movq   8(%2,%0), %%mm1 \n\t"
            "pfmul   (%3,%0), %%mm0 \n\t"
            "pfmul  8(%3,%0), %%mm1 \n\t"
            "pfadd   (%4,%0), %%mm0 \n\t"
            "pfadd  8(%4,%0), %%mm1 \n\t"
            "movq  %%mm0,   (%1,%0) \n\t"
            "movq  %%mm1,  8(%1,%0) \n\t"
            "sub  $16, %0 \n\t"
            "jge  1b \n\t"
            :"+r"(i)
            :"r"(dst), "r"(src0), "r"(src1), "r"(src2)
            :"memory"
        );
    }
    else
        ff_vector_fmul_add_add_c(dst, src0, src1, src2, src3, len, step);
    asm volatile("femms");
}
static void vector_fmul_add_add_sse(float *dst, const float *src0, const float *src1,
                                    const float *src2, int src3, int len, int step){
    x86_reg i = (len-8)*4;
    if(step == 2 && src3 == 0){
        dst += (len-8)*2;
        asm volatile(
            "1: \n\t"
            "movaps   (%2,%0), %%xmm0 \n\t"
            "movaps 16(%2,%0), %%xmm1 \n\t"
            "mulps    (%3,%0), %%xmm0 \n\t"
            "mulps  16(%3,%0), %%xmm1 \n\t"
            "addps    (%4,%0), %%xmm0 \n\t"
            "addps  16(%4,%0), %%xmm1 \n\t"
            "movss     %%xmm0,   (%1) \n\t"
            "movss     %%xmm1, 32(%1) \n\t"
            "movhlps   %%xmm0, %%xmm2 \n\t"
            "movhlps   %%xmm1, %%xmm3 \n\t"
            "movss     %%xmm2, 16(%1) \n\t"
            "movss     %%xmm3, 48(%1) \n\t"
            "shufps $0xb1, %%xmm0, %%xmm0 \n\t"
            "shufps $0xb1, %%xmm1, %%xmm1 \n\t"
            "movss     %%xmm0,  8(%1) \n\t"
            "movss     %%xmm1, 40(%1) \n\t"
            "movhlps   %%xmm0, %%xmm2 \n\t"
            "movhlps   %%xmm1, %%xmm3 \n\t"
            "movss     %%xmm2, 24(%1) \n\t"
            "movss     %%xmm3, 56(%1) \n\t"
            "sub  $64, %1 \n\t"
            "sub  $32, %0 \n\t"
            "jge  1b \n\t"
            :"+r"(i), "+r"(dst)
            :"r"(src0), "r"(src1), "r"(src2)
            :"memory"
        );
    }
    else if(step == 1 && src3 == 0){
        asm volatile(
            "1: \n\t"
            "movaps   (%2,%0), %%xmm0 \n\t"
            "movaps 16(%2,%0), %%xmm1 \n\t"
            "mulps    (%3,%0), %%xmm0 \n\t"
            "mulps  16(%3,%0), %%xmm1 \n\t"
            "addps    (%4,%0), %%xmm0 \n\t"
            "addps  16(%4,%0), %%xmm1 \n\t"
            "movaps %%xmm0,   (%1,%0) \n\t"
            "movaps %%xmm1, 16(%1,%0) \n\t"
            "sub  $32, %0 \n\t"
            "jge  1b \n\t"
            :"+r"(i)
            :"r"(dst), "r"(src0), "r"(src1), "r"(src2)
            :"memory"
        );
    }
    else
        ff_vector_fmul_add_add_c(dst, src0, src1, src2, src3, len, step);
}

static void vector_fmul_window_3dnow2(float *dst, const float *src0, const float *src1,
                                      const float *win, float add_bias, int len){
#ifdef HAVE_6REGS
    if(add_bias == 0){
        x86_reg i = -len*4;
        x86_reg j = len*4-8;
        asm volatile(
            "1: \n"
            "pswapd  (%5,%1), %%mm1 \n"
            "movq    (%5,%0), %%mm0 \n"
            "pswapd  (%4,%1), %%mm5 \n"
            "movq    (%3,%0), %%mm4 \n"
            "movq      %%mm0, %%mm2 \n"
            "movq      %%mm1, %%mm3 \n"
            "pfmul     %%mm4, %%mm2 \n" // src0[len+i]*win[len+i]
            "pfmul     %%mm5, %%mm3 \n" // src1[    j]*win[len+j]
            "pfmul     %%mm4, %%mm1 \n" // src0[len+i]*win[len+j]
            "pfmul     %%mm5, %%mm0 \n" // src1[    j]*win[len+i]
            "pfadd     %%mm3, %%mm2 \n"
            "pfsub     %%mm0, %%mm1 \n"
            "pswapd    %%mm2, %%mm2 \n"
            "movq      %%mm1, (%2,%0) \n"
            "movq      %%mm2, (%2,%1) \n"
            "sub $8, %1 \n"
            "add $8, %0 \n"
            "jl 1b \n"
            "femms \n"
            :"+r"(i), "+r"(j)
            :"r"(dst+len), "r"(src0+len), "r"(src1), "r"(win+len)
        );
    }else
#endif
        ff_vector_fmul_window_c(dst, src0, src1, win, add_bias, len);
}

static void vector_fmul_window_sse(float *dst, const float *src0, const float *src1,
                                   const float *win, float add_bias, int len){
#ifdef HAVE_6REGS
    if(add_bias == 0){
        x86_reg i = -len*4;
        x86_reg j = len*4-16;
        asm volatile(
            "1: \n"
            "movaps       (%5,%1), %%xmm1 \n"
            "movaps       (%5,%0), %%xmm0 \n"
            "movaps       (%4,%1), %%xmm5 \n"
            "movaps       (%3,%0), %%xmm4 \n"
            "shufps $0x1b, %%xmm1, %%xmm1 \n"
            "shufps $0x1b, %%xmm5, %%xmm5 \n"
            "movaps        %%xmm0, %%xmm2 \n"
            "movaps        %%xmm1, %%xmm3 \n"
            "mulps         %%xmm4, %%xmm2 \n" // src0[len+i]*win[len+i]
            "mulps         %%xmm5, %%xmm3 \n" // src1[    j]*win[len+j]
            "mulps         %%xmm4, %%xmm1 \n" // src0[len+i]*win[len+j]
            "mulps         %%xmm5, %%xmm0 \n" // src1[    j]*win[len+i]
            "addps         %%xmm3, %%xmm2 \n"
            "subps         %%xmm0, %%xmm1 \n"
            "shufps $0x1b, %%xmm2, %%xmm2 \n"
            "movaps        %%xmm1, (%2,%0) \n"
            "movaps        %%xmm2, (%2,%1) \n"
            "sub $16, %1 \n"
            "add $16, %0 \n"
            "jl 1b \n"
            :"+r"(i), "+r"(j)
            :"r"(dst+len), "r"(src0+len), "r"(src1), "r"(win+len)
        );
    }else
#endif
        ff_vector_fmul_window_c(dst, src0, src1, win, add_bias, len);
}

static void int32_to_float_fmul_scalar_sse(float *dst, const int *src, float mul, int len)
{
    x86_reg i = -4*len;
    asm volatile(
        "movss  %3, %%xmm4 \n"
        "shufps $0, %%xmm4, %%xmm4 \n"
        "1: \n"
        "cvtpi2ps   (%2,%0), %%xmm0 \n"
        "cvtpi2ps  8(%2,%0), %%xmm1 \n"
        "cvtpi2ps 16(%2,%0), %%xmm2 \n"
        "cvtpi2ps 24(%2,%0), %%xmm3 \n"
        "movlhps  %%xmm1,    %%xmm0 \n"
        "movlhps  %%xmm3,    %%xmm2 \n"
        "mulps    %%xmm4,    %%xmm0 \n"
        "mulps    %%xmm4,    %%xmm2 \n"
        "movaps   %%xmm0,   (%1,%0) \n"
        "movaps   %%xmm2, 16(%1,%0) \n"
        "add $32, %0 \n"
        "jl 1b \n"
        :"+r"(i)
        :"r"(dst+len), "r"(src+len), "m"(mul)
    );
}

static void int32_to_float_fmul_scalar_sse2(float *dst, const int *src, float mul, int len)
{
    x86_reg i = -4*len;
    asm volatile(
        "movss  %3, %%xmm4 \n"
        "shufps $0, %%xmm4, %%xmm4 \n"
        "1: \n"
        "cvtdq2ps   (%2,%0), %%xmm0 \n"
        "cvtdq2ps 16(%2,%0), %%xmm1 \n"
        "mulps    %%xmm4,    %%xmm0 \n"
        "mulps    %%xmm4,    %%xmm1 \n"
        "movaps   %%xmm0,   (%1,%0) \n"
        "movaps   %%xmm1, 16(%1,%0) \n"
        "add $32, %0 \n"
        "jl 1b \n"
        :"+r"(i)
        :"r"(dst+len), "r"(src+len), "m"(mul)
    );
}

static void float_to_int16_3dnow(int16_t *dst, const float *src, long len){
    // not bit-exact: pf2id uses different rounding than C and SSE
    asm volatile(
        "add        %0          , %0        \n\t"
        "lea         (%2,%0,2)  , %2        \n\t"
        "add        %0          , %1        \n\t"
        "neg        %0                      \n\t"
        "1:                                 \n\t"
        "pf2id       (%2,%0,2)  , %%mm0     \n\t"
        "pf2id      8(%2,%0,2)  , %%mm1     \n\t"
        "pf2id     16(%2,%0,2)  , %%mm2     \n\t"
        "pf2id     24(%2,%0,2)  , %%mm3     \n\t"
        "packssdw   %%mm1       , %%mm0     \n\t"
        "packssdw   %%mm3       , %%mm2     \n\t"
        "movq       %%mm0       ,  (%1,%0)  \n\t"
        "movq       %%mm2       , 8(%1,%0)  \n\t"
        "add        $16         , %0        \n\t"
        " js 1b                             \n\t"
        "femms                              \n\t"
        :"+r"(len), "+r"(dst), "+r"(src)
    );
}
static void float_to_int16_sse(int16_t *dst, const float *src, long len){
    asm volatile(
        "add        %0          , %0        \n\t"
        "lea         (%2,%0,2)  , %2        \n\t"
        "add        %0          , %1        \n\t"
        "neg        %0                      \n\t"
        "1:                                 \n\t"
        "cvtps2pi    (%2,%0,2)  , %%mm0     \n\t"
        "cvtps2pi   8(%2,%0,2)  , %%mm1     \n\t"
        "cvtps2pi  16(%2,%0,2)  , %%mm2     \n\t"
        "cvtps2pi  24(%2,%0,2)  , %%mm3     \n\t"
        "packssdw   %%mm1       , %%mm0     \n\t"
        "packssdw   %%mm3       , %%mm2     \n\t"
        "movq       %%mm0       ,  (%1,%0)  \n\t"
        "movq       %%mm2       , 8(%1,%0)  \n\t"
        "add        $16         , %0        \n\t"
        " js 1b                             \n\t"
        "emms                               \n\t"
        :"+r"(len), "+r"(dst), "+r"(src)
    );
}

static void float_to_int16_sse2(int16_t *dst, const float *src, long len){
    asm volatile(
        "add        %0          , %0        \n\t"
        "lea         (%2,%0,2)  , %2        \n\t"
        "add        %0          , %1        \n\t"
        "neg        %0                      \n\t"
        "1:                                 \n\t"
        "cvtps2dq    (%2,%0,2)  , %%xmm0    \n\t"
        "cvtps2dq  16(%2,%0,2)  , %%xmm1    \n\t"
        "packssdw   %%xmm1      , %%xmm0    \n\t"
        "movdqa     %%xmm0      ,  (%1,%0)  \n\t"
        "add        $16         , %0        \n\t"
        " js 1b                             \n\t"
        :"+r"(len), "+r"(dst), "+r"(src)
    );
}

#ifdef HAVE_YASM
void ff_float_to_int16_interleave6_sse(int16_t *dst, const float **src, int len);
void ff_float_to_int16_interleave6_3dnow(int16_t *dst, const float **src, int len);
void ff_float_to_int16_interleave6_3dn2(int16_t *dst, const float **src, int len);
#else
#define ff_float_to_int16_interleave6_sse(a,b,c)   float_to_int16_interleave_misc_sse(a,b,c,6)
#define ff_float_to_int16_interleave6_3dnow(a,b,c) float_to_int16_interleave_misc_3dnow(a,b,c,6)
#define ff_float_to_int16_interleave6_3dn2(a,b,c)  float_to_int16_interleave_misc_3dnow(a,b,c,6)
#endif
#define ff_float_to_int16_interleave6_sse2 ff_float_to_int16_interleave6_sse

#define FLOAT_TO_INT16_INTERLEAVE(cpu, body) \
/* gcc pessimizes register allocation if this is in the same function as float_to_int16_interleave_sse2*/\
static av_noinline void float_to_int16_interleave_misc_##cpu(int16_t *dst, const float **src, long len, int channels){\
    DECLARE_ALIGNED_16(int16_t, tmp[len]);\
    int i,j,c;\
    for(c=0; c<channels; c++){\
        float_to_int16_##cpu(tmp, src[c], len);\
        for(i=0, j=c; i<len; i++, j+=channels)\
            dst[j] = tmp[i];\
    }\
}\
\
static void float_to_int16_interleave_##cpu(int16_t *dst, const float **src, long len, int channels){\
    if(channels==1)\
        float_to_int16_##cpu(dst, src[0], len);\
    else if(channels==2){\
        const float *src0 = src[0];\
        const float *src1 = src[1];\
        asm volatile(\
            "shl $2, %0 \n"\
            "add %0, %1 \n"\
            "add %0, %2 \n"\
            "add %0, %3 \n"\
            "neg %0 \n"\
            body\
            :"+r"(len), "+r"(dst), "+r"(src0), "+r"(src1)\
        );\
    }else if(channels==6){\
        ff_float_to_int16_interleave6_##cpu(dst, src, len);\
    }else\
        float_to_int16_interleave_misc_##cpu(dst, src, len, channels);\
}

FLOAT_TO_INT16_INTERLEAVE(3dnow,
    "1:                         \n"
    "pf2id     (%2,%0), %%mm0   \n"
    "pf2id    8(%2,%0), %%mm1   \n"
    "pf2id     (%3,%0), %%mm2   \n"
    "pf2id    8(%3,%0), %%mm3   \n"
    "packssdw    %%mm1, %%mm0   \n"
    "packssdw    %%mm3, %%mm2   \n"
    "movq        %%mm0, %%mm1   \n"
    "punpcklwd   %%mm2, %%mm0   \n"
    "punpckhwd   %%mm2, %%mm1   \n"
    "movq        %%mm0,  (%1,%0)\n"
    "movq        %%mm1, 8(%1,%0)\n"
    "add $16, %0                \n"
    "js 1b                      \n"
    "femms                      \n"
)

FLOAT_TO_INT16_INTERLEAVE(sse,
    "1:                         \n"
    "cvtps2pi  (%2,%0), %%mm0   \n"
    "cvtps2pi 8(%2,%0), %%mm1   \n"
    "cvtps2pi  (%3,%0), %%mm2   \n"
    "cvtps2pi 8(%3,%0), %%mm3   \n"
    "packssdw    %%mm1, %%mm0   \n"
    "packssdw    %%mm3, %%mm2   \n"
    "movq        %%mm0, %%mm1   \n"
    "punpcklwd   %%mm2, %%mm0   \n"
    "punpckhwd   %%mm2, %%mm1   \n"
    "movq        %%mm0,  (%1,%0)\n"
    "movq        %%mm1, 8(%1,%0)\n"
    "add $16, %0                \n"
    "js 1b                      \n"
    "emms                       \n"
)

FLOAT_TO_INT16_INTERLEAVE(sse2,
    "1:                         \n"
    "cvtps2dq  (%2,%0), %%xmm0  \n"
    "cvtps2dq  (%3,%0), %%xmm1  \n"
    "packssdw   %%xmm1, %%xmm0  \n"
    "movhlps    %%xmm0, %%xmm1  \n"
    "punpcklwd  %%xmm1, %%xmm0  \n"
    "movdqa     %%xmm0, (%1,%0) \n"
    "add $16, %0                \n"
    "js 1b                      \n"
)

static void float_to_int16_interleave_3dn2(int16_t *dst, const float **src, long len, int channels){
    if(channels==6)
        ff_float_to_int16_interleave6_3dn2(dst, src, len);
    else
        float_to_int16_interleave_3dnow(dst, src, len, channels);
}


extern void ff_snow_horizontal_compose97i_sse2(IDWTELEM *b, int width);
extern void ff_snow_horizontal_compose97i_mmx(IDWTELEM *b, int width);
extern void ff_snow_vertical_compose97i_sse2(IDWTELEM *b0, IDWTELEM *b1, IDWTELEM *b2, IDWTELEM *b3, IDWTELEM *b4, IDWTELEM *b5, int width);
extern void ff_snow_vertical_compose97i_mmx(IDWTELEM *b0, IDWTELEM *b1, IDWTELEM *b2, IDWTELEM *b3, IDWTELEM *b4, IDWTELEM *b5, int width);
extern void ff_snow_inner_add_yblock_sse2(const uint8_t *obmc, const int obmc_stride, uint8_t * * block, int b_w, int b_h,
                           int src_x, int src_y, int src_stride, slice_buffer * sb, int add, uint8_t * dst8);
extern void ff_snow_inner_add_yblock_mmx(const uint8_t *obmc, const int obmc_stride, uint8_t * * block, int b_w, int b_h,
                          int src_x, int src_y, int src_stride, slice_buffer * sb, int add, uint8_t * dst8);


static void add_int16_sse2(int16_t * v1, int16_t * v2, int order)
{
    x86_reg o = -(order << 1);
    v1 += order;
    v2 += order;
    asm volatile(
        "1:                          \n\t"
        "movdqu   (%1,%2),   %%xmm0  \n\t"
        "movdqu 16(%1,%2),   %%xmm1  \n\t"
        "paddw    (%0,%2),   %%xmm0  \n\t"
        "paddw  16(%0,%2),   %%xmm1  \n\t"
        "movdqa   %%xmm0,    (%0,%2) \n\t"
        "movdqa   %%xmm1,  16(%0,%2) \n\t"
        "add      $32,       %2      \n\t"
        "js       1b                 \n\t"
        : "+r"(v1), "+r"(v2), "+r"(o)
    );
}

static void sub_int16_sse2(int16_t * v1, int16_t * v2, int order)
{
    x86_reg o = -(order << 1);
    v1 += order;
    v2 += order;
    asm volatile(
        "1:                           \n\t"
        "movdqa    (%0,%2),   %%xmm0  \n\t"
        "movdqa  16(%0,%2),   %%xmm2  \n\t"
        "movdqu    (%1,%2),   %%xmm1  \n\t"
        "movdqu  16(%1,%2),   %%xmm3  \n\t"
        "psubw     %%xmm1,    %%xmm0  \n\t"
        "psubw     %%xmm3,    %%xmm2  \n\t"
        "movdqa    %%xmm0,    (%0,%2) \n\t"
        "movdqa    %%xmm2,  16(%0,%2) \n\t"
        "add       $32,       %2      \n\t"
        "js        1b                 \n\t"
        : "+r"(v1), "+r"(v2), "+r"(o)
    );
}

static int32_t scalarproduct_int16_sse2(int16_t * v1, int16_t * v2, int order, int shift)
{
    int res = 0;
    DECLARE_ALIGNED_16(int64_t, sh);
    x86_reg o = -(order << 1);

    v1 += order;
    v2 += order;
    sh = shift;
    asm volatile(
        "pxor      %%xmm7,  %%xmm7        \n\t"
        "1:                               \n\t"
        "movdqu    (%0,%3), %%xmm0        \n\t"
        "movdqu  16(%0,%3), %%xmm1        \n\t"
        "pmaddwd   (%1,%3), %%xmm0        \n\t"
        "pmaddwd 16(%1,%3), %%xmm1        \n\t"
        "paddd     %%xmm0,  %%xmm7        \n\t"
        "paddd     %%xmm1,  %%xmm7        \n\t"
        "add       $32,     %3            \n\t"
        "js        1b                     \n\t"
        "movhlps   %%xmm7,  %%xmm2        \n\t"
        "paddd     %%xmm2,  %%xmm7        \n\t"
        "psrad     %4,      %%xmm7        \n\t"
        "pshuflw   $0x4E,   %%xmm7,%%xmm2 \n\t"
        "paddd     %%xmm2,  %%xmm7        \n\t"
        "movd      %%xmm7,  %2            \n\t"
        : "+r"(v1), "+r"(v2), "=r"(res), "+r"(o)
        : "m"(sh)
    );
    return res;
}

void dsputil_init_mmx(DSPContext* c, AVCodecContext *avctx)
{
    mm_flags = mm_support();

    if (avctx->dsp_mask) {
        if (avctx->dsp_mask & FF_MM_FORCE)
            mm_flags |= (avctx->dsp_mask & 0xffff);
        else
            mm_flags &= ~(avctx->dsp_mask & 0xffff);
    }

#if 0
    av_log(avctx, AV_LOG_INFO, "libavcodec: CPU flags:");
    if (mm_flags & MM_MMX)
        av_log(avctx, AV_LOG_INFO, " mmx");
    if (mm_flags & MM_MMXEXT)
        av_log(avctx, AV_LOG_INFO, " mmxext");
    if (mm_flags & MM_3DNOW)
        av_log(avctx, AV_LOG_INFO, " 3dnow");
    if (mm_flags & MM_SSE)
        av_log(avctx, AV_LOG_INFO, " sse");
    if (mm_flags & MM_SSE2)
        av_log(avctx, AV_LOG_INFO, " sse2");
    av_log(avctx, AV_LOG_INFO, "\n");
#endif

    if (mm_flags & MM_MMX) {
        const int idct_algo= avctx->idct_algo;

        if(avctx->lowres==0){
            if(idct_algo==FF_IDCT_AUTO || idct_algo==FF_IDCT_SIMPLEMMX){
                c->idct_put= ff_simple_idct_put_mmx;
                c->idct_add= ff_simple_idct_add_mmx;
                c->idct    = ff_simple_idct_mmx;
                c->idct_permutation_type= FF_SIMPLE_IDCT_PERM;
#ifdef CONFIG_GPL
            }else if(idct_algo==FF_IDCT_LIBMPEG2MMX){
                if(mm_flags & MM_MMXEXT){
                    c->idct_put= ff_libmpeg2mmx2_idct_put;
                    c->idct_add= ff_libmpeg2mmx2_idct_add;
                    c->idct    = ff_mmxext_idct;
                }else{
                    c->idct_put= ff_libmpeg2mmx_idct_put;
                    c->idct_add= ff_libmpeg2mmx_idct_add;
                    c->idct    = ff_mmx_idct;
                }
                c->idct_permutation_type= FF_LIBMPEG2_IDCT_PERM;
#endif
            }else if(idct_algo==FF_IDCT_CAVS){
                    c->idct_permutation_type= FF_TRANSPOSE_IDCT_PERM;
            }
        }

        c->put_pixels_clamped = put_pixels_clamped_mmx;
        c->put_signed_pixels_clamped = put_signed_pixels_clamped_mmx;
        c->add_pixels_clamped = add_pixels_clamped_mmx;
        c->clear_blocks = clear_blocks_mmx;

#define SET_HPEL_FUNCS(PFX, IDX, SIZE, CPU) \
        c->PFX ## _pixels_tab[IDX][0] = PFX ## _pixels ## SIZE ## _ ## CPU; \
        c->PFX ## _pixels_tab[IDX][1] = PFX ## _pixels ## SIZE ## _x2_ ## CPU; \
        c->PFX ## _pixels_tab[IDX][2] = PFX ## _pixels ## SIZE ## _y2_ ## CPU; \
        c->PFX ## _pixels_tab[IDX][3] = PFX ## _pixels ## SIZE ## _xy2_ ## CPU

        SET_HPEL_FUNCS(put, 0, 16, mmx);
        SET_HPEL_FUNCS(put_no_rnd, 0, 16, mmx);
        SET_HPEL_FUNCS(avg, 0, 16, mmx);
        SET_HPEL_FUNCS(avg_no_rnd, 0, 16, mmx);
        SET_HPEL_FUNCS(put, 1, 8, mmx);
        SET_HPEL_FUNCS(put_no_rnd, 1, 8, mmx);
        SET_HPEL_FUNCS(avg, 1, 8, mmx);
        SET_HPEL_FUNCS(avg_no_rnd, 1, 8, mmx);

        c->gmc= gmc_mmx;

        c->add_bytes= add_bytes_mmx;
        c->add_bytes_l2= add_bytes_l2_mmx;

        c->draw_edges = draw_edges_mmx;

        if (mm_flags & MM_MMXEXT) {
            c->prefetch = prefetch_mmx2;

            c->put_pixels_tab[0][1] = put_pixels16_x2_mmx2;
            c->put_pixels_tab[0][2] = put_pixels16_y2_mmx2;

            c->avg_pixels_tab[0][0] = avg_pixels16_mmx2;
            c->avg_pixels_tab[0][1] = avg_pixels16_x2_mmx2;
            c->avg_pixels_tab[0][2] = avg_pixels16_y2_mmx2;

            c->put_pixels_tab[1][1] = put_pixels8_x2_mmx2;
            c->put_pixels_tab[1][2] = put_pixels8_y2_mmx2;

            c->avg_pixels_tab[1][0] = avg_pixels8_mmx2;
            c->avg_pixels_tab[1][1] = avg_pixels8_x2_mmx2;
            c->avg_pixels_tab[1][2] = avg_pixels8_y2_mmx2;

            if(!(avctx->flags & CODEC_FLAG_BITEXACT)){
                c->put_no_rnd_pixels_tab[0][1] = put_no_rnd_pixels16_x2_mmx2;
                c->put_no_rnd_pixels_tab[0][2] = put_no_rnd_pixels16_y2_mmx2;
                c->put_no_rnd_pixels_tab[1][1] = put_no_rnd_pixels8_x2_mmx2;
                c->put_no_rnd_pixels_tab[1][2] = put_no_rnd_pixels8_y2_mmx2;
                c->avg_pixels_tab[0][3] = avg_pixels16_xy2_mmx2;
                c->avg_pixels_tab[1][3] = avg_pixels8_xy2_mmx2;
            }

            c->add_png_paeth_prediction= add_png_paeth_prediction_mmx2;
        } else if (mm_flags & MM_3DNOW) {
            c->prefetch = prefetch_3dnow;

            c->put_pixels_tab[0][1] = put_pixels16_x2_3dnow;
            c->put_pixels_tab[0][2] = put_pixels16_y2_3dnow;

            c->avg_pixels_tab[0][0] = avg_pixels16_3dnow;
            c->avg_pixels_tab[0][1] = avg_pixels16_x2_3dnow;
            c->avg_pixels_tab[0][2] = avg_pixels16_y2_3dnow;

            c->put_pixels_tab[1][1] = put_pixels8_x2_3dnow;
            c->put_pixels_tab[1][2] = put_pixels8_y2_3dnow;

            c->avg_pixels_tab[1][0] = avg_pixels8_3dnow;
            c->avg_pixels_tab[1][1] = avg_pixels8_x2_3dnow;
            c->avg_pixels_tab[1][2] = avg_pixels8_y2_3dnow;

            if(!(avctx->flags & CODEC_FLAG_BITEXACT)){
                c->put_no_rnd_pixels_tab[0][1] = put_no_rnd_pixels16_x2_3dnow;
                c->put_no_rnd_pixels_tab[0][2] = put_no_rnd_pixels16_y2_3dnow;
                c->put_no_rnd_pixels_tab[1][1] = put_no_rnd_pixels8_x2_3dnow;
                c->put_no_rnd_pixels_tab[1][2] = put_no_rnd_pixels8_y2_3dnow;
                c->avg_pixels_tab[0][3] = avg_pixels16_xy2_3dnow;
                c->avg_pixels_tab[1][3] = avg_pixels8_xy2_3dnow;
            }
        }

#ifdef CONFIG_SNOW_DECODER
        if(mm_flags & MM_SSE2 & 0){
            c->horizontal_compose97i = ff_snow_horizontal_compose97i_sse2;
#ifdef HAVE_7REGS
            c->vertical_compose97i = ff_snow_vertical_compose97i_sse2;
#endif
            c->inner_add_yblock = ff_snow_inner_add_yblock_sse2;
        }
        else{
            if(mm_flags & MM_MMXEXT){
            c->horizontal_compose97i = ff_snow_horizontal_compose97i_mmx;
#ifdef HAVE_7REGS
            c->vertical_compose97i = ff_snow_vertical_compose97i_mmx;
#endif
            }
            c->inner_add_yblock = ff_snow_inner_add_yblock_mmx;
        }
#endif

        if(mm_flags & MM_3DNOW){
            c->vorbis_inverse_coupling = vorbis_inverse_coupling_3dnow;
            c->vector_fmul = vector_fmul_3dnow;
            if(!(avctx->flags & CODEC_FLAG_BITEXACT)){
                c->float_to_int16 = float_to_int16_3dnow;
                c->float_to_int16_interleave = float_to_int16_interleave_3dnow;
            }
        }
        if(mm_flags & MM_3DNOWEXT){
            c->vector_fmul_reverse = vector_fmul_reverse_3dnow2;
            c->vector_fmul_window = vector_fmul_window_3dnow2;
            if(!(avctx->flags & CODEC_FLAG_BITEXACT)){
                c->float_to_int16_interleave = float_to_int16_interleave_3dn2;
            }
        }
        if(mm_flags & MM_SSE){
            c->vorbis_inverse_coupling = vorbis_inverse_coupling_sse;
            c->ac3_downmix = ac3_downmix_sse;
            c->vector_fmul = vector_fmul_sse;
            c->vector_fmul_reverse = vector_fmul_reverse_sse;
            c->vector_fmul_add_add = vector_fmul_add_add_sse;
            c->vector_fmul_window = vector_fmul_window_sse;
            c->int32_to_float_fmul_scalar = int32_to_float_fmul_scalar_sse;
            c->float_to_int16 = float_to_int16_sse;
            c->float_to_int16_interleave = float_to_int16_interleave_sse;
        }
        if(mm_flags & MM_3DNOW)
            c->vector_fmul_add_add = vector_fmul_add_add_3dnow; // faster than sse
        if(mm_flags & MM_SSE2){
            c->int32_to_float_fmul_scalar = int32_to_float_fmul_scalar_sse2;
            c->float_to_int16 = float_to_int16_sse2;
            c->float_to_int16_interleave = float_to_int16_interleave_sse2;
            c->add_int16 = add_int16_sse2;
            c->sub_int16 = sub_int16_sse2;
            c->scalarproduct_int16 = scalarproduct_int16_sse2;
        }
    }

    if (ENABLE_ENCODERS)
        dsputilenc_init_mmx(c, avctx);

#if 0
    // for speed testing
    get_pixels = just_return;
    put_pixels_clamped = just_return;
    add_pixels_clamped = just_return;

    pix_abs16x16 = just_return;
    pix_abs16x16_x2 = just_return;
    pix_abs16x16_y2 = just_return;
    pix_abs16x16_xy2 = just_return;

    put_pixels_tab[0] = just_return;
    put_pixels_tab[1] = just_return;
    put_pixels_tab[2] = just_return;
    put_pixels_tab[3] = just_return;

    put_no_rnd_pixels_tab[0] = just_return;
    put_no_rnd_pixels_tab[1] = just_return;
    put_no_rnd_pixels_tab[2] = just_return;
    put_no_rnd_pixels_tab[3] = just_return;

    avg_pixels_tab[0] = just_return;
    avg_pixels_tab[1] = just_return;
    avg_pixels_tab[2] = just_return;
    avg_pixels_tab[3] = just_return;

    avg_no_rnd_pixels_tab[0] = just_return;
    avg_no_rnd_pixels_tab[1] = just_return;
    avg_no_rnd_pixels_tab[2] = just_return;
    avg_no_rnd_pixels_tab[3] = just_return;

    //av_fdct = just_return;
    //ff_idct = just_return;
#endif
}
