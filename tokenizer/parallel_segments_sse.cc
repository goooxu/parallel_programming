#include "benchmark.h"
#include <omp.h>
#include <emmintrin.h>

static inline void foo(const __m128i &pattern, const char *begin, const __m128i *mem, int offset, const char **&tokens)
{
    const __m128i data = _mm_loadu_si128(mem);
    const __m128i cmp = _mm_cmpeq_epi16(data, pattern);
    unsigned int mask = _mm_movemask_epi8(cmp);

    int shift = 0;
    while (mask != 0)
    {
        const int pad = __builtin_ctz(mask) + 2;
        mask >>= pad;
        shift += pad;
#pragma omp critical
        *tokens++ = begin + shift + offset;
    }

    mem++;
}

static inline void tokenize_one(const char *begin, const char *end, const char **&tokens)
{
    const __m128i pattern = _mm_set1_epi16('\n' << 8 | '\r');
    const __m128i *mem0 = reinterpret_cast<const __m128i *>(begin);
    const __m128i *mem1 = reinterpret_cast<const __m128i *>(begin + 1);

    for (; begin < end; begin += 16)
    {
        foo(pattern, begin, mem0, 0, tokens);
        foo(pattern, begin, mem1, 1, tokens);
        ++mem0;
        ++mem1;
    }
}

void tokenize(const char *begin, const char *end, const char **tokens)
{
    int blocks = omp_get_max_threads();
    int block_size = (end - begin) / blocks;

#pragma omp parallel for
    for (int i = 0; i < blocks; i++)
    {
        tokenize_one(begin + i * block_size, begin + (i + 1) * block_size, tokens);
    }
}
