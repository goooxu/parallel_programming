#include "benchmark.h"
#include <immintrin.h>

static inline void foo(const __m256i &pattern, const char *begin, const __m256i *mem, int offset, const char **&tokens)
{
    const __m256i data = _mm256_loadu_si256(mem);
    const __m256i cmp = _mm256_cmpeq_epi16(data, pattern);
    unsigned int mask = _mm256_movemask_epi8(cmp);

    int shift = 0;
    while (mask != 0)
    {
        const int pad = __builtin_ctz(mask);
        mask >>= pad;
        mask >>= 2;
        shift += pad + 2;
#pragma omp critical
        *tokens++ = begin + shift + offset;
    }
}

void tokenize(const char *begin, const char *end, const char **tokens)
{
    const __m256i pattern = _mm256_set1_epi16('\n' << 8 | '\r');
    const __m256i *mem0 = reinterpret_cast<const __m256i *>(begin);
    const __m256i *mem1 = reinterpret_cast<const __m256i *>(begin + 1);

    const int blocks = (end - begin) / 32;

#pragma omp parallel for
    for (int i = 0; i < blocks; i++)
    {
        foo(pattern, begin + i * 32, mem0 + i, 0, tokens);
        foo(pattern, begin + i * 32, mem1 + i, 1, tokens);
    }
}