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
        *tokens++ = begin + shift + offset;
    }
}

void tokenize(const char *begin, const char *end, const char **tokens)
{
    const __m256i pattern = _mm256_set1_epi16('\n' << 8 | '\r');
    const __m256i *mem0 = reinterpret_cast<const __m256i *>(begin);
    const __m256i *mem1 = reinterpret_cast<const __m256i *>(begin + 1);

    for (; begin < end; begin += 32)
    {
        foo(pattern, begin, mem0, 0, tokens);
        foo(pattern, begin, mem1, 1, tokens);
        ++mem0;
        ++mem1;
    }
}
