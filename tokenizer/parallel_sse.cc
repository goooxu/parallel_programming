#include <cstddef>
#include <omp.h>
#include <emmintrin.h>

static inline void tokenize_one(const char *begin, const char *end, int offset, const char **&tokens)
{
    const __m128i pattern = _mm_set1_epi16('\n' << 8 | '\r');
    const __m128i *mem = reinterpret_cast<const __m128i *>(begin + offset);

    for (; begin < end; ++mem, begin += 16)
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
    }
}

void tokenize(const char *begin, const char *end, const char **tokens, size_t max_tokens)
{
#pragma omp parallel for
    for (int i = 0; i < 2; i++)
    {
        tokenize_one(begin, end, i, tokens);
    }
}