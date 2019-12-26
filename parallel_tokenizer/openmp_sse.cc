#include <emmintrin.h>

char *allocBuffer(size_t size)
{
    return (char *)aligned_alloc(sizeof(__m128i), size);
}

void freeBuffer(char *buffer)
{
    free(buffer);
}

static inline void foo(const __m128i &pattern, const __m128i *mem, size_t offset, size_t *tokens, size_t &token_index)
{
    const __m128i data = _mm_loadu_si128(mem);
    const __m128i cmp = _mm_cmpeq_epi16(data, pattern);
    unsigned int mask = _mm_movemask_epi8(cmp);

    size_t shift = 0;
    while (mask != 0)
    {
        const int pad = __builtin_ctz(mask) + 2;
        mask >>= pad;
        shift += pad;

        size_t index;
#pragma omp atomic capture
        index = token_index++;
        tokens[index] = offset + shift;
    }
}

void tokenize(const char *buffer, size_t buffer_size, size_t *tokens, size_t token_size)
{
    const __m128i pattern = _mm_set1_epi16('\n' << 8 | '\r');
    size_t token_index = 0;

#pragma omp parallel for
    for (size_t i = 0; i < buffer_size; i += sizeof(__m128i))
    {
        foo(pattern, reinterpret_cast<const __m128i *>(buffer + i), i, tokens, token_index);
        foo(pattern, reinterpret_cast<const __m128i *>(buffer + i + 1), i + 1, tokens, token_index);
    }
}