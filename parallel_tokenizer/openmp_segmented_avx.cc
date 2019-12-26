#include <omp.h>
#include <immintrin.h>

char *allocBuffer(size_t size)
{
    return (char *)aligned_alloc(sizeof(__m256i), size);
}

void freeBuffer(char *buffer)
{
    free(buffer);
}

static inline void foo(const __m256i &pattern, const __m256i *mem, size_t offset, size_t *tokens, size_t &token_index)
{
    const __m256i data = _mm256_loadu_si256(mem);
    const __m256i cmp = _mm256_cmpeq_epi16(data, pattern);
    unsigned int mask = _mm256_movemask_epi8(cmp);

    size_t shift = 0;
    while (mask != 0)
    {
        const int pad = __builtin_ctz(mask);
        mask >>= pad;
        mask >>= 2;
        shift += pad + 2;

        size_t index;
#pragma omp atomic capture
        index = token_index++;
        tokens[index] = offset + shift;
    }
}

static inline void tokenize_segment(const char *buffer, size_t buffer_size, size_t offset, size_t *tokens, size_t &token_index)
{
    const __m256i pattern = _mm256_set1_epi16('\n' << 8 | '\r');

    for (size_t i = 0; i < buffer_size; i += sizeof(__m256i))
    {
        foo(pattern, reinterpret_cast<const __m256i *>(buffer + i), offset + i, tokens, token_index);
        foo(pattern, reinterpret_cast<const __m256i *>(buffer + i + 1), offset + i + 1, tokens, token_index);
    }
}

void tokenize(const char *buffer, size_t buffer_size, size_t *tokens, size_t token_size)
{
    size_t segments = omp_get_max_threads();
    size_t segment_size = buffer_size / segments;
    size_t token_index = 0;

#pragma omp parallel for
    for (size_t i = 0; i < segments; i++)
    {
        if (i == segments - 1)
        {
            tokenize_segment(buffer + i * segment_size, buffer_size - (segments - 1) * segment_size, i * segment_size, tokens, token_index);
        }
        else
        {
            tokenize_segment(buffer + i * segment_size, segment_size, i * segment_size, tokens, token_index);
        }
    }
}
