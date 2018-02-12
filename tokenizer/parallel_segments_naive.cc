#include <cstddef>
#include <omp.h>

static inline void tokenize_one(const char *begin, const char *end, const char **&tokens)
{
    for (const char *p = begin; p != end - 1; ++p)
    {
        if (*p == '\r' && *(p + 1) == '\n')
        {
#pragma omp critical
            *tokens++ = p + 2;
        }
    }
}

void tokenize(const char *begin, const char *end, const char **tokens, size_t max_tokens)
{
    int blocks = omp_get_max_threads();
    int block_size = (end - begin) / blocks;

#pragma omp parallel for
    for (int i = 0; i < blocks; i++)
    {
        tokenize_one(begin + i * block_size, begin + (i + 1) * block_size, tokens);
    }
}