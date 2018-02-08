#include "benchmark.h"

void tokenize(const char *begin, const char *end, const char **tokens)
{
    const int size = end - begin;
#pragma omp parallel for
    for (int i = 0; i < size - 1; ++i)
    {
        if (begin[i] == '\r' && begin[i + 1] == '\n')
        {
#pragma omp critical
            *tokens++ = begin + i + 2;
        }
    }
}