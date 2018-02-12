#include <cstddef>

void tokenize(const char *begin, const char *end, const char **tokens, size_t max_tokens)
{
    const int size = end - begin;

    if (size > 1 && begin[0] == '\r' && begin[1] == '\n')
        *tokens++ = begin + 2;

#pragma omp parallel for
    for (int i = 2; i < size; i += 2)
    {
        if (begin[i] == '\r')
        {
            if (begin[i + 1] == '\n')
#pragma omp critical
                *tokens++ = begin + i + 2;
        }
        else if (begin[i] == '\n')
        {
            if (begin[i - 1] == '\r')
#pragma omp critical
                *tokens++ = begin + i + 1;
        }
    }

    if (size > 1 && (size & 0x01) == 0x01)
    {
        if (end[-2] == '\r' && end[-1] == '\n')
            *tokens++ = end;
    }
}