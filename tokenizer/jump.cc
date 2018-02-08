#include "benchmark.h"

void tokenize(const char *begin, const char *end, const char **tokens)
{
    const int size = end - begin;

    if (size > 1 && *begin == '\r' && *(begin + 1) == '\n')
        *tokens++ = begin + 2;

    for (const char *p = begin + 2; p < end - 1; p += 2)
    {
        if (*p == '\r')
        {
            if (*(p + 1) == '\n')
                *tokens++ = p + 2;
        }
        else if (*p == '\n')
        {
            if (*(p - 1) == '\r')
                *tokens++ = p + 1;
        }
    }

    if (size > 1 && (size & 0x01) == 0x01)
    {
        if (*(end - 2) == '\r' && *(end - 1) == '\n')
            *tokens++ = end;
    }
}