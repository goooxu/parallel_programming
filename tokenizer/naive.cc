#include "benchmark.h"

void tokenize(const char *begin, const char *end, char const **tokens)
{
    for (const char *p = begin; p != end - 1; ++p)
    {
        if (*p == '\r' && *(p + 1) == '\n')
            *tokens++ = p + 2;
    }
}