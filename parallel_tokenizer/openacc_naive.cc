#include <cstddef>
#include <cstdlib>

char *allocBuffer(size_t size)
{
    return (char *)malloc(size);
}

void freeBuffer(char *buffer)
{
    free(buffer);
}

void tokenize(const char *restrict buffer, size_t buffer_size, size_t *restrict tokens, size_t token_size)
{
    size_t token_index = 0;
#pragma acc parallel loop copyin(buffer [0:buffer_size]) copyin(token_index) copyout(tokens [0:token_size])
    for (size_t i = 0; i < buffer_size - 1; ++i)
    {
        if (buffer[i] == '\r' && buffer[i + 1] == '\n')
        {
            size_t index;
#pragma acc atomic capture
            {
                index = token_index;
                ++token_index;
            }
            tokens[index] = i + 2;
        }
    }
}
