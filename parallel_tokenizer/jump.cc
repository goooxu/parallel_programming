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

void tokenize(const char *buffer, size_t buffer_size, size_t *tokens, size_t token_size)
{
    if (buffer_size > 1 && buffer[0] == '\r' && buffer[1] == '\n')
        *tokens++ = 2;

    for (size_t i = 2; i < buffer_size - 1; i += 2)
    {
        if (buffer[i] == '\r')
        {
            if (buffer[i + 1] == '\n')
                *tokens++ = i + 2;
        }
        else if (buffer[i] == '\n')
        {
            if (buffer[i - 1] == '\r')
                *tokens++ = i + 1;
        }
    }

    if (buffer_size > 1 && (buffer_size & 0x01) == 0x01)
    {
        if (buffer[buffer_size - 2] == '\r' && buffer[buffer_size - 1] == '\n')
            *tokens++ = buffer_size;
    }
}