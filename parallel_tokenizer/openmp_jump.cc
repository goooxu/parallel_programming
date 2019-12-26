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
    size_t token_index = 0;

    if (buffer_size > 1 && buffer[0] == '\r' && buffer[1] == '\n')
        tokens[token_index++] = 2;

#pragma omp parallel for
    for (size_t i = 2; i < buffer_size; i += 2)
    {
        if (buffer[i] == '\r')
        {
            if (buffer[i + 1] == '\n')
            {
                size_t index;
#pragma omp atomic capture
                index = token_index++;
                tokens[index] = i + 2;
            }
        }
        else if (buffer[i] == '\n')
        {
            if (buffer[i - 1] == '\r')
            {
                size_t index;
#pragma omp atomic capture
                index = token_index++;
                tokens[index] = i + 1;
            }
        }
    }

    if (buffer_size > 1 && (buffer_size & 0x01) == 0x01)
    {
        if (buffer[buffer_size - 2] == '\r' && buffer[buffer_size - 1] == '\n')
            tokens[token_index++] = buffer_size;
    }
}