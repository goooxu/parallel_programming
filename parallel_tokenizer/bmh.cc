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
	for (size_t i = 0, j = 1; j < buffer_size;)
	{
		if (buffer[j] == '\n')
		{
			if (buffer[i] == '\r')
			{
				*tokens++ = j + 1;
			}
			i += 1;
			j += 1;
		}
		else if (buffer[j] == '\r')
		{
			i += 1;
			j += 1;
		}
		else
		{
			i += 2;
			j += 2;
		}
	}
}