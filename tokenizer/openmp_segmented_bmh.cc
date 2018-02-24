#include <cstddef>
#include <cstdlib>
#include <omp.h>

char *allocBuffer(size_t size)
{
	return (char *)malloc(size);
}

void freeBuffer(char *buffer)
{
	free(buffer);
}

static inline void tokenize_segment(const char *buffer, size_t buffer_size, size_t offset, size_t *tokens, size_t &token_index)
{
	for (size_t i = 0, j = 1; j < buffer_size;)
	{
		if (buffer[j] == '\n')
		{
			if (buffer[i] == '\r')
			{
				size_t index;
#pragma omp atomic capture
				index = token_index++;
				tokens[index] = offset + j + 1;
			}
			i += 2;
			j += 2;
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