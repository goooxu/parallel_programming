#include <cstddef>
#include <omp.h>

static inline const char *find(const char *begin, const char *end)
{
	const char *p = begin;
	const char *q = p + 1;
	while (q < end)
	{
		if (*q == '\n')
		{
			if (*p == '\r')
			{
				return q + 1;
			}
			p += 2;
			q += 2;
		}
		else if (*q == '\r')
		{
			p += 1;
			q += 1;
		}
		else
		{
			p += 2;
			q += 2;
		}
	}
	return end;
}

void tokenize_one(const char *begin, const char *end, const char **&tokens)
{
	for (const char *p = find(begin, end); p != end; p = find(p, end))
#pragma omp critical
		*tokens++ = p;
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