#include "benchmark.h"

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
			else
			{
				p += 2;
				q += 2;
			}
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

void tokenize(const char *begin, const char *end, const char **tokens)
{
	for (const char *p = find(begin, end); p != end; p = find(p, end))
		*tokens++ = p;
}