#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

using namespace std;

template <typename It>
static It find(It begin, It end)
{
	It p = begin;
	while (p != end - 1)
	{
		if (*p == '\r' && *(p + 1) == '\n')
			return p + 2;
		++p;
	}
	return end;
}

template <typename It, typename OutputIt>
static void tokenize(It begin, It end, OutputIt breaks)
{
	for (const char *p = find(begin, end); p != end; p = find(p, end))
	{
		*breaks++ = p;
	}
}

int main(int argc, char *argv[])
{
	FILE *fp = fopen(argv[1], "rb");
	fseek(fp, 0, SEEK_END);
	size_t size = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	auto deleter = [](void *p) { free(p); };
	unique_ptr<char[], decltype(deleter)> buf(reinterpret_cast<char *>(aligned_alloc(16, size)), deleter);

	size_t read_size = 0;
	while (read_size < size)
	{
		read_size += fread(buf.get() + read_size, 1, size - read_size, fp);
	}
	fclose(fp);

	const char *begin = buf.get();
	const char *end = begin + size;

	vector<const char *> actual_tokens;
	tokenize(begin, end, back_inserter(actual_tokens));
	printf("File size: %zu, tokens: %zu\n", size, actual_tokens.size());

	fp = fopen(argv[2], "w");
	for (const char *token : actual_tokens)
	{
		fprintf(fp, "%zu\n", token - begin);
	}
	fclose(fp);
}
