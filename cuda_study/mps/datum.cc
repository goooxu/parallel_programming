#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

using namespace std;

template <typename It> static It find(It begin, It end) {
  It i = begin;
  while (i != end - 1) {
    if (*i == '\r' && *(i + 1) == '\n')
      return i + 2;
    ++i;
  }
  return end;
}

template <typename It, typename OutputIt>
static void tokenize(It begin, It end, OutputIt separators) {
  for (It p = find(begin, end); p != end; p = find(p, end)) {
    *separators++ = distance(begin, p);
  }
}

int main(int argc, char *argv[]) {
  const char *input_file_name = argv[1];

  FILE *fp = fopen(input_file_name, "rb");
  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  unique_ptr<char[]> buf(reinterpret_cast<char *>(aligned_alloc(16, size)));

  size_t read_size = 0;
  while (read_size < size) {
    read_size += fread(buf.get() + read_size, 1, size - read_size, fp);
  }
  fclose(fp);

  const char *begin = buf.get();
  const char *end = begin + size;

  vector<size_t> separators;
  tokenize(begin, end, back_inserter(separators));
  printf("File size: %zu, separator count: %zu\n", size, separators.size());

  return 0;
}