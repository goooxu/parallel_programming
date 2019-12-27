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
static void tokenize(It begin, It end, OutputIt breaks) {
  for (It p = find(begin, end); p != end; p = find(p, end)) {
    *breaks++ = distance(begin, p);
  }
}

int main(int argc, char *argv[]) {
  const char *input_file_name = argv[1];
  const char *log_file_name = argv[2];

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

  vector<size_t> breaks;
  tokenize(begin, end, back_inserter(breaks));
  printf("File size: %zu, token count: %zu\n", size, breaks.size());

  if (log_file_name) {
    fp = fopen(log_file_name, "w");
    for (size_t bre : breaks) {
      fprintf(fp, "%016zx\n", bre);
    }
    fclose(fp);
  }

  return 0;
}