#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using namespace std;
using namespace std::chrono;

void *allocHostMemory(size_t size);
void freeHostMemory(void *ptr);
void test(const char *buffer, size_t buffer_size, size_t *breaks,
          size_t max_breaks_count, steady_clock::time_point &start,
          steady_clock::time_point &stop);

int main(int argc, char *argv[]) {
  const char *input_file_name = argv[1];
  size_t break_count = strtoull(argv[2], nullptr, 10);
  const char *log_file_name = argv[3];

  FILE *fp = fopen(input_file_name, "rb");
  fseek(fp, 0, SEEK_END);
  size_t buffer_size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  char *buffer = reinterpret_cast<char *>(allocHostMemory(buffer_size));
  size_t read_size = 0;
  while (read_size < buffer_size) {
    read_size += fread(buffer + read_size, 1, buffer_size - read_size, fp);
  }
  fclose(fp);

  vector<size_t> breaks(break_count);

  steady_clock::time_point t1, t2;
  test(buffer, buffer_size, breaks.data(), break_count, t1, t2);

  freeHostMemory(buffer);

  printf("Test function took %.3f milliseconds\n",
         1.0f * duration_cast<microseconds>(t2 - t1).count() / 1000);

  if (log_file_name) {
    fp = fopen(log_file_name, "w");
    for (size_t bre : breaks) {
      fprintf(fp, "%016zx\n", bre);
    }
    fclose(fp);
  }

  return 0;
}
