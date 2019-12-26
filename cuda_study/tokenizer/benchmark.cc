#include <chrono>
#include <cstdio>
#include <cstring>
#include <getopt.h>
#include <memory>
#include <vector>

using namespace std;
using namespace std::chrono;

static struct option long_options[] = {
    {"input", required_argument, nullptr, 0},
    {"breaks", required_argument, nullptr, 0},
    {"log", optional_argument, nullptr, 0}};

void *allocHostMemory(size_t size);
void freeHostMemory(void *ptr);
void test(const char *buffer, size_t buffer_size, size_t *breaks,
          size_t max_breaks_count, steady_clock::time_point &start,
          steady_clock::time_point &stop);

int main(int argc, char *argv[]) {
  int option_index;
  char input_file_name[FILENAME_MAX]{0};
  char log_file_name[FILENAME_MAX]{0};
  size_t break_count = 0;

  while (getopt_long(argc, argv, "", long_options, &option_index) != -1) {
    switch (option_index) {
    case 0:
      strcpy(input_file_name, optarg);
      break;
    case 1:
      break_count = strtoull(optarg, nullptr, 10);
      break;
    case 2:
      strcpy(log_file_name, optarg);
      break;
    }
  }

  FILE *fp = fopen(input_file_name, "rb");
  fseek(fp, 0, SEEK_END);
  size_t buffer_size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  char *buffer = reinterpret_cast<char *>(allocHostMemory(buffer_size * 2));
  size_t read_size = 0;
  while (read_size < buffer_size) {
    read_size += fread(buffer + read_size, 1, buffer_size - read_size, fp);
  }
  fclose(fp);

  vector<size_t> breaks(break_count);

  steady_clock::time_point t1, t2;
  test(buffer, buffer_size * 2, breaks.data(), break_count, t1, t2);

  freeHostMemory(buffer);

  printf("Test function took %.3f milliseconds\n",
         1.0f * duration_cast<microseconds>(t2 - t1).count() / 1000);

  if (log_file_name[0] != '\0') {
    FILE *fp = fopen(log_file_name, "w");
    for (size_t i = 0; i < break_count; i++) {
      fprintf(fp, "%016zx\n", breaks[i]);
    }
    fclose(fp);
  }

  return 0;
}
