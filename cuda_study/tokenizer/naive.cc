#include <chrono>
#include <cstddef>
#include <cstdlib>

using namespace std::chrono;

void *allocHostMemory(size_t size) { return malloc(size); }

void freeHostMemory(void *ptr) { free(ptr); }

void test(const char *buffer, size_t buffer_size, size_t *breaks,
          size_t max_breaks_count, steady_clock::time_point &start,
          steady_clock::time_point &stop) {
  start = steady_clock::now();

  for (size_t i = 0; i < buffer_size - 1; i++) {
    if (buffer[i] == '\r' && buffer[i + 1] == '\n') {
      *breaks++ = i + 2;
    }
  }

  stop = steady_clock::now();
}