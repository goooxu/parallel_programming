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

  if (buffer_size > 1 && buffer[0] == '\r' && buffer[1] == '\n') {
    *breaks++ = 2;
  }

  for (size_t i = 2; i < buffer_size - 1; i += 2) {
    if (buffer[i] == '\r') {
      if (buffer[i + 1] == '\n') {
        *breaks++ = i + 2;
      }
    } else if (buffer[i] == '\n') {
      if (buffer[i - 1] == '\r') {
        *breaks++ = i + 1;
      }
    }
  }

  if (buffer_size > 1 && buffer_size % 2 != 0) {
    if (buffer[buffer_size - 2] == '\r' && buffer[buffer_size - 1] == '\n') {
      *breaks++ = buffer_size;
    }
  }

  stop = steady_clock::now();
}