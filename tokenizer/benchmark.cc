#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <memory>
#include <chrono>
#include <thread>
#include <getopt.h>
#include "benchmark.h"

using namespace std;
using namespace std::chrono;
using namespace std::chrono_literals;

static struct option long_options[] = {
    {"sample", required_argument, NULL, 0},
    {"epochs", required_argument, NULL, 0}};

int main(int argc, char *argv[])
{
    int option_index;
    char sample_bin_file[FILENAME_MAX] = {0};
    char sample_idx_file[FILENAME_MAX] = {0};
    size_t epochs = 0;

    while (getopt_long(argc, argv, "", long_options, &option_index) != -1)
    {
        switch (option_index)
        {
        case 0:
            sprintf(sample_bin_file, "%s.bin", optarg);
            sprintf(sample_idx_file, "%s.idx", optarg);
            break;
        case 1:
            epochs = strtoull(optarg, NULL, 10);
            break;
        }
    }

    FILE *fp = fopen(sample_bin_file, "rb");
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

    fp = fopen(sample_idx_file, "r");
    vector<const char *> actual_tokens;
    size_t pos;
    while(fscanf(fp, "%zu", &pos) == 1)
    {
        actual_tokens.push_back(begin + pos);
    }
    fclose(fp);

    printf("File size: %zu, tokens: %zu\n", size, actual_tokens.size());
    this_thread::sleep_for(2s);

    vector<double> elapseds(epochs);
    vector<const char *> tokens(actual_tokens.size());

    for (size_t i = 0; i < epochs; i++)
    {
        auto t1 = steady_clock::now();
        tokenize(begin, end, &tokens[0]);
        auto t2 = steady_clock::now();

        auto timespan = duration_cast<duration<double>>(t2 - t1);
        elapseds.push_back(timespan.count());

        sort(tokens.begin(), tokens.end());
        if (!equal(actual_tokens.begin(), actual_tokens.end(), tokens.begin()))
        {
            printf("Epoch: %zu Fail\n", i);
        }
        printf("Epoch: %zu, took %f seconds\n", i, timespan.count());
    }

    printf("Took %f seconds totally, took %f seconds averagely\n", accumulate(elapseds.begin(), elapseds.end(), 0.0), accumulate(elapseds.begin(), elapseds.end(), 0.0) / epochs);
}
