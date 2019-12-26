#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <memory>
#include <chrono>
#include <getopt.h>

using namespace std;
using namespace std::chrono;

static struct option long_options[] = {
    {"sample", required_argument, NULL, 0},
    {"epochs", required_argument, NULL, 0}};

char *allocBuffer(size_t size);
void freeBuffer(char *buffer);
void tokenize(const char *buffer, size_t buffer_size, size_t *tokens, size_t tokens_size);

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

    if (epochs < 3)
        return -1;

    FILE *fp = fopen(sample_bin_file, "rb");
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char *buffer = allocBuffer(size);

    size_t read_size = 0;
    while (read_size < size)
    {
        read_size += fread(buffer + read_size, 1, size - read_size, fp);
    }
    fclose(fp);

    fp = fopen(sample_idx_file, "r");
    vector<size_t> actual_tokens;
    size_t token;
    while (fscanf(fp, "%zu", &token) == 1)
    {
        actual_tokens.push_back(token);
    }
    fclose(fp);

    printf("File size: %zu, tokens: %zu\n", size, actual_tokens.size());

    vector<float> elapseds;
    vector<size_t> tokens(actual_tokens.size());

    for (size_t i = 0; i < epochs; i++)
    {
        auto t1 = steady_clock::now();
        tokenize(buffer, size, &tokens[0], actual_tokens.size());
        auto t2 = steady_clock::now();

        auto timespan = 1.0f * duration_cast<microseconds>(t2 - t1).count() / 1000;
        elapseds.push_back(timespan);

        sort(tokens.begin(), tokens.end());
        if (!equal(actual_tokens.begin(), actual_tokens.end(), tokens.begin()))
        {
            printf("Epoch: %zu Fail\n", i);
        }
        else
        {
            printf("Epoch: %zu, took %.3f milliseconds\n", i, timespan);
        }
    }
    float totalElapsed = accumulate(elapseds.begin(), elapseds.end(), 0) - *max_element(elapseds.begin(), elapseds.end()) - *min_element(elapseds.begin(), elapseds.end());
    float averageElapsed = totalElapsed / (epochs - 2);
    printf("Took %.3f milliseconds totally, took %.3f milliseconds averagely\n", totalElapsed, averageElapsed);

    freeBuffer(buffer);
}
