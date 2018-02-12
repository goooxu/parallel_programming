#include <cstdio>

__global__ void foo(const char *array, const char *base, const char **tokens, int *index)
{
    int sourceIdx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    if (array[sourceIdx] == '\r')
    {
        if (array[sourceIdx + 1] == '\n')
        {
            int targetIdx = atomicAdd(index, 1);
            tokens[targetIdx] = base + sourceIdx + 2;
        }
    }
    else if (array[sourceIdx] == '\n')
    {
        if (sourceIdx > 0 && array[sourceIdx - 1] == '\r')
        {
            int targetIdx = atomicAdd(index, 1);
            tokens[targetIdx] = base + sourceIdx + 1;
        }
    }
}

void tokenize(const char *begin, const char *end, const char **tokens, size_t max_tokens)
{
    const char *d_array;
    const char **d_tokens;
    int *d_index;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void **)&d_array, end - begin);
    cudaMalloc((void **)&d_tokens, max_tokens * sizeof(const char *));
    cudaMalloc((void **)&d_index, sizeof(int));
    cudaMemcpy((void *)d_array, begin, end - begin, cudaMemcpyHostToDevice);
    cudaMemset(d_index, 0, sizeof(int));

    int blocks = (end - begin) / 2 / 1024;

    cudaEventRecord(start);
    foo<<<blocks, 1024>>>(d_array, begin, d_tokens, d_index);
    cudaEventRecord(stop);

    cudaMemcpy(tokens, d_tokens, max_tokens * sizeof(const char *), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel took milliseconds: %.4fms\n", milliseconds);
}