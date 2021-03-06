#include <cstdio>

static const size_t BLOCK_SIZE = 128;

char *allocBuffer(size_t size)
{
    char *buffer;
    cudaHostAlloc(&buffer, size, cudaHostAllocMapped);
    return buffer;
}

void freeBuffer(char *buffer)
{
    cudaFreeHost(buffer);
}

__global__ void foo(const char *array, size_t *tokens, int *token_index)
{
    __shared__ char s_array[BLOCK_SIZE + 1];

    size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == BLOCK_SIZE - 1)
    {
        s_array[threadIdx.x] = array[offset];
        s_array[threadIdx.x + 1] = array[offset + 1];
    }
    else
    {
        s_array[threadIdx.x] = array[offset];
    }
    __syncthreads();

    if (s_array[threadIdx.x] == '\r' && s_array[threadIdx.x + 1] == '\n')
    {
        int index = atomicAdd(token_index, 1);
        tokens[index] = offset + 2;
    }
}

void tokenize(const char *buffer, size_t buffer_size, size_t *tokens, size_t token_size)
{
    const char *d_buffer;
    size_t *d_tokens;
    int *d_token_index;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void **)&d_buffer, buffer_size + 1);
    cudaMalloc((void **)&d_tokens, token_size * sizeof(size_t));
    cudaMalloc((void **)&d_token_index, sizeof(int));
    cudaMemcpy((void *)d_buffer, buffer, buffer_size, cudaMemcpyHostToDevice);
    cudaMemset(d_token_index, 0, sizeof(int));

    size_t blocks = buffer_size / BLOCK_SIZE;

    cudaEventRecord(start);
    foo<<<blocks, BLOCK_SIZE>>>(d_buffer, d_tokens, d_token_index);
    cudaEventRecord(stop);

    cudaMemcpy(tokens, d_tokens, token_size * sizeof(size_t), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel took %.4f milliseconds\n", milliseconds);

    cudaFree((void *)d_buffer);
    cudaFree((void *)d_tokens);
    cudaFree((void *)d_token_index);
}