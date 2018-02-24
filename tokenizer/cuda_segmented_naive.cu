#include <cstdio>

static const size_t BLOCK_SIZE = 128;
static const size_t SEGMENT_SIZE = sizeof(int);

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
    __shared__ char s_array[SEGMENT_SIZE * BLOCK_SIZE + 1];

    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x) * SEGMENT_SIZE;
    size_t s_offset = threadIdx.x * SEGMENT_SIZE;
    *reinterpret_cast<int *>(&s_array[s_offset]) = *reinterpret_cast<const int *>(&array[offset]);
    if (threadIdx.x == BLOCK_SIZE - 1)
        s_array[s_offset + SEGMENT_SIZE] = array[offset + SEGMENT_SIZE];
    __syncthreads();

    for (size_t i = 0; i < SEGMENT_SIZE; i++)
    {
        if (s_array[s_offset + i] == '\r' && s_array[s_offset + i + 1] == '\n')
        {
            int index = atomicAdd(token_index, 1);
            tokens[index] = offset + i + 2;
        }
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

    size_t blocks = buffer_size / BLOCK_SIZE / SEGMENT_SIZE;

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