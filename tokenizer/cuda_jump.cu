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
    __shared__ char s_array[BLOCK_SIZE * 2 + 1];

    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    size_t s_offset = threadIdx.x * 2;
    *reinterpret_cast<short *>(&s_array[s_offset]) = *reinterpret_cast<const short *>(&array[offset]);
    if (threadIdx.x == BLOCK_SIZE - 1)
        s_array[BLOCK_SIZE * 2] = array[offset + 2];
    __syncthreads();

    if (threadIdx.x == 0)
    {
        if (s_array[0] == '\r' && s_array[1] == '\n')
        {
            int index = atomicAdd(token_index, 1);
            tokens[index] = offset + 2;
        }
        if (s_array[BLOCK_SIZE * 2 - 1] == '\r' && s_array[BLOCK_SIZE * 2] == '\n')
        {
            int index = atomicAdd(token_index, 1);
            tokens[index] = offset + 2 * BLOCK_SIZE + 1;
        }
    }
    else
    {
        if (s_array[s_offset] == '\r')
        {
            if (s_array[s_offset + 1] == '\n')
            {
                int index = atomicAdd(token_index, 1);
                tokens[index] = offset + 2;
            }
        }
        else if (s_array[s_offset] == '\n')
        {
            if (s_array[s_offset - 1] == '\r')
            {
                int index = atomicAdd(token_index, 1);
                tokens[index] = offset + 1;
            }
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

    int blocks = buffer_size / 2 / BLOCK_SIZE;

    cudaEventRecord(start);
    foo<<<blocks, BLOCK_SIZE>>>(d_buffer, d_tokens, d_token_index);
    cudaEventRecord(stop);

    cudaMemcpy(tokens, d_tokens, token_size * sizeof(size_t), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel took %.3f milliseconds\n", milliseconds);

    cudaFree((void *)d_buffer);
    cudaFree((void *)d_tokens);
    cudaFree((void *)d_token_index);
}