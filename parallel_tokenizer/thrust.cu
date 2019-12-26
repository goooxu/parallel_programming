#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/copy.h>

static thrust::host_vector<char> h_input;

char *allocBuffer(size_t size)
{
    h_input.resize(size);
    return h_input.data();
}

void freeBuffer(char *buffer) {}

struct transform
{
    template <typename Tuple>
    __host__ __device__ size_t operator()(Tuple t)
    {
        if (thrust::get<1>(t) == '\r' && thrust::get<2>(t) == '\n')
        {
            return thrust::get<0>(t) + 2;
        }
        else
        {
            return (size_t)-1;
        }
    }
};

struct predicate
{
    __host__ __device__ bool operator()(size_t token)
    {
        return token != (size_t)-1;
    }
};

void tokenize(const char *buffer, size_t buffer_size, size_t *tokens, size_t token_size)
{
    thrust::device_vector<size_t> d_tokens(token_size);
    thrust::device_vector<char> d_input = h_input;

    auto zip_it = thrust::make_zip_iterator(
        thrust::make_tuple(
            thrust::counting_iterator<size_t>(0),
            d_input.begin(),
            d_input.begin() + 1));
    auto transform_it = thrust::make_transform_iterator(zip_it, transform());

    thrust::copy_if(transform_it, transform_it + buffer_size, d_tokens.begin(), predicate());
    thrust::copy(d_tokens.begin(), d_tokens.end(), tokens);
}