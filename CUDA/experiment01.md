我们使用前面提到的计时程序

来看看使用不同的`blocks`和`threads`数量对程序运行时间的影响

有如下的在device上运行的代码
```C++
const long long int SECOND_CLOCK_CYCLES = 1582 * 1000 * 1000;

__global__ void sleep(int *minCycles, int *maxCycles) {

  int cycles = 0;

  long long int start = clock64();
  for (;;) {
    cycles++;

    long long int duration = clock64() - start;
    if (duration >= SECOND_CLOCK_CYCLES) {
      break;
    }
  }

  if (blockIdx.x == 0) {
    atomicMin(minCycles, cycles);
    atomicMax(maxCycles, cycles);
  }
}
```

通过改变`blocks`和`threads`的数量，我们得到以下结果

|blocks|threads|Code run time|
---|---|---
|1|1024|~1s|
|56|1024|~1s|
|57|1024|~2s|
|57|673|~2s|
|57|672|~1s|
|84|672|~1s|
|85|672|~2s|
|85|513|~2s|
|85|512|~1s|
|112|512|~1s|
|113|512|~2s|

下面来解释为什么会出现这样的结果

首先，引出两条规则

> 规则一：**一个block只会在一个Multiprocessor中运行**

> 规则二：**threads只以wrap为最小单位运行的**

情况|解释
---|---
*1 block, 1024 threads*|只会在1个Multiprocessor中运行, 每个thread的运行时间大约是1秒, 总计1024个threads,少于单个Multiprocessor最多2048个threads的限制, 所以运行时间为1倍
*56 blocks, 1024 threads*|每个Multiprocessor分配到的blocks数量为56/28=2, 2个blocks总计2048个threads, 没有超出Multiprocessor的限制, 所以运行时间也为1倍
*57 blocks, 1024 threads*|至少有1个Multiprocessor会分配到3个blocks, 总计3072个threads, 超出了Multiprocessor的限制, 需要至少分2次运行, 所以运行时间为2倍
*57 blocks, 673 threads*|至少有1个Multiprocessor会分配到3个blocks, 总计2019个threads, 虽然threads数目没有超出Multiprocessor的限制, 但是673个threads至少为22个warps, 3个blocks总计66个warps, 超出了Multiprocessor的限制, 需要至少分2次运行, 所以运行时间为2倍
*85 blocks, 672 threads*|至少有1个Multiprocessor会分配到4个blocks, 总计2688个threads, 超出了Multiprocessor的限制, 需要至少分2次运行, 所以运行时间为2倍
*85 blocks, 513 threads*|至少有1个Multiprocessor会分配到4个blocks, 总计2052个threads, 超出了Multiprocessor的限制, 需要至少分2次运行, 所以运行时间为2倍
*113 blocks, 512 threads*|至少有1个Multiprocessor会分配到5个blocks, 总计2560个threads, 超出了Multiprocessor的限制, 需要至少分2次运行, 所以运行时间为2倍
