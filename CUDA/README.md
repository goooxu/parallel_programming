本教程主要讲一些CUDA的优化

我们用一块GeForce GTX 1080 Ti来进行测试，下面是它的一些参数

|GeForce GTX 1080 Ti|Parameters|
---|---
|Computer Capability|6.1|
|Max Clock Rate|1582 MHz|
|Global Memory Size|10.9 Gbytes|
|Multiprocessors|28|
|Total CUDA Cores|3584|
|Threads / Warp|32|
|CUDA Cores / Multiprocessor|128|
|Registers / Multiprocessor|65536|
|Register File Capacity / Multiprocessor|256 Kbytes|
|Constant Memory Size / Multiprocessor|64 Kbytes|
|Shared Memory Size / Multiprocessor|64 Kbytes|
|Max Warps / Multiprocessor|64|
|Max Threads / Multiprocessor|2048|
|Max Thread Blocks / Multiprocessor|32|
|Max Shared Memory Size / Block|48 Kbytes|
|Max Registers / Block|65536|
|Max Threads / block|1024|
|Max Registers / Thread|255|
|Max dimension size of a thread block|1024,1024,64|
|Max dimension size of a grid size|2147483647, 65535, 65535|
|Concurrent copy and kernel execution|Yes with 2 copy engines|

>另外，使用的CUDA开发工具版本如下：CUDA Driver Version: 10.2, Runtime Version: 10.2

我们使用如下一段CUDA代码来做为测试标准
```C++
__global__ void sleep(int64_t waitCycles) {
  long long int start = clock64();
  for (;;) {
    auto total = clock64() - start;
    if (total >= waitCycles) {
      break;
    }
  }
}
```
按照上表中的运行频率1582 MHz来计算，可以得出大概是‭1,582,000,000‬个时钟周期等于1秒钟

### 实验1 ###
[不同blocks和threads对运行时间的影响](experiment01.md) 
