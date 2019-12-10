我们来看一段简单的代码来运行这个kernel并统计时间
```C++
clock_t t1 = clock();

sleep<<<blocks, threads>>>(SECOND);
cudaDeviceSynchronize();

clock_t t2 = clock();

float duration = 1000.0f * (t2 - t1) / CLOCKS_PER_SEC;
printf("Time elapsed %.0f ms\n", duration);
```

通过改变`blocks`和`threads`的值，我们得到一些有趣的结果

|blocks|threads|Kernel run time|
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

下面来解释为什么会提到这样的结果

首先，引出两条规则

> 规则一：**一个block只会在一个Multiprocessor中运行**

> 规则二：**threads只以wrap为最小单位运行的**


1. 对于*1 block, 1024 threads*, 只会在1个Multiprocessor中运行, 每个thread的运行时间大约是1秒, 总计1024个threads,少于单个Multiprocessor的限制 (2048个threads), 所以运行时间为1倍
2. 对于*56 blocks, 1024 threads*, 每个Multiprocessor分配到的blocks数量为56/28=2, 2个blocks总计2048个threads, 没有超出Multiprocessor的限制, 所以运行时间也为1倍
3. 对于*57 blocks, 1024 threads*, 至少有1个Multiprocessor会分配到3个blocks, 总计3072个threads, 超出了Multiprocessor的限制, 需要至少分两次运行, 所以运行时间为2倍
4. 对于*57 blocks, 673 threads*, 至少有1个Multiprocessor会分配到3个blocks, 总计2019个threads, 虽然threads数目没有超出Multiprocessor的限制, 但是673个threads至少为22个warps, 3个blocks总计66个warps, 超出了Multiprocessor的限制, 需要至少分两次运行, 所以运行时间为2倍
5. 对于*57 blocks, 672 threads*, 至少有1个Multiprocessor会分配到3个blocks, 总计2016个threads和63个warps, 都没有超出Multiprocessor的限制, 所以运行时间为1倍
6. 对于*84 blocks, 672 threads*, 每个Multiprocessor分配到3个blocks, 总计2016个threads和64个warps, 没有超出Multiprocessor的限制, 所以运行时间为1倍
7. 对于*85 blocks, 672 threads*, 至少有1个Multiprocessor会分配到4个blocks, 总计2688个threads, 超出了Multiprocessor的限制, 需要至少分两次运行, 所以运行时间为2倍
8. 对于*85 blocks, 513 threads*, 至少有1个Multiprocessor会分配到4个blocks, 总计2052个threads, 超出了Multiprocessor的限制, 需要至少分两次运行, 所以运行时间为2倍
9. 对于*85 blocks, 512 threads*, 至少有1个Multiprocessor会分配到4个blocks, 总计2048个threads和64个warps, 都没有超出Multiprocessor的限制, 所以运行时间为1倍
10. 对于*112 blocks, 512 threads*, 每个Multiprocessor分配到4个blocks, 总计2048个threads和64个warps, 都没有超出Multiprocessor的限制, 所以运行时间为1倍
11. 对于*113 blocks, 512 threads*, 至少有1个Multiprocessor分配到5个blocks, 总计2560个threads, 超出了Multiprocessor的限制, 需要至少分两次运行, 所以运行时间为2倍
