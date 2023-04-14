# 笔记 ： Parallel-Computing Practice in C++, from parallel101

课件链接：[github.com/parallel101/course](https://github.com/parallel101/course)

benchmark在线测试网站：[https://quick-bench.com/](https://quick-bench.com/)

## 1. Cmake

配置、生成、运行：

```shell
> cmake -B build
> cmake --build build
> build/main
```

创建库：

```Cmake
add_library(test STATIC source1.cpp source2.cpp)  # 生成静态库 libtest.a
add_library(test SHARED source1.cpp source2.cpp)  # 生成动态库 libtest.so
target_link_libraries(myexec PUBLIC test)  # 为 myexec 链接刚刚制作的库 libtest.a
```

子模块的头文件如何处理：

```Cmake
target_include_directories(hellolib PUBLIC .)
```

## 2. RAII & 智能指针

三五法则：

1. 如果一个类定义了**解构函数**，那么必须**同时定义或删除拷贝构造函数和拷贝赋值函数**，否则出错。
2. 如果一个类定义了**拷贝构造函数**，那么必须**同时定义或删除拷贝赋值函数**，否则出错，删除可导致低效。
3. 如果一个类定义了**移动构造函数**，那么必须**同时定义或删除移动赋值函数**，否则出错，删除可导致低效。
4. 如果一个类定义了**拷贝构造函数或拷贝赋值函数**，那么必须最好**同时定义移动构造函数或移动赋值函数**，否则低效。

## 4. 从汇编角度看编译器优化

1. 函数尽量写在同一个文件内
2. 避免在 for 循环内调用外部函数
3. 非 const 指针加上 __restrict 修饰
4. 试着用 SOA 取代 AOS
5. 对齐到 16 或 64 字节
6. 简单的代码，不要复杂化
7. 试试看 #pragma omp simd
8. 循环中不变的常量挪到外面来
9. 对小循环体用 #pragma unroll
10. -ffast-math 和 -march=native

## 5. C++多线程编程

### 线程

CMakeLists.txt 里链接 Threads::Threads

```Cmake
ind_package(Threads REQUIRED)
target_link_libraries(cpptest PUBLIC Threads::Threads)
```

创建线程：

```cpp
std::thread t1([&] {
    download("hello.zip");
});
```

主线程等待子线程结束：`t1.join();`。解构函数不再销毁线程：`t1.detach()`。

### 异步

std::async 接受一个带返回值的 lambda，自身返回一个 std::future 对象。lambda 的函数体将在另一个线程里执行。
最后调用 future 的 get() 方法，如果此时 download 还没完成，会等待 download 完成，并获取 download 的返回值。

```cpp
std::future\<int> fret = std::async([&] {
    return download("hello.zip");
});
int ret = fret.get();
```

除了 get() 会等待线程执行完毕外，wait() 也可以等待他执行完，但是不会返回其值。wait_for()和wait_until() 同理。

### 互斥量

std::mutex——创建、上锁、解锁：

```cpp
std::mutex mtx;
mtx.lock();
mtx.unlock();
```

尝试上锁：

```cpp
if (mtx1.try_lock())
    printf("succeed\n");
else
    printf("failed\n");
//同理有try_lock_for()和try_lock_until()
```

std::lock_guard——符合 RAII 思想的 lock()：

```cpp
std::mutex mtx;
std::lock_guard grd(mtx);
```

std::unique_lock——自由度更高（可提前解锁）：

```cpp
std::mutex mtx;
std::unique_lock grd(mtx);

std::unique_lock grd(mtx, std::defer_lock);//不自动上锁

std::unique_lock grd(mtx, std::try_to_lock);//尝试上锁
        if (grd.owns_lock())//检查是否上锁成功
            printf("t1 success\n");

mtx.lock();
std::unique_lock grd(mtx, std::adopt_lock);//默认已上锁
```

### 死锁

解决：

1. 永远不要同时持有两个锁
2. 保证双方上锁顺序一致
3. 用 std::lock 同时对多个上锁（RAII 版本为std::scoped_lock）

同一个线程重复调用 lock() 也会造成死锁，解决：

1. 把函数里的 lock() 去掉，并在其文档中说明：本函数不是线程安全的
2. 改用 std::recursive_mutex，但是有性能损失

### 读写锁

创建、上写锁、解写锁、上读锁、解读锁：

```cpp
std::shared_mutex m_mtx;
m_mtx.lock();
m_mtx.unlock();
m_mtx.lock_shared();
m_mtx.unlock_shared();
```

std::shared_lock：符合 RAII 思想的 lock_shared()：

```cpp
std::shared_mutex m_mtx;
std::unique_lock grd(m_mtx);//写
std::shared_lock grd(m_mtx);//读
```

### 条件变量

必须和 `std::unique_lock<std::mutex>` 一起用：

```cpp
std::condition_variable cv;
std::mutex mtx;
std::thread t1([&] {
    std::unique_lock lck(mtx);
    cv.wait(lck);
    std::cout << "t1 is awake" << std::endl;
});
cv.notify_one();  // will awake t1
```

还可以额外指定一个参数，变成 cv.wait(lck, expr) 的形式，其中 expr 是个 lambda 表达式，只有其返回值为 true 时才会真正唤醒，否则继续等待。
`cv.wait(lck, [&] { return ready; });`

`cv.notify_one()` 只会唤醒其中一个等待中的线程，而 `cv.notify_all()` 会唤醒全部。

> 如果需要用其他类型的 mutex 锁，可以用 std::condition_variable_any

### 原子操作

```cpp
std::atomic\<int\> counter = 0;
counter += 1;
counter.store(0);//相当于=
counter.fetch_add(1);//相当于+=，返回的是旧值
std::cout << counter.load() << std::endl;//读取int值
int old = counter.exchange(3);//写入并返回旧值
bool equal = counter.compare_exchange_strong(old, 3);//如果不相等，则把原子变量的值写入 old。
//如果相等，则把 val 写入原子变量。
```

## 7. 访存优化

### 内存带宽

结论：要想利用全部CPU核心，避免mem-bound，需要函数里有足够的计算量。
当核心数量越多，CPU计算能力越强，相对之下来不及从内存读写数据，从而越容易mem-bound。

### 缓存与局域性

缓存和内存之间传输数据的最小单位是缓存行（64字节）。
设计数据结构时，应该把数据存储的尽可能紧凑，不要松散排列。最好每个缓存行里要么有数据，要么没数据，避免读取缓存行时浪费一部分空间没用。

> 数据结构的底层矢量化和缓存行预取

- AOS（Array of Struct）单个对象的属性紧挨着存
- SOA（Struct of Array）属性分离存储在多个数组
- MyClass 内部是 SOA，外部仍是一个 `vector<MyClass>` 的 AOS——这种内存布局称为 AOSOA。
- **如果几个属性几乎总是同时一起用的**，比如位置矢量pos的xyz分量，可能都是同时读取同时修改的，这时用**AOS**，减轻预取压力。
- **如果几个属性有时只用到其中几个，不一定同时写入**，这时候就用**SOA**比较好，省内存带宽。
- **AOSOA**：在高层保持AOS的统一索引，底层又享受SOA带来的矢量化和缓存行预取等好处

### 预取与直写

- 缓存行预取技术：由硬件自动识别程序的访存规律，决定要预取的地址。一般来说只有线性的地址访问规律（包括顺序、逆序；连续、跨步）能被识别出来，而**如果访存是随机的，那就没办法预测**。
- 为了解决随机访问的问题，把分块的大小调的更大一些，比如 4KB 那么大，即64个缓存行，而不是一个。每次随机出来的是块的位置。
- 预取不能跨越页边界，否则可能会触发不必要的 page fault。所以选取页的大小，因为本来就不能跨页顺序预取，所以被我们切断掉也无所谓。
- 手动预取：`_mm_prefetch(&a[next_r * 16], _MM_HINT_T0);`

绕过缓存，直接写入：`_mm_stream_si32((int *)&a[i], *(int *)&value);`

将一个4字节的写入操作，挂起到临时队列，等凑满64字节后，直接写入内存，从而完全避免读的带宽。只支持int做参数，要用float还得转换一下指针类型，bitcast一下参数。

仅当这些情况才应该用 stream 指令：

1. 该数组只有写入，之前完全没有读取过。
2. 之后没有再读取该数组的地方。

需要注意，stream 系列指令写入的地址，必须是连续的，中间不能有跨步，否则无法合并写入，会产生有中间数据读的带宽。

### 内存分配与分页

malloc 不会实际分配内存，第一次访问会缺页中断。
malloc(n * sizeof(int))、new int[n] 不会初始化数组为0，std::vector\<int>、new int[n]{} 会初始化数组为0。

分配是按页面（4KB）来管理的

### 循环分块、莫顿编码

分块优化**矩阵乘法**：

```cpp
for (int j = 0; j < n; j++) {
    for (int iBase = 0; iBase < n; iBase += 32) {
        for (int t = 0; t < n; t++) {
            for (int i = iBase; i < iBase + 32; i++) {
                a(i, j) += b(i, t) * c(t, j);
            }
        }
    }
}
```

分块优化**小内核卷积**：

```cpp
constexpr int blockSize = 4;
for (int jBase = 0; jBase < n; jBase += blockSize) {
    for (int iBase = 0; iBase < n; iBase += blockSize) {
        for (int l = 0; l < nkern; l++) {
            for (int k = 0; k < nkern; k++) {
                for (int j = jBase; j < jBase + blockSize; j++) {
                    for (int i = iBase; i < iBase + blockSize; i++) {
                        a(i, j) += b(i + k, j + l) * c(i, j);
                    }
                }
            }
        }
    }
}
```

### 多核下的缓存

伪共享：如果多个核心同时访问的地址非常接近，这时候会变得很慢。
解决：把每个核心写入的地址尽可能分散开来。

## 8. CUDA编程

`__host__`:从CPU上调用，在CPU上执行。
`__global__`:从CPU上调用，在GPU上执行。
`__device__`:从GPU上调用，在GPU上执行。
（允许`__host__ __device__` 双重修饰）

`__CUDA_ARCH__`：一个宏定义，表示版本号。

### 线程、板块、网格

- 线程(thread)：并行的最小单位
- 板块(block)：包含若干个线程
- 网格(grid)：指整个任务，包含若干个板块

当前线程在板块中的编号：**threadIdx**
当前板块中的线程数量：**blockDim**
当前板块的编号：**blockIdx**
总的板块数量：**gridDim**

从属关系：线程∈板块∈网格
调用语法：`<<<gridDim, blockDim>>>`

### 内存管理

checkCudaErrors()自动检查错误代码并打印在终端

- 主机内存(host)：malloc、free
- 设备内存(device)：cudaMalloc、cudaFree
- 统一内存(managed)：cudaMallocManaged、cudaFree

### 网格跨步循环

无论指定每个板块多少线程（`blockDim`），总共多少板块（`gridDim`），都能自动根据给定的 n 区间循环，不会越界，也不会漏掉几个元素。

```cu
__global__ void kernel(int *arr, int n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        arr[i] = i;
    }
}
```

### thrust库

|容器|分配在|
|---|---|
|`universal_vector`|统一内存|
|`device_vector`|GPU|
|`host_vector`|CPU|

通过 `=` 运算符在 `device_vector` 和 `host_vector` 之间拷贝数据会自动调用 `cudaMemcpy`

thrust 模板函数会**根据容器类型，自动决定在 CPU 还是 GPU 执行**，如`thrust::generate`、`thrust::for_each`……

### CUDA原子操作

`atomicAdd(dst, src)` 等同于 `*dst += src`，但前者会返回旧值。

`atomicCAS(dst, cmp, src)`：原子地判断`*dst`和`cmp`是否相等，相等则将`src`写入`*dst`，并返回旧值。利用`atomicCAS`可以实现任意原子操作。

原子操作带来的问题是影响性能，解决方法之一是**线程局部变量**。如：

```cu
__global__ void parallel_sum(int *sum, int const *arr, int n) {
    int local_sum = 0;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        local_sum = std::max(local_sum, arr[i]);
    }
    atomicMax(&sum[0], local_sum);
}
```

这样每个线程就只有一次原子操作。

### 板块与共享内存

为什么要有板块（`Block`）？

- GPU 是由多个流式多处理器（SM）组成的。每个 SM 可以处理一个或多个板块。
- SM 又由多个流式单处理器（SP）组成。每个 SP 可以处理一个或多个线程。
- 每个 SM 都有自己的一块共享内存（shared memory），他的性质类似于 CPU 中的缓存——和主存相比很小，但是很快，用于缓冲临时数据。
- 板块数量大于 SM 的数量，这时NVIDIA驱动就会在多个 SM 之间调度各个板块。不过 GPU 不会做时间片轮换，而是板块一旦被调度到了一个 SM 上，就会一直执行，直到执行完退出。
- 一个 SM 可同时运行多个板块，这时多个板块共用同一块共享内存（每块分到的就少了）。而板块内部的每个线程则是被进一步调度到 SM 上的每个 SP。

共享内存变量用`__shared__`修饰符声明。

`__syncthreads()`：强制同步当前板块内的所有线程。

**线程组**（warp）：SM 对线程的调度是按照 32 个线程为一组来调度的，不可能出现一个组里只有单独一个线程被调走。

**线程组分歧**（warp divergence）：线程组中 32 个线程实际是绑在一起执行的，如果出现分支（if）语句时，如果 32 个 cond 中有的为真有的为假，则会导致两个分支都被执行，因此建议 GPU 上的 if 尽可能 32 个线程都处于同一个分支，要么全部真要么全部假，否则实际消耗了两倍时间。

==索引为(0,0)(1,0)(2,0)(3,0)...的thread连续，而不是(0,0)(0,1)(0,2)(0,3)...的连续==

### 共享内存进阶

==如果板块中线程数量*过多*：寄存器打翻（register spill）==

- 板块内的所有的线程共用一个寄存器仓库。当板块中的线程数量（blockDim）过多时，如果程序恰好用到了非常多的寄存器，就无法全部装在寄存器仓库里，而是要把一部分“打翻”到一级缓存中，造成性能上的损失。

==如果板块中的线程数量过少：延迟隐藏（latency hiding）失效==

- 当线程组陷入内存等待时，可以切换到另一个线程组，继续计算，这样一个 warp 的内存延迟就被另一个 warp 的计算延迟给隐藏起来了。因此，如果线程数量太少的话，就无法通过在多个 warp 之间调度来隐藏内存等待的延迟，从而低效。
- 此外，最好让板块中的线程数量（blockDim）为32的整数倍，否则假如是 33 个线程的话，那还是需要启动两个 warp，其中第二个 warp 只有一个线程是有效的，造成浪费。

**结论**：对于使用寄存器较少、访存为主的核函数（例如矢量加法），使用大 blockDim 为宜。反之（例如光线追踪）使用小 blockDim，但也不宜太小。

### 矩阵转置

- 使用二维的 blockDim 和 gridDim，然后在核函数里分别计算 x 和 y 的扁平化线程编号：

```cuda
template <class T>
__global__ void parallel_transpose(T *out, T const *in, int nx, int ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    out[y * nx + x] = in[x * ny + y];
}
/*main()*/parallel_transpose<<<dim3(nx / 32, ny / 32, 1), dim3(32, 32, 1)>>>(out.data(), in.data(), nx, ny);
```

- 利用共享内存，通过把 `in` 分块，按块跨步地读，而块内部则仍是连续地读——从低效全局的内存读到高效的共享内存中，然后在共享内存中跨步地读，连续地写到 `out` 指向的低效的全局内存中。

```cuda
template <int blockSize, class T>
__global__ void parallel_transpose(T *out, T const *in, int nx, int ny) {
    int x = blockIdx.x * blockSize + threadIdx.x;
    int y = blockIdx.y * blockSize + threadIdx.y;
    if (x >= nx || y >= ny) return;
    __shared__ T tmp[blockSize * blockSize];
    int rx = blockIdx.y * blockSize + threadIdx.x;
    int ry = blockIdx.x * blockSize + threadIdx.y;
    tmp[threadIdx.y * blockSize + threadIdx.x] = in[ry * nx + rx];
    __syncthreads();
    out[y * nx + x] = tmp[threadIdx.x * blockSize + threadIdx.y];
}
/*main()*/parallel_transpose<32><<<dim3(nx / 32, ny / 32, 1), dim3(32, 32, 1)>>>(out.data(), in.data(), nx, ny);
```

### 什么是**区块**（bank）

- GPU 的共享内存，实际上是 32 块内存条（bank）通过并联组成的,每个 bank 都可以独立地访问，他们每个时钟周期都可以读取一个 int。
- 然后，他们把地址空间分为 32 分，第 i 根内存条，负责 addr % 32 == i 的那几个 int 的存储。这样交错存储，可以保证随机访问时，访存能够尽量分摊到 32 个区块。
- 比如：`__shared__ int arr[1024];` 那么 arr[0] 是存储在 bank 0，arr[1] 是 bank 1……arr[32] 又是 bank 0，arr[33] 又是 bank 1。

**区块冲突**（bank conflict）

- 两个线程同时访问了 arr[0] 和 arr[32] 就会出现 bank conflict 导致必须排队,影响性能.
- 比如上面的`out[y * nx + x] = tmp[threadIdx.x * blockSize + threadIdx.y];`，线程(1,0)(2,0)(3,0)都会访问bank 0，造成冲突。
- 解决：把 tmp 这个二维数组从 32x32 变成 **33x32**，如下：

```cuda
__shared__ T tmp[(blockSize + 1) * blockSize];
out[y * nx + x] = tmp[threadIdx.x * (blockSize + 1) + threadIdx.y];
```

### GPU 优化方法总结

- 线程组分歧（wrap divergence）：尽量保证 32 个线程都进同样的分支，否则两个分支都会执行。
- 延迟隐藏（latency hiding）：需要有足够的 blockDim 供 SM 在陷入内存等待时调度到其他线程组。
- 寄存器打翻（register spill）：如果核函数用到很多局部变量（寄存器），则 blockDim 不宜太大。
- 共享内存（shared memory）：全局内存比较低效，如果需要多次使用，可以先读到共享内存。
- 跨步访问（coalesced acccess）：建议先顺序读到共享内存，让高带宽的共享内存来承受跨步。
- 区块冲突（bank conflict）：同一个 warp 中多个线程访问共享内存中模 32 相等的地址会比较低效，可以把数组故意搞成不对齐的 33 跨步来避免。

## 10. 稀疏数据结构与量化数据类型

TODO
