# 笔记 ： Parallel-Computing Practice in C++, from parallel101

课件链接：[github.com/parallel101/course](https://github.com/parallel101/course)

benchmark在线测试网站：[https://quick-bench.com/](https://quick-bench.com/)

## 1. Cmake

配置、生成、运行：

```shell
> cmake -B build
> cmake --build build
> build/a.out
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
3. 用 std::lock 同时对多个上锁（RAII版本为std::scoped_lock）

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

必须和 std::unique_lock<std::mutex> 一起用：

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
for (auto _ : bm) {
    for (int j = 0; j < n; j++) {
        for (int iBase = 0; iBase < n; iBase += 32) {
            for (int t = 0; t < n; t++) {
                for (int i = iBase; i < iBase + 32; i++) {
                    a(i, j) += b(i, t) * c(t, j);
                }
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

## 10. 稀疏数据结构与量化数据类型
