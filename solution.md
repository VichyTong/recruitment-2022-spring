# 七边形超算队招新解答

## 设备信息

CPU
- AMD Ryzen 7 5800H
- 内核数：8

内存
- 容量：16GB
- 速度：3200MHz

## Task1

1. 在虚拟机上配置了 `make` ，尝试使用 `makefile`

2. 在 `c++reference` 学习了 `std:thread()` 的基本用法

3. 题目要求使用8个线程来计算分型图，考虑到将第 $i$ 行的计算任务分配给第 $i \% 8$ 个线程。将 `mandelbrotThread.cpp` 中的 `MAX_THREADS` 修改为 8，并在每个线程中每 8 行计算一次，达到了 6.68x 的加速比。

## Task2

1. 阅读题目过后顺序阅读了 `CS149intrin.h` ，明白了 `mask` 和 `vec` 这两种数据类型的作用， `mask` 用作逻辑判断， `vec` 相当于向量寄存器。

2. `main.cpp` 中的 `abs` 函数给我了使用这些函数的样例，弄明白了如何操作向量寄存器。

3. 对着原始的程序一行行地转换到了`SIMD`写法。对于第一个任务，还需要处理 `VECTOR_WIDTH` 不整除于 `N` 的情况。对于多出的一小段，新开一个长度为 `VECTOR_WIDTH` 的向量寄存器，将 `maskAll` 后半部分恒置为 $0$，不进行运算。

## Task3

1. 首先查看了 `Block` 的文档，将矩阵分块进行运算，在测试下发现分块长度为16时较优。

2. 将循环中的 `k` 放在最外层循环，就能将 `a[i][k]` 保存下来，但是测试发现优化不大。

3. 找到了[一个比较有用的网站](https://github.com/flame/how-to-optimize-gemm)。

4. 对矩阵进行1*4的分块，在循环中 `i+=4` ，调用一个 `AddDot1x4` 函数。在函数中把重复使用的值存入寄存器，并采用指针访问优化，达到如下效果：
```
Running, dataset: size 256
time spent: 11955us
Passed, dataset: size 256

Running, dataset: size 512
time spent: 95484us
Passed, dataset: size 512

Running, dataset: size 1024
time spent: 973320us
Passed, dataset: size 1024

Running, dataset: size 2048
time spent: 1.2415e+07us
Passed, dataset: size 2048

```

5. 在 `AddDot1x4` 函数的 `for` 循环中使用间接寻址的方法做优化，优化程度可以忽略不计。

6. 采用 `4*4` 分块，达到如下效果：
```
Running, dataset: size 256
time spent: 8216us
Passed, dataset: size 256

Running, dataset: size 512
time spent: 61762us
Passed, dataset: size 512

Running, dataset: size 1024
time spent: 517034us
Passed, dataset: size 1024

Running, dataset: size 2048
time spent: 5.84973e+06us
Passed, dataset: size 2048
```
7. 采用`SIMD`，遇到非常多的bug。首先是编译不通过，在编译选项中加入了 `-msse4.1` 成功通过。第二个是 `_mm_set_epi32` 函数，它的赋值顺序应该倒转过来，按照 `b[k][3],b[k][2],b[k][1],b[k][0]` 的顺序赋值。但是程序的效率反而降低了，怀疑是`union`中的操作占用时间太多。

```
Running, dataset: size 256
time spent: 22864us
Passed, dataset: size 256

Running, dataset: size 512
time spent: 180093us
Passed, dataset: size 512

Running, dataset: size 1024
time spent: 1.47905e+06us
Passed, dataset: size 1024

Running, dataset: size 2048
time spent: 1.27792e+07us
Passed, dataset: size 2048
```

8. 将union取消使用，替换为 `_mm_load_si128` 函数来储存答案，优化效果并不突出。尝试将`_mm_set_epi32` 替换为 `_mm_load_si128`，降低了少量的时间消耗。经过测试，`_mm_set_epi32(ap, ap, ap, ap)` 的效率比 `_mm_set1_epi32(ap)` 要高。

9. 尝试使用了 `OpenMP` ，效率有极大提升。
```
Running, dataset: size 256
time spent: 8660us
Passed, dataset: size 256

Running, dataset: size 512
time spent: 36628us
Passed, dataset: size 512

Running, dataset: size 1024
time spent: 245792us
Passed, dataset: size 1024

Running, dataset: size 2048
time spent: 1.84662e+06us
Passed, dataset: size 2048
```

10. 没有想清楚为什么使用 `SIMD` 之后程序效率反而降低，将原来的非 `SIMD` 程序替换上来，使用 `OpenMP` ，达到以下效果：
```
Running, dataset: size 256
time spent: 2807us
Passed, dataset: size 256

Running, dataset: size 512
time spent: 15456us
Passed, dataset: size 512

Running, dataset: size 1024
time spent: 114352us
Passed, dataset: size 1024

Running, dataset: size 2048
time spent: 959647us
Passed, dataset: size 2048
```