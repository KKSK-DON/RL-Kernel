# P3 / T06 Router Backward：第一版 CUDA 核心

本实验实现合同 `p3-router-task-contract.v22` 的 `dweights → ds` 数学主体，供
Hash / Learned 两条反向路径共用。它消费人工构造的原始张量，尚未接入正式
`SavedRouteSealedV1`、provider 或算子注册表。这里的测试结果只代表局部核心验证，
不能代替 T01 oracle、P3 WS1 Gate 或训练框架验收。

## 从哪里读

1. `csrc/cuda/moe/router_backward_core.cu`：先读 `route_backward_kernel`，这是要学习和修改的核心。
2. `examples/p3_router_backward/prototype.py`：构建扩展、检查实验输入，以及 CPU 诊断参考。
3. `tests/test_p3_router_backward_core.py`：对照、重复 expert、padding、线程配置与异常输入测试。
4. `examples/p3_router_backward/bindings.cpp`：把 CUDA 启动函数暴露给独立 Python 扩展。

扩展按需编译；本实验不改动主库的 `setup.py` 和运行时分发。

## 从 Softmax / RMSNorm 经验迁移过来

你熟悉的「一个 block 处理一行，shared memory 交换数据，再同步」可以直接使用。
这次一行对应一个 token；输出一行有 256 个 expert 梯度。

| 工作 | 谁负责 | 为什么这样安排 |
| --- | --- | --- |
| 读六个 `g`、`p`、expert ID | thread 0 | 数量很小，第一版先把运算顺序写清楚 |
| 算六路 `c` 和六个 `da` | thread 0 | 精确遵循合同指定的加法树 |
| 保存六个 `da` 和 ID | shared memory | 同一 block 的其他 warp 都要读 |
| 等待计算完成 | 全部线程执行 `__syncthreads()` | 读 shared memory 前建立同步 |
| 写 `ds[token, expert]` | thread `expert` | 每个输出只有一个写入者，连续线程写连续地址 |

默认每 block 256 个线程。测试也运行 128 个线程，此时每个线程负责两个 expert；
每个输出的计算顺序不变，结果应该逐位相同。

### 六路求和：显式写树

```text
gp[i] = g[i] * p[i]
c = ((gp[0] + gp[1]) + (gp[2] + gp[3])) + (gp[4] + gp[5])
da[i] = (1.5 / Z) * (g[i] - c)
```

这里 `p` 是前向保存的未乘 1.5 的比例，`Z` 是前向保存的归一化分母。
Python 实验参数 `z` 表示这个 **Z**，不是前面 gate GEMM 的 logits。
反向直接使用保存的 `p` 和 `Z`，不重新归一化，也不重新加 epsilon。

普通 warp reduction 的结合顺序未必是上面这棵树，所以这里不调用通用
`warp_reduce_sum`。`__fmul_rn` / `__fadd_rn` / `__fsub_rn` / `__fdiv_rn`
分别指定一次 FP32 舍入；乘法和加法分开，不让编译器合成 FMA。
独立扩展也显式关闭 FMA contraction 和 flush-to-zero。
这些构成当前实验的算术选择，仍需与 T01 正式 oracle 对接确认。

### 重复 expert：让输出线程按顺序收集

例如六个 ID 是 `[7, 7, 3, 255, 7, 3]`：

- thread 7 依次累加 `da[0]`、`da[1]`、`da[4]`。
- thread 3 依次累加 `da[2]`、`da[5]`。
- thread 255 写入 `da[3]`；其他线程写 `+0.0f`。

每个线程先把自己的累加器置零，然后从 slot 0 扫到 slot 5，最终写回一次。
这样无需浮点 `atomicAdd`，也不要求调用者预清零输出。
这段顺序累加与前面的六路加法树是两个不同步骤。

padding 分支在整个 block 内一致，直接写零并返回；它不会读取无效 ID、`p` 或 `Z`，
也不会出现一部分线程等待 barrier、另一部分线程提前返回的问题。

## 实验输入与错误检查

`route_backward_core(dweights, ids, p, z, row_active)` 返回 FP32 `[T,256]`。
输入必须连续，且位于同一 NVIDIA CUDA 设备：

| 输入 | dtype / shape |
| --- | --- |
| `dweights`、`p` | FP32 `[T,6]` |
| `ids` | INT32 `[T,6]` |
| `z` | FP32 `[T]` |
| `row_active` | BOOL `[T]` |

实验 wrapper 检查 active 行的有限值、ID 范围、正分母和非负比例；输出溢出时报错。
全 padding 或 `T=0` 时返回零张量，不编译或启动本扩展。
这些同步检查会增加开销，所以 wrapper 耗时不能当作 kernel 耗时。
底层 `_out` binding 仅供测试使用，调用它需要预先保证输入数值合法。

## 在 CUDA 环境中验证

需要现有隔离环境中的 PyTorch CUDA、Ninja 和匹配的 CUDA toolkit。
不用安装整个 RL-Kernel，也不用安装 pytest。先激活该隔离环境，让其 `bin` 目录
（包括 Ninja）进入 `PATH`，再在仓库根目录执行：

```bash
# 共享机器上先确认选定 GPU 空闲，再设置对应编号。
export CUDA_VISIBLE_DEVICES=0
export MAX_JOBS=2
export OMP_NUM_THREADS=1
python -c 'import torch; assert torch.version.cuda and torch.cuda.is_available()'
python -m unittest discover -s tests -p test_p3_router_backward_core.py -v
```

必要时设置 `CUDA_HOME` 指向自己的 toolkit，并把 `TORCH_EXTENSIONS_DIR`、`TMPDIR`
指向自己的构建与临时目录。首次测试会编译独立扩展。
没有 CUDA 时 CUDA 测试会跳过；这种结果不代表 GPU 验证通过。

测试把输出转成 INT32 比较完整位模式，因此也能区分正负零。
CPU FP32 参考逐步执行运算，用于诊断；FP64 autograd 从原始分数独立求导，检查数学。
另有能区分不同加法顺序的消去样例，防止误用通用 reduction。

### 已执行的验证：2026-10-02

在 H100 80GB、PyTorch 2.9.1+cu128、nvcc 12.8.93 上编译运行：

- 14 项测试全部通过，无跳过；包含不同大小的逐位对照、128/256 线程配置和非默认 stream。
- Compute Sanitizer 的 memcheck、racecheck、initcheck 均通过；racecheck 无 warning。
- Sanitizer 运行时关闭 PyTorch CUDA 内存缓存，让未初始化读取检查覆盖新分配的输出。

完整输出、工具版本和源文件 SHA-256 保存在
[`validation/h100-20261002.json`](validation/h100-20261002.json)。
源码变更后应重新验证；这里没有性能结论或官方 P3 Gate 结论。

## 下一步接入边界

T01 起步套件发布后，再把核心接到正式 `hash_route_bwd` / `learned_route_bwd`：

- 消费正式 sealed saved，验证身份、checksum、版本和权重来源。
- 接入指定的 device status、invocation echo、provider readback 协议。
- 使用官方 oracle、recorded fixtures 和 `check_p3` 做验收。

当前 raw-tensor 入口不承担这些职责，也没有伪造对应 schema 或 PASS 状态。
T02 的 `ds → dz`、gate GEMM 反向、专家计算及多卡通信不在这个核心内。
