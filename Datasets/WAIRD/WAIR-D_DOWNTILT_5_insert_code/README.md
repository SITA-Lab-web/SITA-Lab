# WAIRD_DOWNTILT_5：单场景、5 基站、5 Agent 俯角共识优化

## 1. 实验含义

一个 WAIR-D 环境（例如 `00001`）被视为一个独立的多小区无线网络实例。

- 5 个 BS 对应 5 个 MPI Agent；
- 每个 Agent 都维护一份完整的联合俯角向量：

```text
theta = [theta_0, theta_1, theta_2, theta_3, theta_4]
```

- Agent 之间共识的是这份联合配置向量的本地副本；
- 不要求 `theta_0 = ... = theta_4`，5 个 BS 最终可以使用不同俯角；
- 第 i 个 Agent 只使用其 BS 关联用户的数据计算本地黑箱目标；
- 一个场景独立运行一次。多个场景是多个测试实例，不在同一次物理协作中混合。

局部目标：

```text
f_i(theta) = - mean_user_SE_i(theta)
             + outage_weight * mean(max(0, rate_floor - user_SE)^2)
```

全局测试值为五个局部目标的平均值。优化程序是最小化，因此 fitness 越小越好；通信性能还应报告 AvgSE、P10-SE、P5-SE 和 BS-balance。

## 2. 包结构

```text
Benchmarks/
  WairdDowntilt5.h
  Benchmarks.h
  Benchmarks.cpp
  default_config.json
  default_config_snippet.json
  data/W_WAIRD_DOWNTILT_5
tools/
  waird_preprocess_per_environment.py
patches/
  framework.cpp
  main.cpp
  LAC-MAS.cpp
run_one_environment.sh
run_environment_batch.sh
```

`patches/` 中是建议替换的改进版，不是 Python 数据处理代码。

---

# A. 本地电脑：只做一次离线数据处理

## A1. 原始数据放置

把 `WAIRD` 文件夹放在本包根目录：

```text
WAIRD_DOWNTILT_5_insert_code/
  WAIRD/
    00001/
      Info.npy
      Path.npy
      H_2_6_G.npy
      H_6_0_G.npy
      H_28_0_G.npy
      H_60_0_G.npy
      H_100_0_G.npy
    00002/
    ...
    00300/
  tools/
  Benchmarks/
```

## A2. 安装本地 Python 依赖

```bash
pip install numpy
```

## A3. 在 PyCharm / VSCode Terminal 预处理

建议主实验先使用 28 GHz：

```bash
python tools/waird_preprocess_per_environment.py \
  --root WAIRD \
  --out Benchmarks/data/WAIRD_DOWNTILT_5 \
  --carrier-files H_28_0_G.npy \
  --num-envs 300 \
  --subcarriers 64 \
  --bandwidth-mhz 46.08 \
  --delay-unit ns \
  --association strongest
```

Windows PowerShell：

```powershell
python tools/waird_preprocess_per_environment.py `
  --root "WAIRD" `
  --out "Benchmarks/data/WAIRD_DOWNTILT_5" `
  --carrier-files "H_28_0_G.npy" `
  --num-envs 300 `
  --subcarriers 64 `
  --bandwidth-mhz 46.08 `
  --delay-unit ns `
  --association strongest
```

生成结果：

```text
Benchmarks/data/WAIRD_DOWNTILT_5/
  00001/
    bs_00.txt
    bs_01.txt
    bs_02.txt
    bs_03.txt
    bs_04.txt
    meta.txt
  ...
  00300/
```

每个 `bs_XX.txt` 只包含对应 BS 服务用户的本地评价数据。优化运行阶段不再读取 `.npy`。

### 关于 H 与 Path 路径数不一致

正常情况下，同一环境目录内的 `H_28_0_G.npy` 与 `Path.npy` 应匹配。脚本默认发现不一致就报错，以防误把不同环境的文件混在一起。

仅调试时可加：

```text
--mismatch-policy min
```

正式实验不建议使用该选项，除非已确认数据定义允许截断。

## A4. 上传到集群

只需要上传：

```text
Benchmarks/data/WAIRD_DOWNTILT_5/
```

以及下面的 C++ 文件。集群不需要 Python、NumPy 或原始 `.npy` 数据。

---

# B. 插入 LAC-MAS 工程

在 LAC-MAS 根目录执行：

```bash
cp WAIRD_DOWNTILT_5_insert_code/Benchmarks/WairdDowntilt5.h Benchmarks/
cp WAIRD_DOWNTILT_5_insert_code/Benchmarks/Benchmarks.h Benchmarks/
cp WAIRD_DOWNTILT_5_insert_code/Benchmarks/Benchmarks.cpp Benchmarks/
cp WAIRD_DOWNTILT_5_insert_code/Benchmarks/default_config.json Benchmarks/
cp WAIRD_DOWNTILT_5_insert_code/Benchmarks/data/W_WAIRD_DOWNTILT_5 Benchmarks/data/
```

若不希望覆盖原 `default_config.json`，只合并：

```text
Benchmarks/default_config_snippet.json
```

然后将本地处理后的目录复制到：

```text
Benchmarks/data/WAIRD_DOWNTILT_5/
```

最终工程结构：

```text
LAC-MAS/
  Benchmarks/
    WairdDowntilt5.h
    Benchmarks.cpp
    Benchmarks.h
    default_config.json
    data/
      W_WAIRD_DOWNTILT_5
      WAIRD_DOWNTILT_5/
        00001/bs_00.txt ... bs_04.txt
        ...
```

---

# C. 你的原算法需要修改的地方

## 必须做

1. 使用 5 个 MPI 进程：

```bash
mpirun -np 5 ./LAC-MAS WAIRD_DOWNTILT_5
```

2. 问题维度为 5，范围为 `[0, 15]` 度。配置文件已经设置。

3. 每个 Agent 调用同样的 5 维候选向量，但 `local_evaluation()` 会按照 MPI rank 自动读取不同 BS 的本地用户数据，因此 5 个 Agent 的局部黑箱函数不同。

算法的核心“完整向量副本共识”不需要改。

## 强烈建议做

包内 `patches/` 提供可直接替换的三个文件：

```bash
cp patches/framework.cpp framework/framework.cpp
cp patches/main.cpp framework/main.cpp
cp patches/LAC-MAS.cpp ICML26_LAC-MAS/LAC-MAS.cpp
```

这些修改包括：

- `framework.cpp`：用 `MPI_Type_size` 正确统计通信字节数；
- `main.cpp`：检查 MPI 进程数是否等于 5，并修复 rank 0 gather 缓冲区的花括号；
- `LAC-MAS.cpp`：
  - 把 LLM prompt 中写死的“300维”改成运行时 `dim`；
  - 两邻居环形拓扑初始权重改为 `1 / neighbor_count`，保证权重和为 1；
  - LLM 更新后的邻居权重归一化为 1，而不是 0.75；
  - 5 维问题默认 swarm size 从 300 改为 100，减少计算量。该项不是理论必需，可改回 300。

---

# D. 集群编译与运行

## D1. 编译

在 `TEVC2024-MASOIE` 目录：

```bash
mpic++ -std=c++11 \
  MASOIE.cpp \
  ../framework/framework.cpp \
  ../framework/main.cpp \
  ../Benchmarks/Benchmarks.cpp \
  -o MASOIE
```

保留你原工程所需的 Eigen、curl、Python/LLM 链接选项；本 benchmark 本身不新增任何集群依赖。

## D2. 单个场景运行

```bash
export WAIRD_ENV_ID=00001
mpirun -np 5 ./MASOIE WAIRD_DOWNTILT_5
```

或：

```bash
./run_one_environment.sh 00001
```

`WAIRD_ENV_ID` 决定本次加载哪个环境。未设置时默认使用 `00001`。

## D3. 多个场景分别运行

```bash
./run_environment_batch.sh 1 20
```

这会独立运行 `00001` 到 `00020`，日志保存在：

```text
results/00001.log
...
results/00020.log
```

多个环境不是同一次协作中的节点，而是多个独立现实网络实例，用于统计算法稳定性。

---

# E. 输出解释

根进程会输出类似：

```text
WAIRD_DOWNTILT_5 environment=00001 metrics:
AvgSE=..., P10-SE=..., P5-SE=..., Min-SE=..., BS-balance=...
algorithm performance: [fitness:..., disagreement:..., communication cost:...]
```

- `fitness`：五个局部复合目标的平均值，越小越好；
- `AvgSE`：30 个 UE 的频带平均谱效率均值，越大越好；
- `P10-SE` / `P5-SE`：先对每个 UE 平均所有子载波，再对 UE 求分位数；
- `BS-balance`：五个 BS 平均用户谱效率的方差，越小越均衡；
- `disagreement`：5 个 Agent 对联合俯角配置副本的不一致程度；
- `communication cost`：平均每个 Agent 的发送字节数。

最终共识配置可能是：

```text
[4.2, 7.1, 5.8, 10.3, 6.4]
```

所有 Agent 对这份联合方案达成一致，但 BS0 到 BS4 分别执行对应分量，并不是五个 BS 使用相同俯角。

---

# F. 建议的第一版论文设置

先选 10–20 个环境，每个环境独立运行 5 次随机种子，报告跨环境和跨随机种子的均值与标准差：

- fitness；
- AvgSE；
- P10-SE；
- disagreement；
- communication cost。

第一版可将该实验描述为：

> Each WAIR-D environment is treated as an independent five-cell wireless network. The five BSs act as autonomous agents and maintain local copies of the joint downtilt vector. Each agent evaluates a private black-box objective using only the users associated with its own BS. Through neighbor-to-neighbor communication, the agents reach consensus on the joint configuration, while the five BSs are allowed to execute different downtilt values.

## 最终配置与迭代诊断输出

本修改包新增 `patches/framework.h`，并更新了 `patches/framework.cpp`、`patches/main.cpp` 和 `patches/LAC-MAS.cpp`。替换后，程序会集中输出：

- 每个 Agent 的最终 5 维俯角配置；
- 每个 Agent 在自身局部函数上的最终 fitness；
- 每个 Agent 的终止迭代次数；
- 终止原因（速度阈值或评价预算）；
- 最终外部速度 L1 范数；
- 每个 Agent 的通信字节数；
- 平均配置、各维最大跨度、Agent 两两平均/最大 L2 距离。

替换命令：

```bash
cp patches/framework.h framework/framework.h
cp patches/framework.cpp framework/framework.cpp
cp patches/main.cpp framework/main.cpp
cp patches/LAC-MAS.cpp ICML26_LAC-MAS/LAC-MAS.cpp
```
