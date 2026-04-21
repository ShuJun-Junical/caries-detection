# Slurm 集群基础信息

- 采集日期: 2026-04-20
- 采集来源: `sinfo`、`scontrol show nodes -o`、`srun + lscpu + nvidia-smi`
- 说明: GPU 型号与显存为在对应节点申请 `--gres=gpu:1` 后探测结果。

## 1. 分区与节点概览

| 分区 | 状态 | 节点数 (A/I/O/T) | 节点列表 |
|---|---|---|---|
| dlq | up | 1/4/0/5 | compute01-05 |
| hpcq (default) | up | 0/2/0/2 | compute06-07 |

## 2. 登录节点配置

| 节点 | CPU 型号 | CPU 核心数 | 内存总数 | GPU 型号 | GPU 卡数 | 单卡显存 |
|---|---|---:|---:|---|---:|---:|
| login | Intel(R) Xeon(R) Gold 5218R CPU @ 2.10GHz | 80 逻辑核 (40 物理核, 2 路 x 20 核, HT=2) | 125.6 GiB (Slurm 配置 144000 MiB) | 无 | 0 | - |

## 3. GPU 计算节点配置

| 节点 | 分区 | CPU 型号 | CPU 核心数 | 内存总数 | GPU 型号 | GPU 卡数 | 单卡显存 |
|---|---|---|---:|---:|---|---:|---:|
| compute01 | dlq | Intel(R) Xeon(R) Gold 5218R CPU @ 2.10GHz | 80 逻辑核 (40 物理核) | 125.5 GiB | NVIDIA GeForce RTX 3090 | 2 | 24576 MiB |
| compute02 | dlq | Intel(R) Xeon(R) CPU E5-2620 v2 @ 2.10GHz | 24 逻辑核 (12 物理核) | 31.4 GiB | NVIDIA GeForce GTX TITAN X | 4 | 12288 MiB |
| compute03 | dlq | Intel(R) Xeon(R) CPU E5-2620 v2 @ 2.10GHz | 12 核 (Thread(s) per core = 1) | 125.9 GiB | NVIDIA GeForce GTX TITAN X | 4 | 12288 MiB |
| compute04 | dlq | Intel(R) Xeon(R) CPU E5-2620 v2 @ 2.10GHz | 12 核 (Thread(s) per core = 1) | 141.6 GiB | NVIDIA GeForce RTX 3090 | 2 | 24576 MiB |
| compute05 | dlq | Intel(R) Xeon(R) CPU E5-2620 v2 @ 2.10GHz | 12 核 (Thread(s) per core = 1) | 141.6 GiB | NVIDIA GeForce RTX 3090 | 2 | 24576 MiB |
| compute06 | hpcq | Intel(R) Xeon(R) CPU E5-2620 v2 @ 2.10GHz | 12 核 (Thread(s) per core = 1) | 141.6 GiB | NVIDIA Tesla P100-PCIE-16GB | 2 | 16384 MiB |
| compute07 | hpcq | Intel(R) Xeon(R) CPU E5-2620 v2 @ 2.10GHz | 12 核 (Thread(s) per core = 1) | 141.6 GiB | NVIDIA Tesla P100-PCIE-16GB | 2 | 16384 MiB |

## 4. 备注

1. 集群为异构 GPU 环境，包含 RTX 3090、GTX TITAN X、Tesla P100。
2. 若不申请 GPU 资源，部分节点执行 `nvidia-smi` 可能显示 `No devices were found`。
3. 登录节点在 Slurm 中状态含 `IDLE+DRAIN+INVALID_REG`，并有 RealMemory 低于配置值的提示。
