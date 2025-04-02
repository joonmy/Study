# PyTorch Distributed Training Guide

This guide covers essential components of PyTorch distributed training, including initialization methods, environment variables, and tools.

---

## ✅ Basic Concepts

- **Rank**: Process ID (e.g., 0, 1, 2, ...)
- **World Size**: Total number of processes
- **Backend**: Communication backend (use `nccl` for GPU, `gloo` for CPU)
- **Init Method**: How to initialize the process group (e.g., `env://`, `file://`, `tcp://`)

---

## 🧪 Code Example
```python
import torch
import torch.distributed as dist
import argparse
import os

# Check if distributed is available
print(torch.distributed.is_available())  # True

# Initialize parser for --local-rank
parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", "--local_rank", type=int)
parser.add_argument("--world_size", int(os.environ['WORLD_SIZE']))
args = parser.parse_args()
local_rank = args.local_rank
world_size = args.world_size

# Set CUDA device
torch.cuda.set_device(local_rank)

# Initialize process group
# Recommended: env:// with environment variables
# Make sure the following environment variables are set:
# MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK

dist.init_process_group(
    backend='nccl',
    init_method='env://',
    world_size=world_size,
    rank=local_rank
)

# Check if initialized
print(torch.distributed.is_initialized())  # True
print(dist.get_rank())      # Current process's rank
print(dist.get_world_size())  # Total number of processes # from 환경변수

# Synchronize all processes
torch.distributed.barrier()
```

---

## 🚀 Launching Distributed Training

### torch.distributed.launch (Deprecated in favor of torchrun)

#### Single Node Multi-GPU:
```bash
python -m torch.distributed.launch --nproc-per-node=NUM_GPUS_YOU_HAVE \
    YOUR_TRAINING_SCRIPT.py --arg1 --arg2
```

#### Multi Node Multi-GPU:
```bash
python -m torch.distributed.launch --nproc-per-node=NUM_GPUS_YOU_HAVE \
    --nnodes=2 --node-rank=0 --master-addr="192.168.1.1" --master-port=1234 \
    YOUR_TRAINING_SCRIPT.py --arg1 --arg2
```

### torchrun (Recommended in PyTorch >= 1.9)
```bash
torchrun --nproc-per-node=NUM_GPUS_YOU_HAVE \
    YOUR_TRAINING_SCRIPT.py --arg1 --arg2
```

If using `--use_env`, retrieve `local_rank` with:
```python
local_rank = int(os.environ["LOCAL_RANK"])
```

---

## 🧵 Initialization Methods

1. **env://** *(Recommended)*
    - Uses environment variables:
      - `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`

2. **file://**
    - Uses a file system path to coordinate initialization.

3. **tcp://**
    - Uses a TCP URL to coordinate initialization.

---

## 🧱 DistributedDataParallel (DDP)
```python
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    output_device=local_rank
)
```

---

## 🧵 multiprocessing.spawn for DDP
You can also use `torch.multiprocessing.spawn` to spawn multiple processes:

```python
import torch.multiprocessing as mp

mp.spawn(
    main,
    args=(world_size, args),
    nprocs=world_size,
    join=True
)
```
Each spawned subprocess will receive a unique `rank` as the first argument.

---

## 🔧 Miscellaneous

### Monitored Barrier
```python
torch.distributed.monitored_barrier(timeout=datetime.timedelta(seconds=1800))
```
- Safer version of `barrier()` with timeout & debugging support

---

### 중요
torch.distributed.launch를 쓰면, args로 --local_rank or --local-rank가 자동으로 할당. 
--use-env를 사용하면 parser.add_argument("--local_rank", int(os.environ['LOCAL_RANK'])) 로 받아야함.

world_size는 torch.distributed.launch --nproc_per_node=N 으로 설정하면 환경 변수가 지정되어서
parser.add_argument("--world_size", int(os.environ['WORLD_SIZE'])) 이렇게 받으면 됨.



 