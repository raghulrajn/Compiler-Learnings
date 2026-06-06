# Segmentation fault on KNN kernel

```python
import torch
import triton
import triton.language as tl

@triton.jit
def _knn_kernel(
    Points_ptr, X2_ptr, Out_idx_ptr,
    B, N, C,
    K: tl.constexpr,
    stride_b_pts, stride_n_pts, stride_c_pts,
    stride_b_x2, stride_n_x2,
    stride_b_out, stride_n_out, stride_k_out,
    BLOCK_N: tl.constexpr,
    BLOCK_NTARGET: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    row_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = row_offsets < N
    q_x2_ptr = X2_ptr + pid_b * stride_b_x2 + row_offsets * stride_n_x2
    q_x2 = tl.load(q_x2_ptr, mask=row_mask, other=0.0)[:, None]
    topk_dists = tl.full((BLOCK_N, K), value=float('inf'), dtype=tl.float32)
    topk_indices = tl.full((BLOCK_N, K), value=-1, dtype=tl.int32)
    for target_start in range(0, N, BLOCK_NTARGET):
        target_offsets = target_start + tl.arange(0, BLOCK_NTARGET)
        target_mask = target_offsets < N
        t_x2_ptr = X2_ptr + pid_b * stride_b_x2 + target_offsets * stride_n_x2
        t_x2 = tl.load(t_x2_ptr, mask=target_mask, other=0.0)[None, :]
        dist_acc = tl.zeros((BLOCK_N, BLOCK_NTARGET), dtype=tl.float32)
        for c in range(C):
            q_ptr = Points_ptr + pid_b * stride_b_pts + row_offsets[:, None] * stride_n_pts + c * stride_c_pts
            t_ptr = Points_ptr + pid_b * stride_b_pts + target_offsets[None, :] * stride_n_pts + c * stride_c_pts
            dist_acc -= 2.0 * (tl.load(q_ptr, mask=row_mask[:, None]) * tl.load(t_ptr, mask=target_mask[None, :]))
        dist_matrix = q_x2 + t_x2 + dist_acc
        dist_matrix = tl.where(row_offsets[:, None] == target_offsets[None, :], float('inf'), dist_matrix)
        dist_matrix = tl.where(row_mask[:, None] & target_mask[None, :], dist_matrix, float('inf'))
        for j in range(BLOCK_NTARGET):
            curr_dist = tl.sum(tl.where(tl.arange(0, BLOCK_NTARGET)[None, :] == j, dist_matrix, 0.0), axis=1)
            curr_idx = tl.full((BLOCK_N,), value=target_start + j, dtype=tl.int32)
            for k_idx in range(K):
                k_mask = tl.arange(0, K) == k_idx
                target_dist_val = tl.sum(tl.where(k_mask[None, :], topk_dists, 0.0), axis=1)
                target_idx_val = tl.sum(tl.where(k_mask[None, :], topk_indices, 0), axis=1)
                mask_replace = curr_dist < target_dist_val
                topk_dists = tl.where(mask_replace[:, None] & k_mask[None, :], curr_dist[:, None], topk_dists)
                topk_indices = tl.where(mask_replace[:, None] & k_mask[None, :], curr_idx[:, None], topk_indices)
                curr_dist = tl.where(mask_replace, target_dist_val, curr_dist)
                curr_idx = tl.where(mask_replace, target_idx_val, curr_idx)
    out_ptr = Out_idx_ptr + pid_b * stride_b_out + row_offsets[:, None] * stride_n_out + tl.arange(0, K)[None, :]
    tl.store(out_ptr, topk_indices.to(tl.int64), mask=row_mask[:, None])

B, N, C, K = 1, 64, 3, 4  # tiny dims to isolate
points = torch.randn(B, N, C, device='cuda')
x2 = (points * points).sum(dim=-1).contiguous()
out = torch.empty((B, N, K), dtype=torch.int64, device='cuda')
grid = (triton.cdiv(N, 32), B)
print('Launching kernel...')
_knn_kernel[grid](points, x2, out, B, N, C, K, *points.stride(), *x2.stride(), *out.stride(), BLOCK_N=32, BLOCK_NTARGET=64)
torch.cuda.synchronize()
print('Done:', out.shape)
```

## Debug cmd

```shell
TRITON_PRINT_AUTOTUNING=1 TRITON_INTERPRET=1  python test.py
```

```shell
Exit code 139
/bin/bash: line 1: 2782331 Segmentation fault
```

```
With BLOCK_NTARGET and K both marked as constexpr with values of 64 and 16 respectively, the compiler unrolls thousands of iterations, creating massive register pressure and code bloat that overwhelms the Ampere backend.
```

Then tried with simple addition kernel and that too resulted in segmentation fault. 

Suspected that problem would be uncompatible versions of python or cuda or compiler. Couldn't isolate where the bug is. 

```shell
nvidia-smi | head -10 && echo "---" && python3 -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda)" && python3 -c "import triton; print('Triton:', triton.__version__)" 2>/dev/null | head -20

Sat Jun  6 01:25:44 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.05              Driver Version: 595.71.05      CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:1A:00.0 Off |                  N/A |
|  0%   39C    P8             26W /  350W |      38MiB /  24576MiB |      0%      Default |
---
PyTorch: 2.12.0+cu130 | CUDA: 13.0
Triton: 3.7.0
```

```python
import triton
import torch
print('GPU count:', torch.cuda.device_count())
print('GPU name:', torch.cuda.get_device_name(0))
print('Compute cap:', torch.cuda.get_device_capability(0))
try:
    props = triton.runtime.driver.active.utils.get_device_properties(0)
    print('Triton device props:', props)
except Exception as e:
    print('Triton props error:', e)
```
```
GPU count: 4
GPU name: NVIDIA GeForce RTX 3090
Compute cap: (8, 6)
Triton device props: {'max_shared_mem': 101376, 'max_num_regs': 65536, 'multiprocessor_count': 82, 'warpSize': 32, 'sm_clock_rate': 1740000, 'mem_clock_rate': 9751000, 'mem_bus_width': 384}
```

```
Triton's bundled ptxas: CUDA 12.8
PyTorch's bundled ptxas: CUDA 13.0
System /usr/bin/ptxas: CUDA 11.5
/usr/local/cuda-12.6/bin/ptxas: CUDA 12.6
```

```shell
# Quick check: does the crash happen during LLVM IR -> PTX or ptxas?
# Use strace to trace syscalls to see where it dies
timeout 15 strace -e trace=file,process python3 /home/RUS_CIP/st189432/MT-3970/test_minimal.py 2>&1 | tail -30
```

```
newfstatat(48, "", {st_mode=S_IFREG|0644, st_size=1095, ...}, AT_EMPTY_PATH) = 0
newfstatat(48, "", {st_mode=S_IFREG|0644, st_size=1095, ...}, AT_EMPTY_PATH) = 0
openat(AT_FDCWD, "/home/RUS_CIP/st189432/.triton/cache/CC4ZYZDLUPJXCR3TXU7GGQQMR7XFAX7CZCED3UYSY3Z4QO6X464A/_minimal.source", O_RDONLY|O_CLOEXEC) = 48
newfstatat(48, "", {st_mode=S_IFREG|0644, st_size=6282, ...}, AT_EMPTY_PATH) = 0
newfstatat(48, "", {st_mode=S_IFREG|0644, st_size=6282, ...}, AT_EMPTY_PATH) = 0
openat(AT_FDCWD, "/home/RUS_CIP/st189432/.triton/cache/CC4ZYZDLUPJXCR3TXU7GGQQMR7XFAX7CZCED3UYSY3Z4QO6X464A/_minimal.ttir", O_RDONLY|O_CLOEXEC) = 48
newfstatat(48, "", {st_mode=S_IFREG|0644, st_size=3570, ...}, AT_EMPTY_PATH) = 0
newfstatat(48, "", {st_mode=S_IFREG|0644, st_size=3570, ...}, AT_EMPTY_PATH) = 0
openat(AT_FDCWD, "/home/RUS_CIP/st189432/.triton/cache/CC4ZYZDLUPJXCR3TXU7GGQQMR7XFAX7CZCED3UYSY3Z4QO6X464A/_minimal.ttgir", O_RDONLY|O_CLOEXEC) = 48
newfstatat(48, "", {st_mode=S_IFREG|0644, st_size=4185, ...}, AT_EMPTY_PATH) = 0
newfstatat(48, "", {st_mode=S_IFREG|0644, st_size=4185, ...}, AT_EMPTY_PATH) = 0
openat(AT_FDCWD, "/home/RUS_CIP/st189432/.triton/cache/CC4ZYZDLUPJXCR3TXU7GGQQMR7XFAX7CZCED3UYSY3Z4QO6X464A/_minimal.llir", O_RDONLY|O_CLOEXEC) = 48
newfstatat(48, "", {st_mode=S_IFREG|0644, st_size=3304, ...}, AT_EMPTY_PATH) = 0
newfstatat(48, "", {st_mode=S_IFREG|0644, st_size=3304, ...}, AT_EMPTY_PATH) = 0
openat(AT_FDCWD, "/home/RUS_CIP/st189432/.triton/cache/CC4ZYZDLUPJXCR3TXU7GGQQMR7XFAX7CZCED3UYSY3Z4QO6X464A/_minimal.ptx", O_RDONLY|O_CLOEXEC) = 48
newfstatat(48, "", {st_mode=S_IFREG|0644, st_size=3958, ...}, AT_EMPTY_PATH) = 0
newfstatat(48, "", {st_mode=S_IFREG|0644, st_size=3958, ...}, AT_EMPTY_PATH) = 0
openat(AT_FDCWD, "/home/RUS_CIP/st189432/.triton/cache/CC4ZYZDLUPJXCR3TXU7GGQQMR7XFAX7CZCED3UYSY3Z4QO6X464A/_minimal.cubin", O_RDONLY|O_CLOEXEC) = 48
newfstatat(48, "", {st_mode=S_IFREG|0644, st_size=4712, ...}, AT_EMPTY_PATH) = 0
newfstatat(48, "", {st_mode=S_IFREG|0644, st_size=4712, ...}, AT_EMPTY_PATH) = 0
newfstatat(AT_FDCWD, "/home/RUS_CIP/st189432/MT-3970/.venv/lib/python3.11/site-packages/torch/_subclasses", {st_mode=S_IFDIR|0755, st_size=4096, ...}, 0) = 0
newfstatat(AT_FDCWD, "/home/RUS_CIP/st189432/MT-3970/.venv/lib/python3.11/site-packages/torch/_subclasses/schema_check_mode.py", {st_mode=S_IFREG|0644, st_size=9705, ...}, 0) = 0
newfstatat(AT_FDCWD, "/home/RUS_CIP/st189432/MT-3970/.venv/lib/python3.11/site-packages/torch/_subclasses/schema_check_mode.py", {st_mode=S_IFREG|0644, st_size=9705, ...}, 0) = 0
openat(AT_FDCWD, "/home/RUS_CIP/st189432/MT-3970/.venv/lib/python3.11/site-packages/torch/_subclasses/__pycache__/schema_check_mode.cpython-311.pyc", O_RDONLY|O_CLOEXEC) = 48
newfstatat(48, "", {st_mode=S_IFREG|0644, st_size=14736, ...}, AT_EMPTY_PATH) = 0
newfstatat(48, "", {st_mode=S_IFREG|0644, st_size=14736, ...}, AT_EMPTY_PATH) = 0
minimal (tl.full + .to(int64) store):
OK: tensor([-1, -1, -1, -1], device='cuda:0')
exit_group(0)                           = ?
+++ exited with 0 +++
[1]+  Done                    timeout 15 strace -e trace=file,process python3 /home/RUS_CIP/st189432/MT-3970/test_minimal.py 2>&1 | tail -30
done
```

The strace shows it's loading from the Triton cache. But even after clearing the cache. Seg fault occurs

Even the most basic for i in range(N): acc = acc + 1.0 crashes. This confirms that ANY runtime loop in a Triton kernel causes a segfault on this system. 

System has multiple version of CUDA runtime installed, CUDA12 and CUDA13. 
The pip list shows triton 3.5.1, but when we ran `print(triton.__version__)` it showed 3.7.0. 

Completely removed every conflicting packages and reinstalled the suitable versions. 

```python
@triton.jit
def _minimal(Out_ptr, N, K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
```

Even then crashed for a simple loop.

```shell
gdb -batch -ex "set pagination off" -ex "run" -ex "bt full" -ex "quit" --args python3 /home/RUS_CIP/st189432/MT-3970/test_one_loop.py 2>&1 | head -60
```

```
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
[New Thread 0x7ffea05ff640 (LWP 2804115)]
[New Thread 0x7ffe9f9c9640 (LWP 2804116)]
[New Thread 0x7ffe9ed93640 (LWP 2804117)]
[New Thread 0x7ffe9e15d640 (LWP 2804118)]
[New Thread 0x7ffe9d527640 (LWP 2804119)]
[New Thread 0x7ffe9c8f1640 (LWP 2804120)]
[New Thread 0x7ffe9bcbb640 (LWP 2804121)]
[New Thread 0x7ffe9b085640 (LWP 2804122)]
[New Thread 0x7ffe9a44f640 (LWP 2804123)]
[New Thread 0x7ffe99819640 (LWP 2804124)]
[New Thread 0x7ffe98be3640 (LWP 2804125)]
[New Thread 0x7ffe97fad640 (LWP 2804126)]
[New Thread 0x7ffe97377640 (LWP 2804127)]
[New Thread 0x7ffe96741640 (LWP 2804128)]
[New Thread 0x7ffe95b0b640 (LWP 2804129)]
[New Thread 0x7ffe94ed5640 (LWP 2804130)]
[New Thread 0x7ffe9429f640 (LWP 2804131)]
[New Thread 0x7ffe93669640 (LWP 2804132)]
[New Thread 0x7ffe92a33640 (LWP 2804133)]
[New Thread 0x7ffe91dfd640 (LWP 2804134)]
[New Thread 0x7ffe911c7640 (LWP 2804135)]
[New Thread 0x7ffe90591640 (LWP 2804136)]
[New Thread 0x7ffe8f95b640 (LWP 2804137)]
[New Thread 0x7ffe8ed25640 (LWP 2804138)]
[New Thread 0x7ffe8e0ef640 (LWP 2804139)]
[New Thread 0x7ffe8d4b9640 (LWP 2804140)]
[New Thread 0x7ffe8c883640 (LWP 2804141)]
[Detaching after vfork from child process 2804142]
[New Thread 0x7ffe7d9ff640 (LWP 2804195)]
[New Thread 0x7ffe6d77b640 (LWP 2804202)]
[New Thread 0x7ffe6cb45640 (LWP 2804203)]
[New Thread 0x7ffe6acee640 (LWP 2804204)]
[Detaching after vfork from child process 2804205]
[Detaching after vfork from child process 2804206]
[Detaching after vfork from child process 2804217]
[Detaching after vfork from child process 2804218]

Thread 1 "python3" received signal SIGSEGV, Segmentation fault.
0x0000000000526db0 in _PyDictKeys_StringLookup (dk=0x0, key='options') at ../Objects/dictobject.c:1011
1011	../Objects/dictobject.c: No such file or directory.
#0  0x0000000000526db0 in _PyDictKeys_StringLookup (dk=0x0, key='options') at ../Objects/dictobject.c:1011
        kind = <optimized out>
        hash = <optimized out>
#1  0x0000000000505c26 in specialize_dict_access (kind=<optimized out>, base_op=95, hint_op=159, values_op=154, name=<optimized out>, type=0x7ed2c90, instr=0x9841a92, owner=<triton._C.libtriton.ir.builder at remote 0x7ffe6a00e1b0>) at ../Python/specialize.c:625
        keys = <optimized out>
        index = <optimized out>
        cache = 0x9841a94
        dictptr = 0x7ffe6a00e198
        dict = <optimized out>
        cache = <optimized out>
        dictptr = <optimized out>
        dict = <optimized out>
        keys = <optimized out>
        index = <optimized out>
        value = <optimized out>
        hint = <optimized out>
#2  _Py_Specialize_StoreAttr (name=<optimized out>, instr=0x9841a92, owner=<triton._C.libtriton.ir.builder at remote 0x7ffe6a00e1b0>) at ../Python/specialize.c:813
        type = 0x7ed2c90
```

The crash is in Python's "specialization" code — specifically in _PyDictKeys_StringLookup where dk=0x0 (null pointer dereference). This happens when Python's adaptive specializer tries to optimize attribute access (STORE_ATTR) on triton._C.libtriton.ir.builder object.

The issue continues up the stack through _Py_Specialize_StoreAttr, which means Python is attempting to optimize how attributes are stored on this C extension object, but encounters a null dictionary keys pointer during the process.

This is a CPython bug in how the specializer handles C extension types like Triton's ir.builder that may have unusual attribute lookup behavior or missing __dict__ structures. The specializer only kicks in after repeated bytecode execution, which explains why simple kernels without loops work fine. looped kernels crash because the repeated attribute access during loop compilation triggers the specializer's faulty optimization path.


Updated python to stable 3.12 release and the code compiled and run successfully.

