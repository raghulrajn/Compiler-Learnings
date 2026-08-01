# Model Training - Quick Notes

1. **Lazy layers + multi-GPU don't mix.** `LazyLinear` doesn't know its shape until it sees real data, but DDP checks parameter shapes right when it wraps the model. Fix: run one dummy forward pass on a real batch *before* wrapping in DDP, so the layers are already initialized.

2. **Caching tensors in a dict isn't free like OS file caching.** Every file loaded into `self._data_cache` stayed in memory forever, and each DataLoader worker had its *own* copy of the cache. With 4 workers and 32k files, that's how we hit 125GB RAM. Fixed with an LRU cache (OrderedDict, capped at 64 files/worker).

3. **GPU% can lie.** The farthest-point-sampling loop looked vectorized and fine, but it was ~3000 tiny sequential GPU operations, each waiting on the last. Kernel launch overhead dominated, not actual compute — confirmed because batch size 4 vs 32 took the same time. Fixed by chunking points into groups to cut the number of sequential steps.

4. **Watch for accidental N×N computations.** A distance metric was computing a full pairwise distance matrix just to pull out the diagonal (the only part actually needed). Fine at small scale, very expensive at real scale (tens of thousands of points). Lesson: if two point sets already have a 1:1 correspondence, don't reach for an all-pairs function.

5. **`kill -9` doesn't always work.** Processes stuck in `D` state (uninterruptible sleep, e.g. a hung NFS read) can't receive signals until the kernel-level wait resolves — sometimes never. Workaround: run the risky read in a background thread and just stop waiting on it after a timeout, rather than trying to kill it.
