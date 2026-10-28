# Engineering preview gates

- [x] Two model-pinned cascade definitions packaged
- [x] Shared System One inference path for benchmark and serving; dual-path latency, telemetry and structural validation tested on CPU
- [x] CPU-only tests included
- [x] Two dataset delivery modes with checksums and offline tests
- [ ] Download the pinned raw HF source from a clean repository checkout and run its benchmark-owned adapter
- [ ] Final dataset licensing and Gold provenance review
- [ ] Run both Compose stacks, including the unified inference service, on a compatible GPU host
- [ ] Verify runtime model identity and immutable CUDA base image digests
- [ ] Re-run all retained performance benchmarks through the unified engine, including both latency paths, on appropriate frozen datasets
- [ ] Inspect memory/CPU/GPU telemetry attribution and monitoring limitations
- [ ] Check offline/online quality agreement and input eligibility on Laya
- [ ] Obtain a definitive publication license and third-party notices
- [ ] Test clean clone and container build
- [ ] Public release approval

The results/runs directory is empty until validated runs are submitted; no reference GPU measurements are included.
