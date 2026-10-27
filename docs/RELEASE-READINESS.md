# Engineering preview gates

- [x] Two model-pinned cascade definitions packaged
- [x] Shared single-client benchmark runner, telemetry, structural validation
- [x] CPU-only tests included
- [x] Two dataset delivery modes with checksums and offline tests
- [ ] Download the pinned raw HF source from a clean repository checkout and run its benchmark-owned adapter
- [ ] Final dataset licensing and Gold provenance review
- [ ] Run both Compose stacks from this exact release on a compatible GPU host
- [ ] Verify runtime model identity and immutable CUDA base image digests
- [ ] Collect real new benchmark runs using the new runner and the same publicly redistributable frozen datasets
- [ ] Inspect memory/CPU/GPU telemetry attribution and monitoring limitations
- [ ] Check offline/online quality agreement and input eligibility on Laya
- [ ] Obtain a definitive publication license and third-party notices
- [ ] Test clean clone and container build
- [ ] Public release approval

The results/runs directory is empty until validated runs are submitted; no reference GPU measurements are included.
