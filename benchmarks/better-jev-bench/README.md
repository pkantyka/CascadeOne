# better-jev-bench: frozen confidence H1 benchmark

The complete pinned source consists of 11 original held-out data files. All are downloadable unmodified from datasets/sources/better-jev-bench/download.py. Benchmark setup processing occurs only in benchmarks/better-jev-bench/setups/ and the named adapter in benchmarks/adapters.py.

The current inference runner can benchmark eight text-only tasks: BANKING77, CLINC150, CFPB, CUAD, Civil Comments, GoEmotions, LEDGAR, MASSIVE. The remaining three original sources - Atari Head, OS-Atlas, ScreenSpot v2 - reference external images. They are preserved and downloadable but are not run as text-only tasks; an image retrieval and multimodal inference protocol must be supplied before benchmarking them.

From Community root with model servers up:

    python benchmarks/better-jev-bench/run.py --dry-run
    python benchmarks/better-jev-bench/run.py --task cuad --repeats 2
    python benchmarks/better-jev-bench/run.py --repeats 2

BJB BANKING77/CLINC150 contain records overlapping the original-source tests and other source splits; this never causes source deletion. If an analysis needs unique examples, deduplicate only within its benchmark processing, with documented criteria, and retain the original source data intact.
