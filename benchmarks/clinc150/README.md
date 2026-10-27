# CLINC150 confidence H1 benchmark

This benchmark reads original, unaltered source test data using a versioned source ID from datasets/registry.json. Full train/test source acquisition lives under datasets/sources/clinc150/, not here. No dataset is truncated during download or by this benchmark setup.

From Community root (with Pico and Ultra model services online):

    sh benchmarks/clinc150/run.sh --dry-run
    sh benchmarks/clinc150/run.sh --repeats 2

The benchmark does not retrain models or change the frozen H1 threshold. The server must support the full original class taxonomy; unsupported choice width fails instead of silently reducing classes.
