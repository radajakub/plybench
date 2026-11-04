# the repo-local scripts import their helpers from here; they live in the package (plybench.cli.shared), shared with the
# installed tools plybench-run and plybench-train
from plybench.cli.shared import (  # noqa: F401
    add_source_args,
    add_training_source_args,
    benchmark_from_args,
    build_op,
    discover_matchups,
    load_paired_results,
    training_from_args,
)
