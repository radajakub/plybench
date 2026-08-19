# Reasoning and action-error analysis

The reasoning analysis separates questions that should not be answered by one measurement:

1. **Was the action good?** Minimax grades legal moves and supplies regret, severity, tactical labels,
   and refutation depth. This part is deterministic and does not call an LLM.
2. **Did the output match the reasoning?** A blinded judge extracts the move the trace concludes; Python
   compares it with the executed move. Malformed and illegal outputs are counted separately.
3. **What went wrong in the reasoning?** A codebook is induced on discovery games, frozen, and applied to
   traces. Every label must cite text present in the trace.

Only minimax-solvable games can enter this analysis.

## Discovery and evaluation

Complete games are assigned deterministically to two partitions. The default is:

- **20% discovery:** build and consolidate the mistake codebook and choose etalons;
- **80% evaluation:** primary held-out estimates of mistake prevalence and the suboptimal-action
  decomposition.

Consistency and annotation run on all moves. Reports show three populations for each judge:

- `evaluation`: the primary held-out result;
- `all`: the most precise descriptive result;
- `discovery`: a diagnostic for taxonomy fit and possible overfitting.

Use the same split and discovery-sampling parameters in every command belonging to a study. A codebook
records them and refuses incompatible continued induction.

Taxonomy discovery deliberately does not mirror the corpus frequency. Within discovery it groups moves
by `(game, player, opponent, action outcome)`. Small suboptimal strata are kept whole, large ones are
capped at 32 moves, and smaller samples of optimal (4) and non-decision (2) traces act as controls. At
least two available moves from every represented stratum must be read before induction can declare
saturation. Thus a weak model's many errors cannot hide the few errors made by a strong model. This
sampling affects only which traces build the vocabulary; evaluation prevalence is never reweighted.

```bash
# Defaults; writing them explicitly makes a research script self-documenting.
SPLIT="--discovery-fraction 0.2 --split-seed reasoning-evaluation-v1"
```

The shell variable is only shorthand in the examples. You can spell both arguments out instead.

## Recommended workflow

First inspect the deterministic action analysis. This is read-only and costs nothing:

```bash
uv run python scripts/analyze_reasoning.py --experiment ttt \
  --report funnel traces labels --pool model model,presentation
```

Preview paid calls before running them:

```bash
uv run python scripts/analyze_reasoning.py --experiment ttt \
  --discovery-fraction 0.2 --split-seed reasoning-evaluation-v1 \
  --do all --dry-run
```

Build a fresh taxonomy on discovery games. A labelled codebook avoids extending an older exploratory
taxonomy by accident:

```bash
uv run python scripts/analyze_reasoning.py --experiment ttt \
  --discovery-fraction 0.2 --split-seed reasoning-evaluation-v1 \
  --model openai:gpt-5.4 --codebook confirmatory-v1 \
  --do induce etalons
```

The diversity defaults can be written explicitly with
`--discovery-suboptimal-cap 32 --discovery-optimal-cap 4 --discovery-non-decision-cap 2
--discovery-coverage-per-stratum 2`. Induction prints a warning if a suboptimal matchup stratum exists in
the corpus but received no discovery game. Increase `--discovery-fraction` only when that warning or the
held-out uncovered rate shows that 20% is inadequate.

For a robustness check, induce independent forks with different sampling seeds (for example
`--codebook seed-a --seed a` and `--codebook seed-b --seed b`) and compare the categories they recover.
Choose these seeds in advance rather than selecting the most appealing taxonomy afterward.

Apply the frozen codebook and run trace/action consistency. Use at least two independent judge models;
repeat this command with each model. Both passes cover all moves and resume safely after interruption:

```bash
uv run python scripts/analyze_reasoning.py --experiment ttt \
  --discovery-fraction 0.2 --split-seed reasoning-evaluation-v1 \
  --model openai:gpt-5.4 --codebook confirmatory-v1 \
  --do consistency annotate

uv run python scripts/analyze_reasoning.py --experiment ttt \
  --discovery-fraction 0.2 --split-seed reasoning-evaluation-v1 \
  --model openai:gpt-5-mini --codebook confirmatory-v1 \
  --do consistency annotate
```

Finally produce model-level and presentation-level statistical results and a JSON artifact:

```bash
uv run python scripts/analyze_reasoning.py --experiment ttt \
  --discovery-fraction 0.2 --split-seed reasoning-evaluation-v1 \
  --codebook confirmatory-v1 \
  --report funnel labels consistency mistakes reliability \
  --pool model model,presentation --standardize \
  --json analysis/ttt/confirmatory-v1.json
```

`--json` and `--dump-moves` write new analysis outputs; omit them for a read-only report. None of these
commands modify benchmark results under `results/`.

## Reading the results

Use `evaluation` as the primary mistake prevalence and decomposition. Use `all` as a higher-precision
descriptive estimate. Compare them with `discovery`: a large gap, or an `other` rate that is much lower
on discovery, suggests taxonomy overfitting.

Always inspect counts as well as percentages:

- prevalence needs enough annotated evaluation moves;
- decomposition needs suboptimal evaluation moves covered by both consistency and annotation;
- rare code rates are unstable even when total coverage is large;
- agreement between two judges measures reliability, not validity.

Before publication, manually audit a stratified sample containing detected errors, clean traces,
judge disagreements, uncovered (`other`) errors, and costly slips. A held-out split prevents taxonomy
leakage but cannot prove that an LLM judge interpreted every trace correctly.

## Split parameters

`--discovery-fraction` accepts a number strictly between 0 and 1. Increase it only if discovery does not
contain enough diverse errors to stabilize the taxonomy. For a large corpus, the 0.2 default is normally
preferable because strong models produce few suboptimal moves and therefore need a large evaluation set.

`--split-seed` changes which complete games enter each partition. Changing it defines a different study;
do not choose a seed after comparing results. The move export includes `analysis_split`, and JSON reports
record the split parameters.
