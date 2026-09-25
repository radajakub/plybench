# Reasoning and action-error analysis

The reasoning analysis separates questions that should not be answered by one measurement:

1. **Was the action good?** Minimax grades legal moves and supplies regret, severity, tactical labels,
   and refutation depth. This part is deterministic and does not call an LLM.
2. **Did the output match the reasoning?** A blinded judge extracts the move the trace concludes; Python
   compares it with the executed move. Malformed and illegal outputs are counted separately.
3. **What went wrong in the reasoning?** A codebook is induced on a small mistake-enriched sample,
   frozen, and applied to traces. Every label must cite text present in the trace.

4. **Do the two agree?** The reasoning codes are crossed with the solver's own tactical labels, and with
   the length of the trace. This is the join the separate stages cannot make.

Only minimax-solvable games can enter this analysis.

## The three levels

A code sits at one of three tiers, recorded as its scope:

- **universal** — a reasoning step that would fail the same way in any game;
- **family** — tied to the rules underneath, so it occurs under every obfuscation of them;
- **presentation** — only possible given one formulation.

The level is a **claim, and it is measured rather than enforced**. Annotation offers every active code on
every move, whatever level it claims, and the report says where each code was actually found. A code
declared presentation-specific that then turns up under three presentations was mis-levelled, and that is
a result. Restricting the codes on offer to the ones expected to apply would make "this code only occurs
here" true by construction.

Induction assigns a level per proposal one batch at a time. The restructuring pass revises them with the
whole taxonomy in view, but only in the generalising direction — moving a code down a level needs the name
of the family or presentation, which is not in front of that call.

## Open-weight and commercial models are never pooled

`trace_kind` splits cells into `raw_reasoning` (models we host, which return the chain of thought) and
`provider_summary` (commercial APIs, which return a summary their own summariser wrote). An error rate
over a summary is a rate over what the summariser kept and is not the same measurement. Pooling that
crosses the two prints a warning naming the group; pool by `trace_kind` as well, or read the groups
apart.

## How completeness is measured

There is no discovery/evaluation partition. Induction reads on the order of two thousand moves out of a
corpus of a hundred and seventy thousand, so reserving a fifth of the data to protect one figure against
it was out of proportion. The exclusion is exact instead: the codebook records every move induction was
shown _and the game it came from_, and the completeness figure is computed over the games it never read.
Game level rather than move level, because two moves of one game are two positions one ply apart.

Every prevalence report therefore carries two uncovered rates:

- `uncovered` over every annotated move -- descriptive, and inflated by the traces the codebook was
  built from;
- the same rate holding back the induced games, printed with its numerator and denominator. This is the
  completeness figure. A rate without its denominator cannot be acted on.

The freeze threshold is **5% of annotated moves carrying an uncovered error**, fixed in
`reasoning/stats.py` as `FREEZE_THRESHOLD` before the first wave was run. Above it, the `OTHER`
descriptions go back into induction and another wave runs. It is reported per model as well as pooled: a
pooled 4% hiding one model at 20% means that model's errors are not in the codebook and none of its rates
are comparable with the others'.

What the split also provided was a commitment device. That is replaced by freezing the codebook: its
content hash is `Codebook.version`, every annotation records it, and any change to the taxonomy after
seeing results creates a visible new version rather than a quiet edit.

Taxonomy discovery deliberately does not mirror the corpus frequency. It groups moves by
`(game, player, opponent, action outcome)`. Small suboptimal strata are kept whole, large ones are
capped at 32 moves, and smaller samples of optimal (4) and non-decision (2) traces act as controls. At
least two available moves from every represented stratum must be read before induction can declare
saturation. Thus a weak model's many errors cannot hide the few errors made by a strong model. Before any
of that, at most two moves are drawn from any one `(cell, position)`: 60% of traced moves sit in a
position their own model reached more than once, and reading the same board again buys no new failure
mode. This sampling affects only which traces build the vocabulary; prevalence is never reweighted.

## Recommended workflow

First inspect the deterministic action analysis. This is read-only and costs nothing:

```bash
uv run python scripts/analyze_reasoning.py --experiment ttt \
  --report funnel traces labels --pool model model,presentation
```

Preview paid calls before running them:

```bash
uv run python scripts/analyze_reasoning.py --experiment ttt \
  --do all --dry-run
```

Build a fresh taxonomy on discovery games. A labelled codebook avoids extending an older exploratory
taxonomy by accident:

```bash
uv run python scripts/analyze_reasoning.py --experiment ttt \
  --model openai:gpt-5.4 --codebook confirmatory-v1 \
  --do induce etalons
```

The diversity defaults can be written explicitly with
`--discovery-suboptimal-cap 32 --discovery-optimal-cap 4 --discovery-non-decision-cap 2
--discovery-coverage-per-stratum 2 --discovery-state-cap 2`. Induction prints a warning if a suboptimal
matchup stratum exists in the corpus but received no move, how many moves the per-position cap dropped,
and a Good-Turing / Chao1 estimate of how much of the failure space the run has not seen.

For a robustness check, induce independent forks with different sampling seeds (for example
`--codebook seed-a --seed a` and `--codebook seed-b --seed b`) and compare the categories they recover.
Choose these seeds in advance rather than selecting the most appealing taxonomy afterward.

Before paying for the census, measure whether the codebook is complete. `--coverage-test` annotates only
moves from games induction never read, so the uncovered rate means what it says, and the run ends with the
per-model breakdown the freeze decision rests on:

```bash
uv run python scripts/analyze_reasoning.py --experiment ttt \
  --model metacentrum:gemma-4 --codebook confirmatory-v1 \
  --coverage-test --per-stratum 8 --seed wave1 \
  --do annotate
```

If any model is above the 5% threshold, read the printed `uncovered` descriptions, run `--do induce`
again to extend the codebook, and repeat on a _new_ sample. The cost of a missed code is then one wave,
not one census.

Apply the frozen codebook and run trace/action consistency. Use at least two independent judge models;
repeat this command with each model. Both passes cover all moves and resume safely after interruption:

```bash
uv run python scripts/analyze_reasoning.py --experiment ttt \
  --model openai:gpt-5.4 --codebook confirmatory-v1 \
  --do consistency annotate

uv run python scripts/analyze_reasoning.py --experiment ttt \
  --model openai:gpt-5-mini --codebook confirmatory-v1 \
  --do consistency annotate
```

Finally produce model-level and presentation-level statistical results and a JSON artifact:

```bash
uv run python scripts/analyze_reasoning.py --experiment ttt \
  --codebook confirmatory-v1 \
  --report funnel labels consistency mistakes reliability \
  --pool model model,presentation --standardize \
  --json analysis/ttt/confirmatory-v1.json
```

Finally, the two joins, which need no further paid calls:

```bash
uv run python scripts/analyze_reasoning.py --experiment ttt \
  --codebook confirmatory-v1 --report correlations --pool trace_kind
```

Read the `lift` line under each code: the code's rate in that bucket over its rate everywhere. 1.0 means
the two classifications say nothing about each other, which is itself an answer — it would mean the trace
is narration written beside the decision rather than an account of it. The columns overlap by
construction, because one move carries several tactical labels, so no independence test belongs on this
table.

`--json` and `--dump-moves` write new analysis outputs; omit them for a read-only report. None of these
commands modify benchmark results under `results/`.

## Reading the results

Read the uncovered rate on the induction-naive games, never the pooled descriptive one, and read it per
model. Read it with its denominator, which is printed beside it.

Always inspect counts as well as percentages:

- prevalence needs enough annotated moves;
- decomposition needs suboptimal moves covered by both consistency and annotation;
- rare code rates are unstable even when total coverage is large;
- a code's denominator is every annotated move, including presentations it does not claim to cover, so a
  presentation-level code reads low overall and its `by_presentation` row is the one to read;
- agreement between two judges measures reliability, not validity.

Before publication, manually audit a stratified sample containing detected errors, clean traces,
judge disagreements, uncovered (`other`) errors, and costly slips. Holding back the induced games
prevents taxonomy leakage but cannot prove that an LLM judge interpreted every trace correctly.

## Sampling parameters

`--discovery-state-cap` bounds how many moves may be drawn from any one `(cell, position)`. It is a
sampling parameter, not a filtering stage: it changes which moves are drawn, never which moves exist, and
it needs no similarity threshold, because positions are equal or they are not. On the `ttt` corpus,
lowering it from unlimited to the default 2 takes the discovery pool from 5046 moves to 3892 for the same
position coverage.

`--min-instances` refuses to call induction saturated until that many error instances have been coded.
The default is 300: a 20-30 code taxonomy over a skewed distribution needs several hundred instances
before its tail is represented at all, and a low guard lets a run claim completeness having seen almost
nothing.

`--seed` orders discovery deterministically and selects the sample. Choose it in advance. The move export
includes `game_uid`, so the induction hold-out can be checked against the codebook from outside.
