# Name That LLM

Which of 18 large language models wrote this passage?

This repository is our work on the Kaggle competition **Name That LLM**. Each passage was written by one of 18 models from six labs (OpenAI, Google, Anthropic, Qwen, ZAI and Moonshot AI), and the task is to name the model. Beyond the leaderboard, the goal is to understand which style traits set each model and each lab apart.

## The problem

- **Training data:** 2,699 passages generated from 150 prompts. Every prompt was answered by every model, one passage each (one passage is missing).
- **Hidden test set:** about 4,000 passages from prompts that **never appear in training**. The classifier must generalize to new topics, not memorize prompts.
- **Passages:** 40–250 words, with a randomly chosen length.
- **Metric:** balanced accuracy (mean per-class recall). The mix of models differs between the public and private leaderboards.
- **Hard part:** sibling models from the same lab are very similar (for example `gpt-5.6-luna`, `gpt-5.6-sol` and `gpt-5.6-terra`).
- **Submission:** a Kaggle notebook, offline, within 9 hours of CPU or GPU, writing `submission.parquet` with columns `id,model`.

## How we evaluate

Because test prompts are unseen, **every evaluation uses 5-fold GroupKFold on `prompt_id`**: all passages of a prompt stay in the same fold. Random splits are never used, since they would leak topic information and overstate accuracy.

Model experiments also report per-class recall, lab-level accuracy, the share of errors that fall within the same lab, and accuracy by passage length. The baseline run includes a shuffled-label control to check for leakage: it must score at chance, and it does.

## Results so far

| Experiment | Result |
|---|---|
| Length EDA (E001) | Passage length does **not** identify the model (ε² = 0.009, at noise level). The number of sentences does vary by model, which makes sentence length a candidate style feature. |
| Baseline (E002) | Character n-gram TF-IDF + logistic regression: **0.312 balanced accuracy** (chance 0.056). 52% of errors are confusions between sibling models of the same lab. Shuffled-label control: 0.056. |

**Target:** at least **0.412** balanced accuracy (baseline + 10 points) under the same validation protocol, with local validation and the public leaderboard moving in the same direction over 3 consecutive submissions.

**Next step:** a Kaggle submission notebook for the baseline, to check that local validation tracks the leaderboard.

## Repository layout

```
src/name_that_llm/          project code (importable package)
  labels.py                 model -> lab mapping
  validation.py             prompt-grouped folds and out-of-fold predictions
  metrics.py                balanced accuracy, within-lab error share, accuracy by length
  baseline.py               fixed TF-IDF + logistic regression baseline
  eda_length.py             length features and statistical tests
  experiments/              runnable experiments (python -m ...)
tests/                      pytest suite
kb/                         research knowledge base (see below)
  mission/CHALLENGE.md      mission, success criteria and constraints
  ACTIVE.md                 current objective and next step
  research/                 hypotheses, experiments and findings
  research/data/            raw metrics and figures from each experiment
scripts/                    Limina framework tooling (not project code)
data/                       competition data (not in git)
```

## Getting started

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Download the competition data (needs Kaggle credentials and accepted rules)
uv run kaggle competitions download -c name-that-llm -p data
unzip data/name-that-llm.zip -d data

# Run the tests
uv run pytest tests

# Reproduce the experiments
source .venv/bin/activate
python -m name_that_llm.experiments.eda_length     # E001, writes kb/research/data/eda_length/
python -m name_that_llm.experiments.run_baseline   # E002, writes kb/research/data/E002/ (~4 min on CPU)
```

The baseline experiment uses fixed fold seeds, so a re-run reproduces the stored results exactly.

## Research workflow

The project follows [Limina](AGENTS.md), a research workflow in which every result comes from a chain of three notes in `kb/research/`:

1. **Hypothesis (H):** a falsifiable claim, with confirm and reject criteria written **before** running anything.
2. **Experiment (E):** setup, exact commands, results and analysis, with raw metrics in `kb/research/data/`.
3. **Finding (F):** what the evidence established, what is still open, and what to do next.

| Chain | Question | Outcome |
|---|---|---|
| H001 → E001 → F001 | Does passage length depend on the model? | No (confirmed) |
| H002 → E002 → F002 | Do baseline errors concentrate within each lab? | Yes, narrowly: 52% (confirmed) |

`kb/ACTIVE.md` always holds the current objective and next step. Run `python3 scripts/kb_validate.py` to check the knowledge base.

## Conventions

- GitHub issues, pull requests and commit messages are written in English.
- Project code lives in `src/name_that_llm/`. `scripts/` belongs to the Limina framework.
- Experiments are planned in a GitHub issue and land through a pull request.

## Key dates

- **2027-01-06:** last day to accept the rules and merge teams.
- **2027-01-13, 23:59 UTC:** final submission deadline.

## License

[MIT](LICENSE)
