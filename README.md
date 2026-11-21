# LLM Statistical Reasoning Benchmark & Error Audit

A small, reproducible audit of how well large language models reason about
statistics: 40 hand-written, multi-step prompts with verified ground-truth
answers, scored across two models and two prompting styles, with the
accuracy gap tested for statistical significance instead of just eyeballed.

## Why this exists

It bridges a B.Sc. in Statistics with day-to-day AI evaluation work. The
prompts are not trivia — they target the exact places LLMs are known to
slip: base rates, p-values, regression assumptions, and priors. The analysis
then treats the models' accuracy like any other experiment: contingency
tables, a chi-square test, and a two-proportion z-test.

## Contents

| File | What it is |
|---|---|
| `prompts_dataset.csv` | 40 prompts with verified ground-truth answers and worked solutions |
| `model_responses.csv` | Scored responses (see the demo note below) |
| `audit_analysis.py` | Scoring, failure-mode breakdown, significance tests, chart |
| `requirements.txt` | pandas, scipy, matplotlib, numpy |
| `error_breakdown.png` | Failure-mode counts by model (generated) |
| `summary_stats.csv` | Accuracy by model and category (generated) |

## The prompts

40 prompts in four categories, 10 each, mixed difficulty:

- **Conditional Probability** — Bayes' rule, base rates, Monty Hall, screening tests
- **Hypothesis Testing** — p-values, Type I/II errors, power, confidence intervals, A/B tests
- **Regression Interpretation** — coefficients, R², omitted-variable bias, VIF, heteroscedasticity
- **Bayesian Inference** — priors, Beta posteriors, credible intervals, Bayes factors

Every prompt has a verified ground-truth answer and a worked solution in
`solution_notes`, so scoring is checkable, not a matter of opinion.

## Failure-mode rubric

Each wrong answer is tagged with one failure mode:

- **base-rate neglect** — ignoring the prior/prevalence (e.g. answering 99% on a screening-test question)
- **p-value misinterpretation** — treating p as P(H0 is true), or a CI as a probability about this interval
- **arithmetic slip** — right method, wrong number
- **assumption violation** — missing heteroscedasticity, multicollinearity, or another broken assumption
- **prior misuse** — dropping or mis-weighting the prior in a Bayesian update
- **regression misread** — e.g. "slope 0 means no effect" when the pattern is nonlinear

## How to run

```bash
pip install -r requirements.txt
python audit_analysis.py
```

It prints overall and per-category accuracy, the failure-mode table, the
chi-square test (model × correctness), the two-proportion z-test
(zero-shot vs chain-of-thought), and saves `error_breakdown.png` and
`summary_stats.csv`.

## Demo findings

> **Demo note:** the shipped `model_responses.csv` is an illustrative demo
> dataset (two anonymised models, "Model A" zero-shot and "Model B"
> chain-of-thought) so the pipeline runs end to end. For a real audit,
> replace it with responses collected from the target models — same columns,
> one row per prompt per model — and re-run the script. Nothing in the
> analysis depends on the demo data.

| | Model A (zero-shot) | Model B (chain-of-thought) |
|---|---|---|
| Overall | 65.0% (26/40) | 82.5% (33/40) |
| Conditional Probability | 70% | 80% |
| Hypothesis Testing | 60% | 80% |
| Regression Interpretation | 60% | 80% |
| Bayesian Inference | 70% | 90% |

Failure modes across the 21 wrong answers: base-rate neglect (5),
arithmetic slip (4), prior misuse (4), assumption violation (3),
regression misread (3), p-value misinterpretation (2). Chain-of-thought
eliminated both p-value errors and cut every other failure mode — but the
headline gap is **not statistically significant**: chi-square p = 0.127,
two-proportion z-test p = 0.075. At 40 prompts per model the study is
underpowered; the honest conclusion is "suggestive, needs more prompts",
which is exactly why the significance test is in the pipeline.

## Collecting real responses

To audit real models, prompt each model with the 40 questions in
`prompts_dataset.csv` (once zero-shot, once with "think step by step"),
record the answer, score it against `ground_truth`, and tag any failure
with the rubric above. Drop the rows into `model_responses.csv` and re-run.

## Limitations

- 40 prompts is a pilot-scale benchmark; the significance tests above show
  it directly.
- Scoring is binary (right/wrong); partial credit would need a graded
  rubric.
- The demo responses are illustrative, not measured model outputs.
