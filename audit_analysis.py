"""LLM Statistical Reasoning Benchmark & Error Audit.

Scores LLM responses to a curated set of statistics prompts, breaks down
accuracy by category and prompting style, categorises failure modes, and
tests whether accuracy differences are statistically significant.

Inputs (same folder):
    prompts_dataset.csv  - prompt_id, category, difficulty, prompt,
                           ground_truth, solution_notes
    model_responses.csv  - prompt_id, model, prompting_style, response,
                           correct (1/0), failure_mode

Outputs:
    error_breakdown.png  - failure-mode counts by model
    summary_stats.csv    - accuracy by model and category

Usage:
    pip install -r requirements.txt
    python audit_analysis.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

DATA_DIR = Path(__file__).parent
ALPHA = 0.05


def load_data() -> pd.DataFrame:
    prompts = pd.read_csv(DATA_DIR / "prompts_dataset.csv")
    responses = pd.read_csv(DATA_DIR / "model_responses.csv")
    df = responses.merge(prompts, on="prompt_id", how="inner", validate="many_to_one")
    assert len(df) == len(responses), "every response must match exactly one prompt"
    return df


def two_proportion_ztest(x1: int, n1: int, x2: int, n2: int):
    """Two-sided two-proportion z-test (pooled standard error)."""
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value


def main() -> None:
    df = load_data()
    models = sorted(df["model"].unique())
    styles = sorted(df["prompting_style"].unique())

    # --- Overall accuracy by model -------------------------------------
    overall = (
        df.groupby("model")["correct"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "correct", "count": "n", "mean": "accuracy"})
    )
    print("== Overall accuracy ==")
    for model, row in overall.iterrows():
        print(f"  {model}: {row['accuracy']:.1%} ({int(row['correct'])}/{int(row['n'])})")

    # --- Accuracy by model x category ----------------------------------
    by_cat = df.pivot_table(
        index="category", columns="model", values="correct", aggfunc="mean"
    )
    print("\n== Accuracy by category ==")
    print(by_cat.map(lambda v: f"{v:.0%}").to_string())

    # --- Accuracy by prompting style -----------------------------------
    by_style = (
        df.groupby("prompting_style")["correct"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "correct", "count": "n", "mean": "accuracy"})
    )
    print("\n== Accuracy by prompting style ==")
    for style, row in by_style.iterrows():
        print(f"  {style}: {row['accuracy']:.1%} ({int(row['correct'])}/{int(row['n'])})")

    # --- Failure modes (incorrect answers only) ------------------------
    wrong = df[df["correct"] == 0]
    failure_counts = (
        wrong.groupby(["failure_mode", "model"]).size().unstack(fill_value=0)
    )
    print("\n== Failure modes (incorrect answers only) ==")
    print(failure_counts.to_string())
    print(f"\nTotal incorrect: {len(wrong)} of {len(df)}")

    # --- Chi-square test: model x correctness --------------------------
    contingency = pd.crosstab(df["model"], df["correct"])
    chi2, p_chi, dof, _ = stats.chi2_contingency(contingency)
    print("\n== Chi-square test of independence (model x correctness) ==")
    print(contingency.to_string())
    print(f"  chi2 = {chi2:.3f}, df = {dof}, p = {p_chi:.4f}")
    print(
        "  Significant at alpha = 0.05"
        if p_chi < ALPHA
        else "  Not significant at alpha = 0.05"
    )

    # --- Two-proportion z-test: prompting styles -----------------------
    if len(styles) == 2:
        s1, s2 = styles
        x1 = int(df.loc[df["prompting_style"] == s1, "correct"].sum())
        n1 = int((df["prompting_style"] == s1).sum())
        x2 = int(df.loc[df["prompting_style"] == s2, "correct"].sum())
        n2 = int((df["prompting_style"] == s2).sum())
        z, p_z = two_proportion_ztest(x1, n1, x2, n2)
        print(f"\n== Two-proportion z-test ({s1} vs {s2}) ==")
        print(f"  {s1}: {x1}/{n1} = {x1 / n1:.1%}")
        print(f"  {s2}: {x2}/{n2} = {x2 / n2:.1%}")
        print(f"  z = {z:.3f}, p = {p_z:.4f}")
        print(
            "  Significant at alpha = 0.05"
            if p_z < ALPHA
            else "  Not significant at alpha = 0.05"
        )

    # --- Save summary stats --------------------------------------------
    summary = by_cat.copy()
    summary.loc["Overall"] = overall["accuracy"]
    summary.to_csv(DATA_DIR / "summary_stats.csv")

    # --- Chart: failure-mode breakdown ---------------------------------
    counts = wrong.groupby(["failure_mode", "model"]).size().unstack(fill_value=0)
    counts = counts.reindex(counts.sum(axis=1).sort_values(ascending=True).index)
    ax = counts.plot.barh(figsize=(10, 6))
    ax.set_xlabel("Number of incorrect answers")
    ax.set_ylabel("Failure mode")
    ax.set_title("LLM error breakdown by failure mode")
    ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig(DATA_DIR / "error_breakdown.png", dpi=150)
    print("\nSaved: error_breakdown.png, summary_stats.csv")


if __name__ == "__main__":
    main()
