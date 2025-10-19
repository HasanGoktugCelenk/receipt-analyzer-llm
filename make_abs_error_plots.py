# make_abs_error_plots.py
# Improve the "Absolute Error per Numeric Question" visualization:
# - horizontal grouped bars
# - sorted by difficulty (mean abs error)
# - wrapped question labels
# - clear annotations and grid
#
# Run:
#   python make_abs_error_plots.py
#
# Inputs (defaults):
#   eval/out/llm_eval_per_q-1.5B.json
#   eval/out/llm_eval_per_q-7B.json
#
# Outputs:
#   eval/report_out/abs_error_per_question_better.png
#   eval/report_out/abs_error_percent_per_question_better.png

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

# ---------- config ----------
PERQ_15 = Path("eval/out/llm_eval_per_q-1.5B.json")
PERQ_7  = Path("eval/out/llm_eval_per_q-7B.json")
OUTDIR  = Path("eval/report_out")
PALETTE = sns.color_palette("tab10")
sns.set_theme(style="whitegrid", context="talk")

def load_perq(path, tag):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for r in data:
        if r.get("type") != "num":
            continue
        rows.append({
            "model_tag": tag,
            "question": r.get("q"),
            "pred": float(r["pred"]) if isinstance(r.get("pred"), (int,float)) else np.nan,
            "expect": float(r["expect"]) if isinstance(r.get("expect"), (int,float)) else np.nan,
        })
    return pd.DataFrame(rows)

def wrap(q, width=42):
    return "\n".join(textwrap.wrap(q, width=width, break_long_words=False, replace_whitespace=False)) or q

def make_grouped_horizontal(df_num, outfile, value_col, title, xlab, fmt="{:,.2f}", pad=0.01):
    """
    df_num: columns [question, model_tag, abs_err | pct_err]
    value_col: 'abs_err' or 'pct_err'
    """

    # compute order by mean error across models (hardest first)
    order = (
        df_num.groupby("question")[value_col]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    df_plot = df_num.copy()
    df_plot["question_wrapped"] = df_plot["question"].apply(lambda s: wrap(s, width=48))
    # keep same order after wrapping
    wrap_map = {q: wrap(q, 48) for q in order}
    order_wrapped = [wrap_map[q] for q in order]

    models = df_plot["model_tag"].unique().tolist()
    models.sort()  # stable order: 1.5B, 7B
    colors = dict(zip(models, PALETTE[:len(models)]))

    # Prepare figure
    n_q = len(order_wrapped)
    height = max(6, 0.5 * n_q + 2)  # dynamic height
    plt.figure(figsize=(14, height))
    ax = plt.gca()

    # Bar geometry
    bar_h = 0.35
    offsets = np.linspace(-bar_h/2, bar_h/2, num=len(models))

    # y positions by question
    y_base = np.arange(n_q)

    # Plot bars per model
    for i, m in enumerate(models):
        sub = df_plot[df_plot["model_tag"] == m]
        # align to order
        vals = sub.set_index("question")[[value_col]].reindex(order)[value_col].values
        ax.barh(y_base + offsets[i], vals, height=bar_h, label=m, color=colors[m])

        # Annotate values at the end of each bar
        for yi, v in enumerate(vals):
            if pd.isna(v):
                continue
            x = v if np.isfinite(v) else 0.0
            ax.text(x + pad*max(1.0, np.nanmax(vals)), y_base[yi] + offsets[i],
                    fmt.format(x),
                    va="center", ha="left", fontsize=11, color="black")

    # Axis/labels
    ax.set_yticks(y_base)
    ax.set_yticklabels(order_wrapped, fontsize=12)
    ax.invert_yaxis()  # highest error on top
    ax.set_xlabel(xlab)
    ax.set_title(title, pad=12)
    ax.legend(title="Model", loc="best", frameon=True)

    # X limits with a little headroom
    xmax = df_plot[value_col].replace([np.inf, -np.inf], np.nan).max()
    if pd.notna(xmax):
        ax.set_xlim(0, xmax * 1.15)

    # Style
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTDIR / outfile, dpi=220, bbox_inches="tight")
    plt.close()

def main():
    # Load
    df15 = load_perq(PERQ_15, "Qwen2.5-1.5B")
    df7  = load_perq(PERQ_7,  "Qwen2.5-7B")
    df   = pd.concat([df15, df7], ignore_index=True)

    # Metrics
    df["abs_err"] = (df["pred"] - df["expect"]).abs()
    # Avoid divide-by-zero in percent error
    df["pct_err"] = np.where(df["expect"] != 0, df["abs_err"] / df["expect"] * 100.0, np.nan)

    # Plot 1: Absolute error (SEK)
    make_grouped_horizontal(
        df_num=df,
        outfile="abs_error_per_question_better.png",
        value_col="abs_err",
        title="Absolute Error by Numeric Question (lower is better)",
        xlab="Absolute Error (SEK)",
        fmt="{:,.2f}",
        pad=0.012,
    )

    # Plot 2: Relative error (%)
    make_grouped_horizontal(
        df_num=df,
        outfile="abs_error_percent_per_question_better.png",
        value_col="pct_err",
        title="Relative Error by Numeric Question (lower is better)",
        xlab="Relative Error (%)",
        fmt="{:,.1f}%",
        pad=0.06,
    )

    print(f"Saved improved plots to: {OUTDIR}")

if __name__ == "__main__":
    main()
