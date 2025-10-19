# make_llm_eval_report.py
# Build side-by-side visualizations + a PDF report comparing 1.5B vs 7B runs.
# Default inputs (override via CLI flags):
#   eval/out/llm_eval_per_q-1.5B.json
#   eval/out/llm_eval_summary-1.5B.json
#   eval/out/llm_eval_per_q-7B.json
#   eval/out/llm_eval_summary-7B.json
#
# Run:
#   python make_llm_eval_report.py
# Or with explicit paths:
#   python make_llm_eval_report.py --perq_15 "eval/out/llm_eval_per_q-1.5B.json" --sum_15 "eval/out/llm_eval_summary-1.5B.json" --perq_7 "eval/out/llm_eval_per_q-7B.json" --sum_7 "eval/out/llm_eval_summary-7B.json" --outdir "eval/report_out"

import os, json, argparse, math, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# --- Plot theme (clean report style) ---
warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid", context="talk")
PALETTE = sns.color_palette("tab10")

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_run(tag, perq_path, summary_path):
    perq = load_json(perq_path)
    summ = load_json(summary_path)

    rows = []
    for r in perq:
        rows.append({
            "model_tag": tag,
            "type": r.get("type"),
            "question": r.get("q"),
            "answer_raw": r.get("ans"),
            "pred_num": float(r["pred"]) if isinstance(r.get("pred"), (int,float)) else np.nan,
            "pred_bool": str(r["pred"]) if isinstance(r.get("pred"), str) else np.nan,
            "expect_num": float(r["expect"]) if isinstance(r.get("expect"), (int,float)) else np.nan,
            "expect_bool": str(r["expect"]) if isinstance(r.get("expect"), str) else np.nan,
            "ok": bool(r.get("ok")),
            "latency_s": float(r.get("latency_s")) if r.get("latency_s") is not None else np.nan,
        })
    df = pd.DataFrame(rows)

    meta = {
        "model": summ.get("model", tag),
        "n_numeric": summ.get("n_numeric"),
        "numeric_accuracy": summ.get("numeric_accuracy"),
        "n_bool": summ.get("n_bool"),
        "bool_accuracy": summ.get("bool_accuracy"),
        "latency_mean_s": summ.get("latency_mean_s"),
        "latency_p50_s": summ.get("latency_p50_s"),
        "latency_p95_s": summ.get("latency_p95_s"),
        "limit_context_receipts": summ.get("limit_context_receipts"),
        "top_store": (summ.get("meta") or {}).get("top_store"),
        "recent_month": (summ.get("meta") or {}).get("recent_month"),
        "threshold": (summ.get("meta") or {}).get("threshold"),
    }
    return df, meta

def compute_metrics(df):
    out = {}
    num = df[df["type"] == "num"].copy()
    if not num.empty:
        num["abs_err"] = (num["pred_num"] - num["expect_num"]).abs()
        num["squared_err"] = (num["pred_num"] - num["expect_num"])**2
        num["pct_err"] = (num["abs_err"] / num["expect_num"].replace(0, np.nan)).abs() * 100.0

        out["num_acc"] = float(num["ok"].mean())
        out["mae"] = float(num["abs_err"].mean())
        out["rmse"] = float(np.sqrt(num["squared_err"].mean()))
        out["within_1_sek"] = float((num["abs_err"] <= 1.0).mean())
        out["median_abs_err"] = float(num["abs_err"].median())
    else:
        out.update({"num_acc": np.nan, "mae": np.nan, "rmse": np.nan,
                    "within_1_sek": np.nan, "median_abs_err": np.nan})

    boo = df[df["type"] == "bool"].copy()
    out["bool_acc"] = float((boo["ok"] == True).mean()) if not boo.empty else np.nan

    if df["latency_s"].notna().any():
        out["lat_mean"] = float(df["latency_s"].mean())
        out["lat_p50"] = float(df["latency_s"].median())
        out["lat_p95"] = float(df["latency_s"].quantile(0.95))
    else:
        out["lat_mean"] = out["lat_p50"] = out["lat_p95"] = np.nan

    return out

def nice_number(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "–"
    return f"{x:.3f}" if isinstance(x, float) else str(x)

def plot_all(df_all, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / "LLM_Eval_Report.pdf"

    # Compute per-model metrics table
    table_rows = []
    for tag in sorted(df_all["model_tag"].unique()):
        sub = df_all[df_all["model_tag"] == tag]
        m = compute_metrics(sub)
        table_rows.append({
            "Model": tag,
            "Numeric Accuracy": nice_number(m["num_acc"]),
            "Bool Accuracy": nice_number(m["bool_acc"]),
            "MAE (SEK)": nice_number(m["mae"]),
            "RMSE (SEK)": nice_number(m["rmse"]),
            "Within ±1 SEK": nice_number(m["within_1_sek"]),
            "Median Abs Err (SEK)": nice_number(m["median_abs_err"]),
            "Latency mean (s)": nice_number(m["lat_mean"]),
            "Latency p50 (s)": nice_number(m["lat_p50"]),
            "Latency p95 (s)": nice_number(m["lat_p95"]),
        })
    metrics_df = pd.DataFrame(table_rows)
    metrics_df.to_csv(outdir / "summary_metrics.csv", index=False)

    with PdfPages(pdf_path) as pdf:
        # 1) Accuracy bars (numeric vs bool) by model
        acc_data = []
        for tag in sorted(df_all["model_tag"].unique()):
            sub = df_all[df_all["model_tag"] == tag]
            num_acc = sub[sub.type=="num"]["ok"].mean() if not sub[sub.type=="num"].empty else np.nan
            bool_acc = sub[sub.type=="bool"]["ok"].mean() if not sub[sub.type=="bool"].empty else np.nan
            acc_data.append({"Model": tag, "Metric": "Numeric", "Accuracy": num_acc})
            acc_data.append({"Model": tag, "Metric": "YES/NO", "Accuracy": bool_acc})
        acc_df = pd.DataFrame(acc_data)

        plt.figure(figsize=(10,6))
        ax = sns.barplot(data=acc_df, x="Metric", y="Accuracy", hue="Model", palette=PALETTE)
        ax.set_ylim(0, 1.05)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", label_type="edge", padding=2)
        plt.title("LLM Accuracy by Task Type")
        plt.tight_layout()
        plt.savefig(outdir / "accuracy_bars.png", dpi=180)
        pdf.savefig(); plt.close()

        # 2) Scatter: predicted vs expected (numeric only)
        num_all = df_all[df_all["type"]=="num"].copy()
        if not num_all.empty:
            plt.figure(figsize=(8,8))
            ax = sns.scatterplot(
                data=num_all, x="expect_num", y="pred_num", hue="model_tag",
                style="model_tag", s=140, edgecolor="black", palette=PALETTE
            )
            lo = float(np.nanmin([num_all["expect_num"].min(), num_all["pred_num"].min()])) - 5
            hi = float(np.nanmax([num_all["expect_num"].max(), num_all["pred_num"].max()])) + 5
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray")
            ax.set_xlabel("Expected (SEK)")
            ax.set_ylabel("Predicted (SEK)")
            plt.title("Predicted vs Expected (Numeric Questions)")
            plt.tight_layout()
            plt.savefig(outdir / "scatter_pred_vs_expect.png", dpi=180)
            pdf.savefig(); plt.close()

        # 3) Absolute error by question, grouped by model
        if not num_all.empty:
            num_all["abs_err"] = (num_all["pred_num"] - num_all["expect_num"]).abs()
            def short_q(q):
                return (q[:40] + "…") if len(q) > 43 else q
            num_all["q_short"] = num_all["question"].apply(short_q)

            g = sns.catplot(
                data=num_all, x="q_short", y="abs_err", hue="model_tag",
                kind="bar", height=6, aspect=1.8, palette=PALETTE
            )
            g.set_axis_labels("Question", "Absolute Error (SEK)")
            for ax in g.axes.flat:
                for c in ax.containers:
                    ax.bar_label(c, fmt="%.2f", padding=2)
                ax.tick_params(axis="x", rotation=25)
            plt.suptitle("Absolute Error per Numeric Question", y=1.02)
            plt.tight_layout()
            g.savefig(outdir / "abs_error_per_question.png", dpi=180)
            pdf.savefig(g.fig); plt.close(g.fig)

        # 4) Latency distribution (KDE + rug) by model
        plt.figure(figsize=(10,6))
        ax = sns.kdeplot(
            data=df_all, x="latency_s", hue="model_tag", fill=True,
            common_norm=False, alpha=0.35, palette=PALETTE
        )
        sns.rugplot(data=df_all, x="latency_s", hue="model_tag", height=0.05, ax=ax, palette=PALETTE)
        plt.xlabel("Latency (seconds)")
        plt.title("Latency Distribution")
        plt.tight_layout()
        plt.savefig(outdir / "latency_distribution.png", dpi=180)
        pdf.savefig(); plt.close()

        # 5) Violin + box for latency
        plt.figure(figsize=(9,6))
        sns.violinplot(data=df_all, x="model_tag", y="latency_s", inner=None, palette=PALETTE)
        sns.boxplot(data=df_all, x="model_tag", y="latency_s", width=0.18, showcaps=True, boxprops={'zorder': 2})
        plt.xlabel("Model")
        plt.ylabel("Latency (seconds)")
        plt.title("Latency Spread by Model")
        plt.tight_layout()
        plt.savefig(outdir / "latency_violin_box.png", dpi=180)
        pdf.savefig(); plt.close()

        # 6) Metrics table as an image
        fig, ax = plt.subplots(figsize=(12, 2 + 0.45*len(metrics_df)))
        ax.axis("off")
        tbl = ax.table(
            cellText=metrics_df.values,
            colLabels=metrics_df.columns,
            cellLoc="center",
            loc="upper left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.2)
        plt.title("Summary Metrics", loc="left")
        plt.tight_layout()
        plt.savefig(outdir / "summary_metrics_table.png", dpi=180)
        pdf.savefig(); plt.close()

    print(f"\nSaved figures and PDF report to: {outdir}")
    print(f"- {pdf_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perq_15", default="eval/out/llm_eval_per_q-1.5B.json")
    ap.add_argument("--sum_15",  default="eval/out/llm_eval_summary-1.5B.json")
    ap.add_argument("--perq_7",  default="eval/out/llm_eval_per_q-7B.json")
    ap.add_argument("--sum_7",   default="eval/out/llm_eval_summary-7B.json")
    ap.add_argument("--outdir",  default="eval/report_out")
    args = ap.parse_args()

    perq_15 = Path(args.perq_15); sum_15 = Path(args.sum_15)
    perq_7  = Path(args.perq_7);  sum_7  = Path(args.sum_7)
    outdir  = Path(args.outdir)

    # Load both runs
    df15, _ = load_run("Qwen2.5-1.5B", perq_15, sum_15)
    df7,  _ = load_run("Qwen2.5-7B",   perq_7,  sum_7)

    # Combine and export tidy CSV
    df_all = pd.concat([df15, df7], ignore_index=True)
    outdir.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(outdir / "per_question_tidy.csv", index=False)

    # Generate plots + PDF
    plot_all(df_all, outdir)

    # Console summary
    for tag in sorted(df_all["model_tag"].unique()):
        sub = df_all[df_all["model_tag"] == tag]
        m = compute_metrics(sub)
        print(f"{tag} → "
              f"NumAcc={nice_number(m['num_acc'])}, "
              f"BoolAcc={nice_number(m['bool_acc'])}, "
              f"MAE={nice_number(m['mae'])} SEK, "
              f"RMSE={nice_number(m['rmse'])} SEK, "
              f"Within±1SEK={nice_number(m['within_1_sek'])}, "
              f"Latency mean={nice_number(m['lat_mean'])}s")

if __name__ == "__main__":
    main()
