# llm_eval_easy.py — minimal LLM evaluation on your receipts DB (no parser needed)
# - Computes ground truth with SQL
# - Asks the LLM the same questions
# - Scores numeric & YES/NO accuracy
# - Saves CSV/JSON and plots to eval/out/
#
# Usage:
#   python llm_eval_easy.py --model Qwen/Qwen2.5-1.5B-Instruct --limit 20
# Optional:
#   python llm_eval_easy.py --model Qwen/Qwen2.5-0.5B-Instruct --limit 50

import os, json, time, argparse, math, sqlite3
from statistics import mean
import matplotlib.pyplot as plt

OUT_DIR = "eval/out"
os.makedirs(OUT_DIR, exist_ok=True)

def _percentiles(xs, ps=(50,95)):
    if not xs: return {p: None for p in ps}
    xs = sorted(xs)
    out = {}
    for p in ps:
        k = (len(xs)-1) * (p/100)
        f = math.floor(k); c = math.ceil(k)
        out[p] = xs[int(k)] if f==c else xs[f] + (xs[c]-xs[f])*(k-f)
    return out

def _save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _save_csv(path, rows, header=None):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        for r in rows:
            if isinstance(r, dict):
                if header: w.writerow([r.get(h, "") for h in header])
                else: w.writerow(list(r.values()))
            else:
                w.writerow(r)

def build_compact_context(limit=20, max_chars=8000):
    con = sqlite3.connect("receipts.db"); con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("""
        SELECT r.receipt_id AS rid,
               s.name       AS store_name,
               r.date       AS date,
               r.total      AS total
        FROM receipts r
        LEFT JOIN stores s ON r.store_id = s.store_id
        ORDER BY r.receipt_id DESC
        LIMIT ?
    """, (limit,))
    recs = [dict(row) for row in cur.fetchall()]
    con.close()
    s = json.dumps(recs, ensure_ascii=False)
    return s[:max_chars] if len(s) > max_chars else s, recs

def ground_truths(recs):
    # recs: list of {rid, store_name, date, total}
    if not recs: 
        return [], {}
    totals = [float(r["total"] or 0) for r in recs]
    overall = round(sum(totals), 2)
    avg = round(overall / len(totals), 2)
    mx = max(totals); mn = min(totals)
    max_rec = recs[totals.index(mx)]
    min_rec = recs[totals.index(mn)]

    # top store by spend inside these recs
    spend_by_store = {}
    for r in recs:
        spend_by_store.setdefault(r["store_name"], 0.0)
        spend_by_store[r["store_name"]] += float(r["total"] or 0)
    top_store = max(spend_by_store.items(), key=lambda x: x[1])[0]
    top_store_sum = round(spend_by_store[top_store], 2)

    # a simple month question: use YYYY-MM of the most recent
    recent_date = (recs[0]["date"] or "")[:7]
    month_sum = round(sum(float(r["total"] or 0) for r in recs if (r["date"] or "").startswith(recent_date)), 2)

    # one YES/NO threshold question (above/below)
    threshold = round(overall * 0.9, 2)  # 90% of overall
    yesno_gt = "YES" if overall >= threshold else "NO"

    qa = [
        {"type":"num", "q":"What is my total spend overall?", "expect": overall, "tol": 1.0},
        {"type":"num", "q":"What is my average spend per receipt?", "expect": avg, "tol": 1.0},
        {"type":"num", "q":f"What is my total spend at {top_store}?", "expect": top_store_sum, "tol": 1.0},
        {"type":"num", "q":f"For month {recent_date}, what is my total spend?", "expect": month_sum, "tol": 1.0},
        {"type":"num", "q":"What is the maximum receipt total I have?", "expect": round(mx,2), "tol": 1.0},
        {"type":"num", "q":"What is the minimum receipt total I have?", "expect": round(mn,2), "tol": 1.0},
        {"type":"bool","q":f"Is my overall spend greater than or equal to {threshold}?", "expect": yesno_gt},
        {"type":"num", "q":"How many receipts are there in the context?", "expect": float(len(recs)), "tol": 0.01},
    ]
    meta = {
        "top_store": top_store,
        "recent_month": recent_date,
        "threshold": threshold
    }
    return qa, meta

def load_model(model_id):
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", dtype="auto", low_cpu_mem_usage=True
    )
    gen = pipeline(
        "text-generation", model=mdl, tokenizer=tok,
        max_new_tokens=64, do_sample=False, temperature=0.0, top_p=1.0
    )
    return tok, gen

def last_number(text):
    import re
    nums = re.findall(r"-?\d+(?:[.,]\d+)?", text or "")
    if not nums: return None
    return float(nums[-1].replace(",", "."))

def eval_llm(model_id="Qwen/Qwen2.5-1.5B-Instruct", limit=20):
    if not os.path.exists("receipts.db"):
        print("receipts.db not found"); return

    # pull compact rows up-front
    ctx_json, recs = build_compact_context(limit=limit, max_chars=20000)
    if not recs:
        print("No receipts in DB."); return

    # compute ground truth & metadata once
    qa, meta = ground_truths(recs)

    # precompute helpful minimal views (lists of numbers / per-store)
    totals = [float(r["total"] or 0) for r in recs]
    store_totals = {}
    for r in recs:
        store_totals.setdefault(r["store_name"], []).append(float(r["total"] or 0))
    recent_month = meta["recent_month"]
    month_totals = [float(r["total"] or 0) for r in recs if (r["date"] or "").startswith(recent_month)]

    SYSTEM = "Answer strictly with the required output. No words. No SEK. No extra text."

    tok, gen = load_model(model_id)

    def ask_number(numbers_list, question_text):
        # numbers_list is a short Python-like list of floats
        user = f"""
You are given a list of numbers:
{numbers_list}

Return ONLY the final number for: {question_text}
- Use standard arithmetic.
- No words, no units, no steps. Only the final number.
""".strip()
        messages = [{"role":"system","content":SYSTEM},{"role":"user","content":user}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        t0=time.time(); out = gen(prompt)[0]["generated_text"]; dt=time.time()-t0
        ans = out[len(prompt):].strip()
        return ans, dt

    def ask_yesno(numbers_list, question_text):
        user = f"""
You are given a list of numbers:
{numbers_list}

Answer ONLY YES or NO (uppercase) for: {question_text}
""".strip()
        messages = [{"role":"system","content":SYSTEM},{"role":"user","content":user}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        t0=time.time(); out = gen(prompt)[0]["generated_text"]; dt=time.time()-t0
        ans = out[len(prompt):].strip()
        return ans, dt

    rows=[]; times=[]
    correct_num=correct_bool=total_num=total_bool=0

    for ex in qa:
        if ex["type"]=="num":
            # choose the smallest sensible context for each question
            q = ex["q"]
            if "overall" in q.lower() or "average" in q.lower() or "maximum" in q.lower() or "minimum" in q.lower() or "how many receipts" in q.lower():
                nums = totals[:]  # all totals
            elif "month" in q.lower():
                nums = month_totals[:]  # totals for recent month only
            elif "at " in q.lower() and meta["top_store"] in q:
                nums = store_totals[meta["top_store"]][:]
            else:
                nums = totals[:]

            # tweak question phrasing to be unambiguous
            if "overall" in q.lower():
                q_clean = "Sum of all numbers"
            elif "average" in q.lower():
                q_clean = "Average of all numbers"
            elif "maximum" in q.lower():
                q_clean = "Maximum of all numbers"
            elif "minimum" in q.lower():
                q_clean = "Minimum of all numbers"
            elif "how many receipts" in q.lower():
                q_clean = "Count of numbers (return an integer)"
            elif "month" in q.lower():
                q_clean = "Sum of these numbers"
            elif "at " in q.lower():
                q_clean = "Sum of these numbers"
            else:
                q_clean = q

            ans, dt = ask_number(nums, q_clean)
            pred = last_number(ans)
            ok = (pred is not None and abs(pred - ex["expect"]) <= ex["tol"])
            correct_num += 1 if ok else 0; total_num += 1
            times.append(dt)
            rows.append({"type":"num","q":ex["q"],"ans":ans,"pred":pred,"expect":ex["expect"],"ok":ok,"latency_s":round(dt,2)})
            print(f"Q: {ex['q']}\nA: {ans}\n→ pred={pred} vs {ex['expect']}  ok={ok}  t={dt:.2f}s\n")

        else:  # YES/NO
            # threshold question uses overall list
            q_clean = ex["q"].replace(str(meta["threshold"]), f"{meta['threshold']}")
            ans, dt = ask_yesno(totals, q_clean)
            pred_bool = "YES" if "YES" in ans.upper() and "NO" not in ans.upper() else ("NO" if "NO" in ans.upper() else None)
            ok = (pred_bool == ex["expect"])
            correct_bool += 1 if ok else 0; total_bool += 1
            times.append(dt)
            rows.append({"type":"bool","q":ex["q"],"ans":ans,"pred":pred_bool,"expect":ex["expect"],"ok":ok,"latency_s":round(dt,2)})
            print(f"Q: {ex['q']}\nA: {ans}\n→ pred={pred_bool} vs {ex['expect']}  ok={ok}  t={dt:.2f}s\n")

    num_acc  = (correct_num/total_num) if total_num else None
    bool_acc = (correct_bool/total_bool) if total_bool else None
    lat_mean = round(mean(times),2) if times else None
    pct = _percentiles(times, (50,95))

    summary = {
        "model": model_id,
        "n_numeric": total_num, "numeric_accuracy": round(num_acc,3) if num_acc is not None else None,
        "n_bool": total_bool,   "bool_accuracy": round(bool_acc,3) if bool_acc is not None else None,
        "latency_mean_s": lat_mean,
        "latency_p50_s": round(pct[50],2) if pct[50] is not None else None,
        "latency_p95_s": round(pct[95],2) if pct[95] is not None else None,
        "limit_context_receipts": len(recs),
        "meta": meta
    }

    _save_json(os.path.join(OUT_DIR, "llm_eval_per_q.json"), rows)
    _save_json(os.path.join(OUT_DIR, "llm_eval_summary.json"), summary)

    # plots (same as before)
    plt.figure(); xs=[]; ys=[]
    if num_acc is not None: xs.append("Numeric"); ys.append(num_acc)
    if bool_acc is not None: xs.append("YES/NO"); ys.append(bool_acc)
    if ys:
        plt.bar(xs, ys); 
        for i,v in enumerate(ys): plt.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        plt.ylim(0,1); plt.ylabel("accuracy"); plt.title("LLM Accuracy")
        plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "llm_accuracy_bar.png")); plt.close()
    if times:
        plt.figure(); plt.hist(times, bins=10); plt.xlabel("seconds"); plt.ylabel("count"); plt.title("LLM latency distribution")
        plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "llm_latency_hist.png")); plt.close()

    print("[LLM eval] Summary:", summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--limit", type=int, default=20, help="how many latest receipts to include in context")
    args = ap.parse_args()
    eval_llm(model_id=args.model, limit=args.limit)

if __name__ == "__main__":
    main()
