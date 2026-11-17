"""
run_analysis.py - STEP 2: run every mining technique and print the results.

Run:   python run_analysis.py

Prints the main findings in the terminal and saves them in results/report.json.
The dashboard (app.py) reads that file for the Evaluation tab.
Takes about 2 minutes (the resume-matcher test is the slow part).
"""
import json
import time
from datetime import datetime

import pandas as pd

import config
import mining
from prepare_data import load_clean, prepare


def line(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    t0 = time.time()
    _, clean_stats = prepare(verbose=True)
    df = load_clean()
    report = {"created": datetime.now().strftime("%Y-%m-%d %H:%M"), "data": clean_stats}

    # 1. skill demand
    top = mining.skill_counts(df, top=20)
    line("Top 20 skills in all job ads (% of jobs)")
    for _, r in top.iterrows():
        print(f"  {r['label']:28s} {r['jobs']:6,d} jobs  {r['pct_of_jobs']:5.1f}%")
    report["top_skills"] = top[["label", "jobs", "pct_of_jobs"]].round(2).to_dict("records")

    fresher = mining.skill_counts(df[df["exp_band"] == "Fresher (0 yrs)"], top=10)
    line("Top 10 skills in fresher jobs")
    print("  " + ", ".join(f"{a} ({b:.1f}%)" for a, b in zip(fresher["label"], fresher["pct_of_jobs"])))
    report["top_fresher_skills"] = fresher[["label", "jobs", "pct_of_jobs"]].round(2).to_dict("records")

    by_family = {}
    for fam in df["family"].value_counts().index:
        sub = mining.skill_counts(df[df["family"] == fam], top=8)
        by_family[fam] = sub[["label", "pct_of_jobs"]].round(1).to_dict("records")
    report["top_skills_by_family"] = by_family

    sal, overall = mining.salary_by_skill(df)
    line(f"Best-paid skills (median of jobs that show salary; all jobs: {overall:.1f} lakh/yr)")
    for _, r in sal.head(10).iterrows():
        print(f"  {r['label']:28s} {r['median_lpa']:5.1f} lakh/yr   (typical experience {r['median_exp']:.0f} yrs, {r['jobs']} jobs)")
    report["salary"] = {"overall_median_lpa": overall,
                        "top": sal.head(15)[["label", "jobs", "median_lpa", "median_exp"]].round(1).to_dict("records")}

    # 2. association rules
    frequent, n = mining.apriori(df["skills"].tolist())
    rules = mining.association_rules(frequent, n)
    strong = rules[rules["confidence_pct"] >= 50].sort_values("lift", ascending=False)
    line(f"Skills that go together (Apriori: {len(frequent)} frequent skill sets, {len(rules)} rules)")
    for _, r in strong.head(12).iterrows():
        print(f"  {r['if_job_asks_for']:34s} -> {r['it_also_asks_for']:24s} "
              f"confidence {r['confidence_pct']:4.0f}%  lift {r['lift']:5.1f}")
    report["apriori"] = {"min_support_pct": config.MIN_SUPPORT * 100, "min_confidence_pct": config.MIN_CONFIDENCE * 100,
                         "frequent_sets": len(frequent), "rules": int(len(rules)),
                         "examples": strong.head(15).round(2).to_dict("records")}

    # 3. clustering
    labels, summary, parts = mining.cluster_jobs(df)
    quality = mining.cluster_quality(df, labels, parts)
    ks = mining.silhouette_for_k(df)
    line(f"Job groups found by K-Means (k = {config.N_CLUSTERS})")
    for _, r in summary.iterrows():
        print(f"  {r['jobs']:6,d} jobs  {r['name']:58s} mostly {r['main_family']}")
    print(f"  Silhouette score: {quality['silhouette']:.3f}   Purity: {quality['purity_pct']:.1f}%")
    print("  Silhouette for other k: " + ", ".join(f"k={k}: {s:.3f}" for k, s in zip(ks["k"], ks["silhouette"])))
    report["clustering"] = {"k": config.N_CLUSTERS, **{a: round(b, 4) for a, b in quality.items()},
                            "silhouette_by_k": ks.round(4).to_dict("records"),
                            "clusters": summary.round(1).to_dict("records")}

    # 4. resume matcher
    matcher = mining.JobMatcher(df)
    overall_eval, fam_eval = mining.evaluate_matcher(matcher, n_queries=1000)
    line("Resume matcher test (precision@10 = share of the 10 matches in the right job family)")
    print(f"  Precision@10: {overall_eval['precision_at_10_pct']:.1f}%   "
          f"Random guessing: {overall_eval['random_baseline_pct']:.1f}%   ({overall_eval['queries']} test jobs)")
    report["matcher"] = {**{a: (round(b, 2) if isinstance(b, float) else b) for a, b in overall_eval.items()},
                         "by_family": fam_eval.round(1).to_dict("records")}

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_FILE.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved {config.REPORT_FILE} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
