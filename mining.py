"""
mining.py - the four web-mining techniques used in this project.

1. Skill demand   : count how many job ads ask for each skill (overall, by family, city, experience)
                    and the typical salary of jobs that ask for it.
2. Apriori        : association rule mining - which skills are asked for TOGETHER
                    ("if a job asks for Power BI, it usually also asks for SQL").
3. Clustering     : K-Means groups jobs with similar skills into job families, without being told
                    the job types (after LSA squeezes the skills into 50 topics). Checked with the
                    silhouette score and purity.
4. Resume matcher : TF-IDF + cosine similarity finds the job ads closest to your resume and lists
                    the skills you are missing (skill gap). Checked with precision@10.
"""
import re
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

import config
from prepare_data import EXPERIENCE_ORDER, normalize_skill, skill_label


# ======================================================================
# 1. Skill demand and salary
# ======================================================================
def skill_counts(df, top=None):
    """How many jobs ask for each skill, and what % of all jobs that is."""
    counts = Counter(s for skills in df["skills"] for s in skills)
    out = pd.DataFrame(counts.most_common(top), columns=["skill", "jobs"])
    out["pct_of_jobs"] = out["jobs"] / max(len(df), 1) * 100
    out["label"] = out["skill"].map(skill_label)
    return out


def demand_by_group(df, group_col, skill, groups=None):
    """% of jobs in each group (city, family, experience band) that ask for one skill."""
    has = df["skills"].map(lambda s: skill in s)
    table = df.assign(has=has).groupby(group_col)["has"].agg(["mean", "sum", "size"]).reset_index()
    table.columns = [group_col, "pct_of_jobs", "jobs_with_skill", "jobs_in_group"]
    table["pct_of_jobs"] *= 100
    if groups is not None:
        table = table.set_index(group_col).reindex(groups).dropna().reset_index()
    return table


def salary_by_skill(df, min_jobs=config.MIN_JOBS_FOR_SALARY):
    """Median salary (lakhs per year) of jobs asking for each skill, compared with all jobs."""
    paid = df[df["salary_lpa"].notna()]
    overall = float(paid["salary_lpa"].median())
    rows = paid[["skills", "salary_lpa", "min_exp"]].explode("skills")
    g = rows.groupby("skills").agg(jobs=("salary_lpa", "size"),
                                   median_lpa=("salary_lpa", "median"),
                                   median_exp=("min_exp", "median"))
    g = g[g["jobs"] >= min_jobs].reset_index().rename(columns={"skills": "skill"})
    g["vs_all_jobs_pct"] = (g["median_lpa"] / overall - 1) * 100
    g["label"] = g["skill"].map(skill_label)
    return g.sort_values("median_lpa", ascending=False), overall


# ======================================================================
# 2. Apriori: frequent skill sets and association rules
# ======================================================================
def apriori(transactions, min_support=config.MIN_SUPPORT, max_len=3):
    """
    Classic Apriori. A skill set is 'frequent' if it appears in at least
    min_support (e.g. 0.5%) of all jobs. Bigger sets are only checked if all
    their smaller parts are frequent (the Apriori rule), which saves a lot of work.
    Returns {frozenset(skills): number_of_jobs}.
    """
    n = len(transactions)
    min_count = max(2, int(np.ceil(min_support * n)))
    baskets = [set(t) for t in transactions if t]

    counts = Counter(s for b in baskets for s in b)
    frequent = {frozenset([s]): c for s, c in counts.items() if c >= min_count}
    current = {next(iter(k)) for k in frequent}

    for size in range(2, max_len + 1):
        if not current:
            break
        prev_sets = {k for k in frequent if len(k) == size - 1}
        counter = Counter()
        for b in baskets:
            items = sorted(b & current)
            if len(items) < size:
                continue
            for combo in combinations(items, size):
                if size > 2 and any(frozenset(sub) not in prev_sets for sub in combinations(combo, size - 1)):
                    continue
                counter[combo] += 1
        new = {frozenset(k): c for k, c in counter.items() if c >= min_count}
        frequent.update(new)
        current = {s for k in new for s in k}
    return frequent, n


def association_rules(frequent, n, min_confidence=config.MIN_CONFIDENCE):
    """
    Turn frequent skill sets into rules  A -> B.
      support    = share of all jobs that ask for A and B together
      confidence = of the jobs that ask for A, the share that also ask for B
      lift       = confidence / share of all jobs that ask for B
                   (lift > 1: A and B really go together; lift = 1: no link)
    """
    rows = []
    for itemset, count in frequent.items():
        if len(itemset) < 2:
            continue
        for k in range(1, len(itemset)):
            for left in combinations(sorted(itemset), k):
                left = frozenset(left)
                right = itemset - left
                if len(right) != 1:
                    continue
                conf = count / frequent[left]
                if conf < min_confidence:
                    continue
                lift = conf / (frequent[right] / n)
                rows.append({
                    "if_job_asks_for": " + ".join(skill_label(s) for s in sorted(left)),
                    "it_also_asks_for": skill_label(next(iter(right))),
                    "jobs": count,
                    "support_pct": count / n * 100,
                    "confidence_pct": conf * 100,
                    "lift": lift,
                    "size": len(itemset),
                })
    cols = ["if_job_asks_for", "it_also_asks_for", "jobs", "support_pct", "confidence_pct", "lift", "size"]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["lift", "confidence_pct"], ascending=False).reset_index(drop=True)


def skills_asked_with(df, skill, top=10):
    """For one skill: the skills most often in the same job ads (confidence and lift)."""
    has = df["skills"].map(lambda s: skill in s)
    n_skill = int(has.sum())
    if n_skill == 0:
        return pd.DataFrame(columns=["label", "jobs", "confidence_pct", "lift"])
    all_counts = Counter(s for skills in df["skills"] for s in skills)
    together = Counter(s for skills in df.loc[has, "skills"] for s in skills if s != skill)
    rows = [{"label": skill_label(s), "jobs": c, "confidence_pct": c / n_skill * 100,
             "lift": (c / n_skill) / (all_counts[s] / len(df))}
            for s, c in together.most_common(top)]
    return pd.DataFrame(rows)


# ======================================================================
# 3. Clustering jobs into families
# ======================================================================
def _skills_as_tokens(skills):
    return skills


def skill_matrix(df, min_df=20):
    """Each job becomes a row of numbers: one column per skill (TF-IDF weights)."""
    vec = TfidfVectorizer(analyzer=_skills_as_tokens, min_df=min_df, sublinear_tf=True)
    X = vec.fit_transform(df["skills"])
    return X, vec


def _reduce(X):
    """
    Squeeze ~4,000 skill columns into 50 'topics' (LSA / Truncated SVD). Skills that often
    appear together end up in the same topic, so K-Means finds cleaner groups.
    """
    return make_pipeline(TruncatedSVD(n_components=50, random_state=config.RANDOM_STATE),
                         Normalizer()).fit_transform(X)


def cluster_jobs(df, k=config.N_CLUSTERS):
    """
    K-Means clustering of jobs by their skills.
    Returns: a cluster number per job (-1 = too few common skills to cluster),
             a summary table per cluster, and what is needed to score the clusters.
    """
    X, vec = skill_matrix(df)
    usable = np.diff(X.indptr) >= 2                      # jobs with at least 2 common skills
    Z = _reduce(X[usable])
    km = KMeans(n_clusters=k, n_init=4, random_state=config.RANDOM_STATE)
    lab_usable = km.fit_predict(Z)
    labels = np.full(len(df), -1)
    labels[usable] = lab_usable

    names = vec.get_feature_names_out()
    Xu = X[usable]
    rows = []
    for c in range(k):
        member_rows = lab_usable == c
        weights = np.asarray(Xu[member_rows].mean(axis=0)).ravel()
        top = np.argsort(weights)[::-1][:6]
        members = df[labels == c]
        fam = members.loc[members["family"] != config.OTHER_FAMILY, "family"].value_counts(normalize=True)
        rows.append({
            "cluster": c,
            "name": " / ".join(skill_label(names[i]) for i in top[:3]),
            "jobs": int(len(members)),
            "top_skills": ", ".join(skill_label(names[i]) for i in top),
            "main_family": fam.index[0] if len(fam) else config.OTHER_FAMILY,
            "main_family_pct": float(fam.iloc[0] * 100) if len(fam) else 0.0,
            "median_salary_lpa": float(members["salary_lpa"].median()) if members["salary_lpa"].notna().any() else np.nan,
        })
    summary = pd.DataFrame(rows).sort_values("jobs", ascending=False).reset_index(drop=True)
    return labels, summary, (Z, lab_usable, usable)


def cluster_quality(df, labels, parts, sample=4000):
    """
    Silhouette score (-1 to +1): how well each job fits its own cluster compared with the
    next-closest cluster. Above 0 = the clusters are separated; higher is better.
    Purity: share of jobs whose cluster's most common job family is their own family.
    """
    Z, lab_usable, usable = parts
    rng = np.random.RandomState(config.RANDOM_STATE)
    idx = rng.choice(len(lab_usable), size=min(sample, len(lab_usable)), replace=False)
    sil = float(silhouette_score(Z[idx], lab_usable[idx], metric="cosine"))
    known = (df["family"].values != config.OTHER_FAMILY) & (labels >= 0)
    tab = pd.crosstab(labels[known], df["family"].values[known])
    purity = float(tab.max(axis=1).sum() / tab.values.sum())
    return {"silhouette": sil, "purity_pct": purity * 100,
            "clustered_pct": float(usable.mean() * 100)}


def silhouette_for_k(df, ks=(8, 10, 12, 14, 16, 18, 20)):
    """Try several numbers of clusters and report the silhouette score of each (the 'elbow' check)."""
    X, _ = skill_matrix(df)
    usable = np.diff(X.indptr) >= 2
    Z = _reduce(X[usable])
    idx = np.random.RandomState(0).choice(Z.shape[0], size=min(4000, Z.shape[0]), replace=False)
    out = []
    for k in ks:
        lab = KMeans(n_clusters=k, n_init=3, random_state=config.RANDOM_STATE).fit_predict(Z)
        out.append({"k": k, "silhouette": float(silhouette_score(Z[idx], lab[idx], metric="cosine"))})
    return pd.DataFrame(out)


# ======================================================================
# 4. Resume matcher (TF-IDF + cosine similarity)
# ======================================================================
def job_text(row_title, row_skills, row_desc):
    skills = ", ".join(row_skills)
    return f"{row_title}. {skills}. {skills}. {row_desc[:400]}"   # skills counted twice: they matter most


class JobMatcher:
    """Finds the job ads most similar to a piece of text (a resume or a list of skills)."""

    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        texts = [job_text(t, s, d) for t, s, d in zip(self.df["title"], self.df["skills"], self.df["description"])]
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3,
                                   max_features=60000, sublinear_tf=True)
        self.X = self.vec.fit_transform(texts)
        counts = Counter(s for skills in self.df["skills"] for s in skills)
        self.known_skills = {s for s, c in counts.items() if c >= 30 and len(s) >= 2}
        self._skill_patterns = {s: re.compile(rf"(?<![a-z0-9+#.]){re.escape(s)}(?![a-z0-9+#])")
                                for s in self.known_skills}
        self._synonyms = {k: v for k, v in config.SKILL_SYNONYMS.items() if v in self.known_skills}

    def skills_in_text(self, text):
        """Find known skills inside free text (e.g. a pasted resume)."""
        t = " " + re.sub(r"\s+", " ", str(text).lower()) + " "
        found = {s for s, p in self._skill_patterns.items() if p.search(t)}
        for alias, s in self._synonyms.items():
            if re.search(rf"(?<![a-z0-9+#.]){re.escape(alias)}(?![a-z0-9+#])", t):
                found.add(s)
        # keep the longest names only ("sap" is dropped if "sap hana" was found)
        return sorted(s for s in found if not any(s != o and f" {s} " in f" {o} " for o in found))

    def match(self, text, top=10, mask=None, exclude_index=None):
        q = self.vec.transform([text])
        sims = linear_kernel(q, self.X).ravel()
        if mask is not None:
            sims = np.where(mask, sims, -1)
        if exclude_index is not None:
            sims[exclude_index] = -1
        order = np.argsort(sims)[::-1][:top]
        order = [i for i in order if sims[i] > 0]
        result = self.df.iloc[order].copy()
        result["match_pct"] = sims[order] * 100
        return result

    def skill_gap(self, text, matches, top=12):
        """Skills that the best-matching jobs ask for, but the resume does not mention."""
        have = set(self.skills_in_text(text))
        need = Counter(s for skills in matches["skills"] for s in skills if s not in have)
        return pd.DataFrame([{"skill": skill_label(s), "in_matched_jobs": c} for s, c in need.most_common(top)])


def evaluate_matcher(matcher, n_queries=1000, k=10):
    """
    precision@10: take a real job ad, use ONLY its skill list as a 'resume', and look at the
    10 most similar OTHER jobs. Precision = share of those 10 in the same job family.
    Compared with picking 10 jobs at random (the baseline).
    """
    df = matcher.df
    known = np.where(df["family"].values != config.OTHER_FAMILY)[0]
    rng = np.random.RandomState(config.RANDOM_STATE)
    queries = rng.choice(known, size=min(n_queries, len(known)), replace=False)
    family_share = df["family"].value_counts(normalize=True)
    rows = []
    for i in queries:
        q = ", ".join(df.at[i, "skills"])
        top = matcher.match(q, top=k, exclude_index=i)
        same = (top["family"].values == df.at[i, "family"]).mean() if len(top) else 0.0
        rows.append({"family": df.at[i, "family"], "precision": same,
                     "baseline": family_share[df.at[i, "family"]]})
    res = pd.DataFrame(rows)
    by_family = res.groupby("family").agg(queries=("precision", "size"),
                                          precision_at_10_pct=("precision", "mean"),
                                          random_baseline_pct=("baseline", "mean")).reset_index()
    by_family[["precision_at_10_pct", "random_baseline_pct"]] *= 100
    overall = {"queries": int(len(res)), "precision_at_10_pct": float(res["precision"].mean() * 100),
               "random_baseline_pct": float(res["baseline"].mean() * 100)}
    return overall, by_family.sort_values("precision_at_10_pct", ascending=False)


# ======================================================================
# Small helpers for the app
# ======================================================================
def filter_jobs(df, family=None, city=None, exp_band=None, work_mode=None):
    m = pd.Series(True, index=df.index)
    if family and family != "All":
        m &= df["family"] == family
    if city and city != "All":
        m &= df["city"] == city
    if exp_band and exp_band != "All":
        m &= df["exp_band"] == exp_band
    if work_mode and work_mode != "All":
        m &= df["work_mode"] == work_mode
    return df[m]


def find_skill(query, df):
    """Turn what the user typed ('ReactJS') into the standard skill name ('react')."""
    s = normalize_skill(query)
    if s is None:
        return None
    counts = Counter(x for skills in df["skills"] for x in skills)
    if s in counts:
        return s
    close = [k for k in counts if k.startswith(s)]
    return max(close, key=counts.get) if close else None


EXPERIENCE_BANDS = [b for b in EXPERIENCE_ORDER if b != "Not given"]
