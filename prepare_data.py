"""
prepare_data.py - STEP 1: clean the raw job postings.

Run:   python prepare_data.py

What it does (data cleaning / preprocessing):
  1. Removes repeated postings (the same job posted again for many cities).
  2. Cleans the job description: removes HTML tags like <br> and extra spaces.
  3. Standardises the skill tags: lowercase, merges spellings ("ReactJS", "React.js" -> React),
     drops vague tags ("senior", "development").
  4. Puts every job into a job family from its title (Software, Sales, Finance ...).
  5. Cleans the city ("Hybrid - Pune" -> Pune) and puts experience into bands (Fresher, 1-3 yrs ...).
  6. Works out a yearly salary in lakhs, for the jobs that show their salary.

The result is saved as data/jobs_clean.csv.gz. The app runs this step automatically if needed.
"""
import html
import re
import time

import numpy as np
import pandas as pd

import config

# ----------------------------------------------------------------------
# Skills
# ----------------------------------------------------------------------
def normalize_skill(tag):
    """One skill tag -> its standard form, or None if it is too vague."""
    s = str(tag).strip().lower()
    s = re.sub(r"\s+", " ", s).strip(" .-_/")
    if not s or len(s) > 40:
        return None
    s = config.SKILL_SYNONYMS.get(s, s)
    if s in config.GENERIC_TAGS:
        return None
    return s


def split_skills(tags):
    """'Python, SQL,ReactJS' -> ['python', 'sql', 'react'] (no repeats, order kept)."""
    if not isinstance(tags, str):
        return []
    out = []
    for t in tags.split(","):
        s = normalize_skill(t)
        if s and s not in out:
            out.append(s)
    return out


def skill_label(skill):
    """How a skill is shown on screen: 'power bi' -> 'Power BI', 'python' -> 'Python'."""
    if skill in config.SKILL_DISPLAY:
        return config.SKILL_DISPLAY[skill]
    words = []
    for w in skill.split(" "):
        words.append(config.SKILL_DISPLAY.get(w, w[:1].upper() + w[1:]))
    return " ".join(words)


# ----------------------------------------------------------------------
# Job families
# ----------------------------------------------------------------------
def _compile_families():
    compiled = []
    for family, words in config.JOB_FAMILIES:
        parts = []
        for w in words:
            core = re.escape(w.strip())
            right = r"(?![a-z0-9])" if (w.endswith(" ") or len(w.strip()) <= 3) else ""
            parts.append(rf"(?<![a-z0-9]){core}{right}")
        compiled.append((family, re.compile("|".join(parts))))
    return compiled


_FAMILY_PATTERNS = _compile_families()


def job_family(title):
    t = str(title).lower()
    for family, pattern in _FAMILY_PATTERNS:
        if pattern.search(t):
            return family
    return config.OTHER_FAMILY


# ----------------------------------------------------------------------
# Other columns
# ----------------------------------------------------------------------
def clean_description(text):
    t = html.unescape(str(text))
    t = re.sub(r"<[^>]+>", " ", t)            # remove HTML tags
    t = re.sub(r"\s+", " ", t).strip()
    return t


def clean_city(location):
    """'Hybrid - Pune(Hinjewadi), Mumbai' -> 'Pune' (the first city named)."""
    t = str(location).strip()
    t = re.sub(r"(?i)^(hybrid|remote)\s*-\s*", "", t)
    t = re.split(r"[,(/]", t)[0].strip()
    t = {"Bangalore": "Bengaluru", "Gurgaon": "Gurugram", "Mumbai Suburban": "Mumbai",
         "Navi Mumbai": "Navi Mumbai", "New Delhi": "Delhi", "Delhi / NCR": "Delhi"}.get(t, t)
    return t if t else "Unknown"


def work_mode(location):
    t = str(location).lower()
    if "remote" in t:
        return "Remote"
    if "hybrid" in t:
        return "Hybrid"
    return "Office"


def experience_band(min_exp):
    if pd.isna(min_exp):
        return "Not given"
    if min_exp < 1:
        return "Fresher (0 yrs)"
    if min_exp < 3:
        return "1-3 yrs"
    if min_exp < 6:
        return "3-6 yrs"
    if min_exp < 10:
        return "6-10 yrs"
    return "10+ yrs"


EXPERIENCE_ORDER = ["Fresher (0 yrs)", "1-3 yrs", "3-6 yrs", "6-10 yrs", "10+ yrs", "Not given"]


# ----------------------------------------------------------------------
# Main cleaning step
# ----------------------------------------------------------------------
def prepare(verbose=True):
    t0 = time.time()
    raw = pd.read_csv(config.RAW_FILE, encoding="utf-8", low_memory=False)
    stats = {"raw_postings": int(len(raw))}

    df = raw.drop_duplicates(subset="jobId")
    df = df.drop_duplicates(subset=["title", "companyName", "jobDescription"])
    stats["after_removing_repeats"] = int(len(df))

    df = df.assign(
        description=df["jobDescription"].map(clean_description),
        skills=df["tagsAndSkills"].map(split_skills),
    )
    df = df[df["skills"].map(len) > 0]
    stats["with_skills"] = int(len(df))

    # salary in lakhs per year (only INR jobs that show a salary)
    has_pay = (df["currency"] == "INR") & (df["minimumSalary"] > 0) & (df["maximumSalary"] > 0)
    mid = (df["minimumSalary"] + df["maximumSalary"]) / 2 / 100000
    df["salary_lpa"] = np.where(has_pay & (mid >= 0.5) & (mid <= 200), mid, np.nan)

    out = pd.DataFrame({
        "job_id": df["jobId"].astype(str),
        "title": df["title"].astype(str).str.strip(),
        "company": df["companyName"].astype(str).str.strip(),
        "family": df["title"].map(job_family),
        "city": df["location"].map(clean_city),
        "work_mode": df["location"].map(work_mode),
        "experience": df["experience"].fillna("Not given"),
        "exp_band": df["minimumExperience"].map(experience_band),
        "min_exp": df["minimumExperience"],
        "salary_lpa": df["salary_lpa"].round(2),
        "rating": df["AggregateRating"],
        "skills": df["skills"].map("|".join),
        "description": df["description"],
    })
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.CLEAN_FILE, index=False, encoding="utf-8", compression="gzip")

    stats.update({
        "clean_postings": int(len(out)),
        "unique_skills": int(out["skills"].str.split("|").explode().nunique()),
        "salary_shown_pct": round(float(out["salary_lpa"].notna().mean() * 100), 1),
        "other_family_pct": round(float((out["family"] == config.OTHER_FAMILY).mean() * 100), 1),
    })
    if verbose:
        print(f"Raw postings:                 {stats['raw_postings']:,}")
        print(f"After removing repeats:       {stats['after_removing_repeats']:,}")
        print(f"Clean postings with skills:   {stats['clean_postings']:,}")
        print(f"Different skills after merge: {stats['unique_skills']:,}")
        print(f"Jobs that show their salary:  {stats['salary_shown_pct']}%")
        print(f"Jobs not matched to a family: {stats['other_family_pct']}%")
        print(f"Saved {config.CLEAN_FILE.name} in {time.time() - t0:.0f} s")
    return out, stats


def load_clean(rebuild=False):
    """Read the cleaned file (making it first if it does not exist yet)."""
    if rebuild or not config.CLEAN_FILE.exists() or \
            config.CLEAN_FILE.stat().st_mtime < config.RAW_FILE.stat().st_mtime:
        prepare(verbose=False)
    df = pd.read_csv(config.CLEAN_FILE, encoding="utf-8", dtype={"job_id": str}, low_memory=False)
    df["skills"] = df["skills"].fillna("").str.split("|")
    df["description"] = df["description"].fillna("")
    return df


if __name__ == "__main__":
    prepare(verbose=True)
