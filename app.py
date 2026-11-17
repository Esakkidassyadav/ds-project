"""
app.py - STEP 3: the dashboard (a small website that runs on your laptop).

Run:   streamlit run app.py      (opens http://localhost:8501)

Tabs
  Overview                 the main numbers and findings in plain words
  Top skills               most-asked skills, with filters for job family, city, experience
  Skill explorer           everything about one skill: demand, salary, cities, partner skills
  Skills that go together  association rules found with Apriori
  Salaries                 which skills come with higher pay (and why experience matters)
  Job groups               K-Means clustering of jobs by their skills
  Resume matcher           paste your skills, get matching jobs and your skill gap
  Evaluation               how good the matcher and the clustering are
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import mining
from prepare_data import load_clean, skill_label

st.set_page_config(page_title="Job Market Skill Analyzer", page_icon=":bar_chart:", layout="wide")

BLUE, ORANGE, QUIET, INK = "#2a78d6", "#eb6834", "#b4b2ac", "#52514e"
SEQUENTIAL = [[0, "#eaf2fd"], [1, "#104281"]]
BANDS = mining.EXPERIENCE_BANDS


def salary_text(values):
    return values.map(lambda x: f"{x:.1f}" if pd.notna(x) else "not shown")


def fmt(x, spec=".1f", na="n/a"):
    return na if x is None or pd.isna(x) else format(x, spec)


def hbar(labels, values, colour=BLUE, height=None, xtitle="", hover="%{y}: %{x:.1f}", text=None, key=None):
    """Horizontal bar chart, biggest bar on top."""
    labels, values = list(labels)[::-1], list(values)[::-1]
    colours = colour[::-1] if isinstance(colour, list) else colour
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colours, marker_line_width=0,
                           text=(list(text)[::-1] if text is not None else None), textposition="outside",
                           hovertemplate=hover + "<extra></extra>"))
    fig.update_layout(height=height or max(260, 26 * len(labels) + 60), margin=dict(l=10, r=40, t=10, b=10),
                      xaxis_title=xtitle, bargap=0.3)
    st.plotly_chart(fig, key=key)


# ----------------------------------------------------------------------
# Loading (cached, so it only happens once)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Cleaning about 98,000 job ads (first start only, about 15 seconds)...")
def get_jobs():
    return load_clean()


@st.cache_resource(show_spinner="Building the resume matcher (first time only, about 15 seconds)...")
def get_matcher():
    return mining.JobMatcher(get_jobs())


@st.cache_data(show_spinner="Finding skills that go together (Apriori)...")
def get_rules(family, min_support, min_confidence):
    df = get_jobs()
    if family != "All":
        df = df[df["family"] == family]
    frequent, n = mining.apriori(df["skills"].tolist(), min_support=min_support)
    return mining.association_rules(frequent, n, min_confidence=min_confidence), len(frequent), n


@st.cache_data(show_spinner="Grouping the jobs with K-Means...")
def get_clusters(k):
    df = get_jobs()
    labels, summary, parts = mining.cluster_jobs(df, k=k)
    return labels, summary, mining.cluster_quality(df, labels, parts)


@st.cache_data(show_spinner=False)
def get_salary():
    return mining.salary_by_skill(get_jobs())


@st.cache_data(show_spinner=False)
def get_counts():
    return mining.skill_counts(get_jobs())


def load_report():
    if config.REPORT_FILE.exists():
        try:
            return json.loads(config.REPORT_FILE.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


jobs = get_jobs()
counts_all = get_counts()
families = list(jobs["family"].value_counts().index)
top_cities = list(jobs["city"].value_counts().head(15).index)

# ----------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    st.caption("Used by the Overview, Top skills and Skill explorer tabs.")
    f_family = st.selectbox("Job family", ["All"] + families)
    f_city = st.selectbox("City", ["All"] + top_cities)
    f_exp = st.selectbox("Experience asked for", ["All"] + BANDS)
    f_mode = st.selectbox("Work mode", ["All", "Office", "Hybrid", "Remote"])
    view = mining.filter_jobs(jobs, f_family, f_city, f_exp, f_mode)
    st.metric("Job ads selected", f"{len(view):,}")
    st.divider()
    st.caption("Data: 97,929 Naukri.com job ads, mostly posted in September 2025 "
               "(Kaggle dataset by Shivam Shrivastava). For education only.")

st.title("Job Market Skill Analyzer")
st.caption("Which skills do Indian employers ask for? Web mining on real Naukri.com job ads: skill demand, "
           "skills that go together, job groups, and a resume matcher.")

tabs = st.tabs(["Overview", "Top skills", "Skill explorer", "Skills that go together", "Salaries",
                "Job groups", "Resume matcher", "Evaluation"])

# ---------------- Overview ----------------
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Job ads (after cleaning)", f"{len(view):,}")
    c[1].metric("Companies", f"{view['company'].nunique():,}")
    c[2].metric("Different skills", f"{view['skills'].explode().nunique():,}")
    c[3].metric("Cities", f"{view['city'].nunique():,}")
    c[4].metric("Ads that show salary", f"{view['salary_lpa'].notna().mean() * 100:.0f}%")

    if view.empty:
        st.warning("No job ads match these filters.")
    else:
        top = mining.skill_counts(view, top=1).iloc[0]
        fresher = mining.skill_counts(view[view["exp_band"] == "Fresher (0 yrs)"], top=3)
        sal, overall = get_salary()
        report = load_report()
        st.subheader("What we found")
        lines = [f"- The most-asked skill in the selected ads is **{top['label']}**, in "
                 f"**{top['pct_of_jobs']:.1f}%** of them."]
        if len(fresher):
            lines.append("- For **freshers**, the top skills are " +
                         ", ".join(f"**{a}**" for a in fresher["label"]) + ".")
        lines.append(f"- Across all ads that show pay, the typical (median) salary is **{overall:.1f} lakh a year**. "
                     f"Ads asking for **{sal.iloc[0]['label']}** pay a median of **{sal.iloc[0]['median_lpa']:.1f} lakh**, "
                     f"but they also ask for about {sal.iloc[0]['median_exp']:.0f} years of experience.")
        if report:
            lines.append(f"- The resume matcher puts the right kind of job in its top 10 "
                         f"**{report['matcher']['precision_at_10_pct']:.0f}%** of the time, against "
                         f"**{report['matcher']['random_baseline_pct']:.0f}%** for random picks.")
        st.markdown("\n".join(lines))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top 15 skills in the selected ads**")
            t15 = mining.skill_counts(view, top=15)
            hbar(t15["label"], t15["pct_of_jobs"], xtitle="% of job ads", hover="%{y}: in %{x:.1f}% of ads",
                 key="ov_skills")
        with c2:
            st.markdown("**Job ads per job family**")
            fam = view["family"].value_counts()
            hbar(fam.index, fam.values, colour=INK, xtitle="Job ads", hover="%{y}: %{x:,} ads", key="ov_fam")
        st.caption("Job family is worked out from the job title (for example, titles with 'developer' go to "
                   "Software Development). Titles that match no family are 'Other'.")

# ---------------- Top skills ----------------
with tabs[1]:
    if view.empty:
        st.warning("No job ads match these filters.")
    else:
        n = st.slider("How many skills to show", 10, 40, 20, step=5)
        tn = mining.skill_counts(view, top=n)
        st.markdown(f"**Top {n} skills** in {len(view):,} selected job ads")
        hbar(tn["label"], tn["pct_of_jobs"], xtitle="% of job ads",
             hover="%{y}: in %{x:.1f}% of ads", key="top_n")

        st.subheader("How demand changes with experience")
        top12 = mining.skill_counts(view, top=12)["skill"].tolist()
        grid = []
        for s in top12:
            g = mining.demand_by_group(view, "exp_band", s, groups=BANDS)
            grid.append(g.set_index("exp_band")["pct_of_jobs"].reindex(BANDS).values)
        z = np.array(grid, dtype=float)
        heat = go.Figure(go.Heatmap(z=z, x=BANDS, y=[skill_label(s) for s in top12], colorscale=SEQUENTIAL,
                                    text=np.round(z, 1), texttemplate="%{text}%", showscale=False,
                                    hovertemplate="%{y}, %{x}: %{z:.1f}% of ads<extra></extra>"))
        heat.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(heat, key="exp_heat")
        st.caption("Each cell: % of job ads at that experience level that ask for the skill. Darker = more in demand.")

# ---------------- Skill explorer ----------------
with tabs[2]:
    options = counts_all[counts_all["jobs"] >= 20]
    labels = dict(zip(options["skill"], options["label"]))
    default = list(labels).index("python") if "python" in labels else 0
    skill = st.selectbox("Pick a skill (type to search)", list(labels), index=default, format_func=labels.get)
    has = jobs["skills"].map(lambda s: skill in s)
    sj = jobs[has]
    sal, overall = get_salary()
    srow = sal[sal["skill"] == skill]
    paid = sj["salary_lpa"].dropna()
    rank = int(counts_all.index[counts_all["skill"] == skill][0]) + 1
    c = st.columns(4)
    c[0].metric("Job ads asking for it", f"{len(sj):,}")
    c[1].metric("Share of all ads", f"{len(sj) / len(jobs) * 100:.2f}%")
    c[2].metric("Rank among all skills", f"#{rank}")
    c[3].metric("Median salary", f"{paid.median():.1f} lakh" if len(paid) >= 30 else "too few ads",
                help=f"Median of the {len(paid)} ads that show pay. All ads: {overall:.1f} lakh a year.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Which job families ask for it** (% of the skill's ads)")
        fam = sj["family"].value_counts(normalize=True).head(8) * 100
        hbar(fam.index, fam.values, xtitle="% of this skill's ads", hover="%{y}: %{x:.1f}%", key="sx_fam")
    with c2:
        st.markdown("**Demand by city** (% of each city's ads)")
        city = mining.demand_by_group(jobs, "city", skill, groups=top_cities[:10]).sort_values(
            "pct_of_jobs", ascending=False)
        hbar(city["city"], city["pct_of_jobs"], xtitle="% of the city's ads", hover="%{y}: %{x:.2f}%",
             key="sx_city")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Demand by experience level** (% of ads at each level)")
        exp = mining.demand_by_group(jobs, "exp_band", skill, groups=BANDS)
        fig = go.Figure(go.Bar(x=exp["exp_band"], y=exp["pct_of_jobs"], marker_color=BLUE, marker_line_width=0,
                               hovertemplate="%{x}: %{y:.2f}% of ads<extra></extra>"))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="% of ads", bargap=0.35)
        st.plotly_chart(fig, key="sx_exp")
    with c2:
        st.markdown("**Skills most often asked together with it**")
        partners = mining.skills_asked_with(jobs, skill, top=10)
        st.dataframe(partners.rename(columns={"label": "Skill", "jobs": "Ads with both",
                                              "confidence_pct": "% of its ads", "lift": "Lift"}),
                     hide_index=True, column_config={
                         "% of its ads": st.column_config.NumberColumn(format="%.1f"),
                         "Lift": st.column_config.NumberColumn(format="%.1f")})
        st.caption("Lift above 1 means the two skills appear together more often than chance.")

    st.markdown("**Some job ads that ask for it**")
    sample = sj.sample(min(12, len(sj)), random_state=1)
    st.dataframe(pd.DataFrame({
        "Job title": sample["title"], "Company": sample["company"], "City": sample["city"],
        "Experience": sample["experience"], "Salary (lakh/yr)": salary_text(sample["salary_lpa"]),
        "All skills": sample["skills"].map(lambda s: ", ".join(skill_label(x) for x in s))}),
        hide_index=True)

# ---------------- Skills that go together ----------------
with tabs[3]:
    st.markdown(
        "**Association rule mining (Apriori).** Each job ad is treated like a shopping basket of skills. "
        "Apriori finds skills that are often 'bought together', the same way a shop finds that bread and "
        "butter are bought together.")
    c1, c2, c3 = st.columns(3)
    r_family = c1.selectbox("Job family", ["All"] + families, key="r_family")
    r_support = c2.select_slider("Minimum support (% of ads)", options=[0.1, 0.2, 0.3, 0.5, 1.0, 2.0], value=0.2,
                                 help="A skill set must appear in at least this share of ads.")
    r_conf = c3.select_slider("Minimum confidence (%)", options=[20, 30, 40, 50, 60, 70, 80], value=30)
    rules, n_sets, n_ads = get_rules(r_family, r_support / 100, r_conf / 100)
    st.caption(f"{n_ads:,} job ads, {n_sets:,} frequent skill sets, {len(rules):,} rules.")
    if rules.empty:
        st.info("No rules with these settings. Lower the minimum support or confidence.")
    else:
        best = rules.sort_values("lift", ascending=False).head(15)
        st.markdown("**15 strongest rules (highest lift)**")
        hbar([f"{a} → {b}" for a, b in zip(best["if_job_asks_for"], best["it_also_asks_for"])], best["lift"],
             xtitle="Lift", hover="%{y}<br>lift %{x:.1f}", key="rules_lift")
        ex = best.iloc[0]
        st.markdown(
            f"**How to read the top rule:** {ex['if_job_asks_for']} → {ex['it_also_asks_for']}. "
            f"Both appear together in {int(ex['jobs']):,} ads (**support** {ex['support_pct']:.2f}%). "
            f"Of the ads asking for {ex['if_job_asks_for']}, {ex['confidence_pct']:.0f}% also ask for "
            f"{ex['it_also_asks_for']} (**confidence**). That is {ex['lift']:.0f} times more often than chance "
            f"(**lift**).")
        st.dataframe(rules.head(300).rename(columns={
            "if_job_asks_for": "If an ad asks for", "it_also_asks_for": "it also asks for", "jobs": "Ads",
            "support_pct": "Support %", "confidence_pct": "Confidence %", "lift": "Lift", "size": "Skills in rule"}),
            hide_index=True, column_config={
                "Support %": st.column_config.NumberColumn(format="%.2f"),
                "Confidence %": st.column_config.NumberColumn(format="%.0f"),
                "Lift": st.column_config.NumberColumn(format="%.1f")})

# ---------------- Salaries ----------------
with tabs[4]:
    sal, overall = get_salary()
    st.markdown(f"Only **{jobs['salary_lpa'].notna().mean() * 100:.0f}%** of ads show a salary, so these numbers "
                f"use those ads only. The median salary of all of them is **{overall:.1f} lakh a year**.")
    top_sal = sal.head(20)
    st.markdown(f"**20 best-paid skills** (median salary, skills with at least {config.MIN_JOBS_FOR_SALARY} paid ads)")
    hbar(top_sal["label"], top_sal["median_lpa"], xtitle="Median salary (lakh per year)",
         hover="%{y}: %{x:.1f} lakh", key="sal_top")

    st.subheader("Experience explains a lot of the difference")
    fig = go.Figure(go.Scatter(
        x=sal["median_exp"], y=sal["median_lpa"], mode="markers",
        marker=dict(size=np.clip(np.sqrt(sal["jobs"]) * 0.9, 6, 30), color=BLUE, opacity=0.6,
                    line=dict(width=1, color="white")),
        text=sal["label"], customdata=sal["jobs"],
        hovertemplate="%{text}<br>%{y:.1f} lakh, typical experience %{x:.0f} yrs<br>%{customdata} paid ads<extra></extra>"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Typical years of experience asked", yaxis_title="Median salary (lakh per year)")
    st.plotly_chart(fig, key="sal_scatter")
    st.caption("Each dot is a skill (bigger = more ads). Skills on the right ask for more experience and pay more, "
               "so a high salary does not only come from the skill itself.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Median salary by experience level**")
        by_exp = jobs.dropna(subset=["salary_lpa"]).groupby("exp_band")["salary_lpa"].median().reindex(BANDS)
        hbar(by_exp.index, by_exp.values, colour=INK, xtitle="Lakh per year", hover="%{y}: %{x:.1f} lakh",
             key="sal_exp")
    with c2:
        st.markdown("**Median salary by job family**")
        by_fam = jobs.dropna(subset=["salary_lpa"]).groupby("family")["salary_lpa"].median().sort_values(ascending=False)
        hbar(by_fam.index, by_fam.values, colour=INK, xtitle="Lakh per year", hover="%{y}: %{x:.1f} lakh",
             key="sal_fam")

# ---------------- Job groups ----------------
with tabs[5]:
    st.markdown(
        "**Clustering (K-Means).** The computer groups job ads with similar skills, without being told the job "
        "titles. Then we check if the groups make sense by comparing them with the job families from the titles.")
    k = st.slider("Number of groups (k)", 8, 20, config.N_CLUSTERS, step=2)
    labels_k, summary, quality = get_clusters(k)
    c = st.columns(3)
    c[0].metric("Silhouette score", f"{quality['silhouette']:.2f}",
                help="-1 to +1. Above 0 means the groups are separated; higher is better.")
    c[1].metric("Purity", f"{quality['purity_pct']:.0f}%",
                help="How often a job's group is mostly made of its own job family.")
    c[2].metric("Jobs grouped", f"{quality['clustered_pct']:.0f}%",
                help="Jobs with fewer than 2 common skills are left out of the grouping.")
    st.dataframe(summary.rename(columns={
        "name": "Group (top 3 skills)", "jobs": "Job ads", "top_skills": "Top skills",
        "main_family": "Mostly which family", "main_family_pct": "% of that family",
        "median_salary_lpa": "Median salary (lakh)"}).drop(columns=["cluster"]),
        hide_index=True, column_config={
            "% of that family": st.column_config.NumberColumn(format="%.0f"),
            "Median salary (lakh)": st.column_config.NumberColumn(format="%.1f")})

    st.markdown("**Which job families end up in which group** (% of each group)")
    known = (jobs["family"] != config.OTHER_FAMILY).values & (labels_k >= 0)
    tab = pd.crosstab(labels_k[known], jobs["family"].values[known], normalize="index") * 100
    names = summary.set_index("cluster")["name"]
    tab = tab.loc[summary["cluster"]]
    heat = go.Figure(go.Heatmap(z=tab.values, x=list(tab.columns), y=[names[i] for i in tab.index],
                                colorscale=SEQUENTIAL, showscale=False,
                                hovertemplate="%{y}<br>%{x}: %{z:.0f}%<extra></extra>"))
    heat.update_layout(height=40 * len(tab) + 160, margin=dict(l=10, r=10, t=10, b=10),
                       yaxis=dict(autorange="reversed"), xaxis=dict(tickangle=-35))
    st.plotly_chart(heat, key="cluster_heat")
    st.caption("A dark cell means that group is mostly that job family. Groups that spread across many "
               "families share general skills such as communication or team management.")

# ---------------- Resume matcher ----------------
with tabs[6]:
    examples = {
        "Data analyst (fresher)": "Fresher data analyst. Skills: Python, SQL, Excel, Power BI, data analysis, "
                                  "statistics, dashboards.",
        "Production / supply chain engineer (fresher)": "Production engineering graduate. Skills: AutoCAD, "
            "SolidWorks, lean manufacturing, six sigma, quality control, SAP MM, supply chain, MS Excel.",
        "Java backend developer": "Backend developer with 3 years in Java, Spring Boot, microservices, REST APIs, "
                                  "MySQL, Docker, AWS and Git.",
        "Sales executive": "Sales executive with field sales, B2B sales, lead generation, negotiation, CRM and "
                           "good communication in English and Hindi.",
        "HR recruiter": "HR recruiter: talent acquisition, screening, sourcing, onboarding, joining formalities, "
                        "MS Excel.",
    }
    pick = st.selectbox("Start from an example, or type your own below", list(examples))
    text = st.text_area("Your resume or skills", value=examples[pick], height=110)
    c1, c2, c3 = st.columns(3)
    m_city = c1.selectbox("City", ["All"] + top_cities, key="m_city")
    m_exp = c2.slider("Maximum experience the job may ask for (years)", 0, 15, 15)
    m_family = c3.selectbox("Job family", ["All"] + families, key="m_family")
    matcher = get_matcher()
    if text.strip():
        mdf = matcher.df
        mask = np.ones(len(mdf), dtype=bool)
        if m_city != "All":
            mask &= (mdf["city"] == m_city).values
        if m_family != "All":
            mask &= (mdf["family"] == m_family).values
        mask &= (mdf["min_exp"].fillna(0) <= m_exp).values
        found = matcher.skills_in_text(text)
        st.markdown("**Skills found in your text:** " +
                    (", ".join(f"`{skill_label(s)}`" for s in found) if found else "none of the common skills"))
        best = matcher.match(text, top=10, mask=mask)
        if best.empty:
            st.warning("No matching jobs with these filters. Try 'All' cities or more experience.")
        else:
            have = set(found)
            st.dataframe(pd.DataFrame({
                "Match %": best["match_pct"].round(0),
                "Job title": best["title"], "Company": best["company"], "City": best["city"],
                "Experience": best["experience"], "Salary (lakh/yr)": salary_text(best["salary_lpa"]),
                "Skills you have": best["skills"].map(lambda s: ", ".join(skill_label(x) for x in s if x in have)),
                "Skills you lack": best["skills"].map(lambda s: ", ".join(skill_label(x) for x in s if x not in have)),
            }), hide_index=True, column_config={"Match %": st.column_config.NumberColumn(format="%.0f")})
            st.caption("Match % is the cosine similarity between your text and the job ad, after both are turned "
                       "into TF-IDF numbers. 100% would mean the same words with the same weights.")
            gap = matcher.skill_gap(text, matcher.match(text, top=30, mask=mask))
            if not gap.empty:
                st.markdown("**Your skill gap:** skills the 30 best-matching jobs ask for that your text does not mention")
                hbar(gap["skill"], gap["in_matched_jobs"], colour=ORANGE, xtitle="Number of the 30 jobs",
                     hover="%{y}: in %{x} of the 30 jobs", key="gap")

# ---------------- Evaluation ----------------
with tabs[7]:
    report = load_report()
    if report is None:
        st.info("Run `python run_analysis.py` once (about 2 minutes) to create the evaluation results, "
                "then reload this page.")
    else:
        d = report["data"]
        st.subheader("1. Data cleaning")
        st.dataframe(pd.DataFrame([
            ("Job ads in the raw file", d["raw_postings"]),
            ("After removing repeated ads", d["after_removing_repeats"]),
            ("With at least one usable skill", d["clean_postings"]),
            ("Different skills after merging spellings", d["unique_skills"])],
            columns=["Step", "Count"]), hide_index=True)

        mt = report["matcher"]
        st.subheader("2. Resume matcher: precision@10")
        st.markdown(
            f"We took **{mt['queries']:,} real job ads**, used only their skill list as a 'resume', and checked "
            f"the 10 most similar other ads. **{mt['precision_at_10_pct']:.1f}%** of them were in the same job "
            f"family. Picking 10 ads at random would give only **{mt['random_baseline_pct']:.1f}%**.")
        fam = pd.DataFrame(mt["by_family"]).sort_values("precision_at_10_pct")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=fam["family"], x=fam["precision_at_10_pct"], orientation="h", name="Matcher",
                             marker_color=BLUE, marker_line_width=0,
                             hovertemplate="%{y}: %{x:.0f}%<extra>Matcher</extra>"))
        fig.add_trace(go.Bar(y=fam["family"], x=fam["random_baseline_pct"], orientation="h", name="Random picks",
                             marker_color=QUIET, marker_line_width=0,
                             hovertemplate="%{y}: %{x:.0f}%<extra>Random</extra>"))
        fig.update_layout(barmode="group", height=520, margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_title="Precision@10 (%)", bargap=0.25,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig, key="eval_prec")

        cl = report["clustering"]
        st.subheader("3. Clustering: silhouette score for different k")
        ks = pd.DataFrame(cl["silhouette_by_k"])
        fig = go.Figure(go.Scatter(x=ks["k"], y=ks["silhouette"], mode="lines+markers",
                                   line=dict(color=BLUE, width=2), marker=dict(size=9),
                                   hovertemplate="k = %{x}: %{y:.3f}<extra></extra>"))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Number of groups (k)",
                          yaxis_title="Silhouette score")
        st.plotly_chart(fig, key="eval_k")
        st.markdown(f"With k = {cl['k']}: silhouette **{cl['silhouette']:.3f}**, purity **{cl['purity_pct']:.1f}%**. "
                    "More groups usually score a little higher, so we kept 16 groups that are still easy to name.")

        ap = report["apriori"]
        st.subheader("4. Apriori")
        st.markdown(f"Minimum support {ap['min_support_pct']:.1f}% and confidence {ap['min_confidence_pct']:.0f}% "
                    f"gave **{ap['frequent_sets']:,} frequent skill sets** and **{ap['rules']:,} rules**. A rule is "
                    "useful when its lift is well above 1.")
        st.caption(f"Results created {report['created']} by run_analysis.py.")
