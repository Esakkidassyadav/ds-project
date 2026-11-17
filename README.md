# Job Market Skill Analyzer: Mining In-Demand Skills from Indian Job Postings

Web Data Mining mini-project. It studies 97,929 real job ads from Naukri.com (mostly posted in
September 2025) and finds:

1. **Top skills**: which skills are asked for most, by job family, city, experience and work mode.
2. **Skills that go together**: association rules found with the Apriori algorithm.
3. **Job groups**: K-Means clustering of jobs by their skills.
4. **Salaries**: which skills come with higher pay, and how experience affects it.
5. **Resume matcher**: TF-IDF + cosine similarity finds the jobs closest to your skills and shows your skill gap.

## Files

| File | What it does |
|---|---|
| `config.py` | All settings: job-family keywords, skill spellings to merge, mining settings |
| `prepare_data.py` | **Step 1.** Cleans the data: removes repeated ads, cleans skills, cities, experience, salary |
| `mining.py` | The four mining techniques: skill counts, Apriori, K-Means, resume matcher (+ their tests) |
| `run_analysis.py` | **Step 2.** Runs everything, prints the results and saves `results/report.json` |
| `app.py` | **Step 3.** The dashboard (8 tabs) |
| `data/naukri_jobs_2025.csv.gz` | The job ads (compressed CSV, 15 MB) |
| `results/report.json` | Results of my test run (made again when you run step 2) |
| `requirements.txt` | Python libraries needed |
| `run_project.bat` | One-click start on Windows |

## How to run (Windows)

You need Python 3.10 or newer ([python.org](https://www.python.org/downloads/), tick "Add python.exe to PATH").

**Option A:** unzip the folder and double-click `run_project.bat`.

**Option B:** open Command Prompt in this folder and run:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python prepare_data.py
python run_analysis.py
streamlit run app.py
```

On Mac or Linux use `source venv/bin/activate` instead of the second line.
No internet is needed after the libraries are installed: the data comes with the project.

## Data source and ethics

- **Data:** "Indian Job Market Dataset 2025-2026" by Shivam Shrivastava on Kaggle
  ([link](https://www.kaggle.com/datasets/shivamshrivastava21/indian-job-market-dataset-2025-2026)),
  97,929 job ads collected from Naukri.com. I downloaded it from a public GitHub copy
  ([Annusharma30/Job-Market-Intelligence-and-Skill-Analysis](https://github.com/Annusharma30/Job-Market-Intelligence-and-Skill-Analysis)).
  The Kaggle page does not state a licence, so it is used here for education only, with credit.
- Job ads are public, and the data has no personal information about job seekers.
- I did not scrape Naukri.com myself: its search pages are protected against bots, so using an
  existing public dataset is the ethical route.

This project is for learning. It is not career or salary advice.

## Troubleshooting

| Problem | Fix |
|---|---|
| `'streamlit' is not recognized` | Activate the environment first: `venv\Scripts\activate` |
| `python` not found | Install Python from python.org with "Add to PATH" ticked, or type `py` instead of `python` |
| PowerShell says running scripts is disabled | Use Command Prompt (cmd) instead of PowerShell |
| The Evaluation tab says to run the analysis | Run `python run_analysis.py` once, then reload the page |
| Port 8501 is already in use | `streamlit run app.py --server.port 8502` |
