"""
config.py - all project settings in one place.
"""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

RAW_FILE = DATA_DIR / "naukri_jobs_2025.csv.gz"     # the job postings (comes with the project)
CLEAN_FILE = DATA_DIR / "jobs_clean.csv.gz"         # made by prepare_data.py
REPORT_FILE = RESULTS_DIR / "report.json"           # made by run_analysis.py

# ----------------------------------------------------------------------
# Mining settings
# ----------------------------------------------------------------------
MIN_SUPPORT = 0.002        # a skill set must appear in at least 0.2% of jobs (about 170) to count as "frequent"
MIN_CONFIDENCE = 0.30      # keep rules that are right at least 30% of the time
N_CLUSTERS = 16            # number of job groups the clustering looks for
MIN_JOBS_FOR_SALARY = 100  # a skill needs this many jobs with a disclosed salary to show its salary
RANDOM_STATE = 42

# ----------------------------------------------------------------------
# Job families: a job title goes to the FIRST family whose words it contains.
# (Order matters: "account executive" is a sales job, so Sales comes before Finance.)
# ----------------------------------------------------------------------
JOB_FAMILIES = [
    ("Data & AI", ["data scien", "data analy", "data engineer", "machine learning", "ml engineer",
                   "ai/ml", "ai / ml", "ai engineer", "artificial intelligence", "analytics",
                   "business intelligence", "power bi", "bi developer", "data architect",
                   "big data", "deep learning", "nlp", "generative ai", "gen ai", "mis ", "data platform",
                   "decision sci"]),
    ("SAP & ERP", ["sap ", "sap-", "sap/", " sap", "oracle", "erp", "salesforce", "servicenow",
                   "workday", "dynamics 365"]),
    ("Software Development", ["developer", "software", "full stack", "fullstack", "frontend",
                              "front end", "backend", "back end", "programmer", "sde", "devops",
                              "application lead", "application designer", "application architect",
                              "java", "python", ".net", "react", "angular", "node", "php",
                              "android", "ios ", "flutter", "cloud engineer", "site reliability",
                              "qa ", "quality analyst", "test engineer", "tester", "testing",
                              "automation engineer", "technical lead", "tech lead", "architect", "test lead",
                              "scrum master", "product owner",
                              "web designer", "salesforce developer"]),
    ("IT Support & Networks", ["support engineer", "technical support", "desktop support",
                               "it support", "network", "system admin", "system engineer",
                               "infrastructure", "cyber", "security analyst", "soc analyst",
                               "service desk", "helpdesk", "help desk", "it executive", "tech support",
                               "security engineer"]),
    ("Sales & Business Development", ["sales", "business development", "bde", "bdm",
                                      "relationship manager", "relationship officer",
                                      "account executive", "account manager", "key account",
                                      "telecaller", "tele caller", "telesales", "pre sales",
                                      "presales", "business associate", "channel partner",
                                      "insurance advisor", "loan", "telecall", "client acquisition"]),
    ("Marketing & Content", ["marketing", "seo", "social media", "content", "brand", "digital",
                             "copywriter", "public relations", "media"]),
    ("Customer Support & BPO", ["customer", "bpo", "voice", "call center", "call centre",
                                "chat process", "trust & safety", "process associate",
                                "process executive", "non voice", "contact center", "chat support"]),
    ("HR & Recruitment", ["hr ", "hr-", "human resource", "recruit", "talent acquisition",
                          "payroll", "hrbp", "hr generalist", "hr executive", "hr manager"]),
    ("Finance & Accounts", ["account", "finance", "financial", "chartered", "ca ", "audit", "tax",
                            "gst", "credit", "treasury", "banking", "invest", "actuar", "billing",
                            "collection", "underwrit", "fp&a", "costing"]),
    ("Design & Creative", ["designer", "graphic", "ui/ux", "ux", "ui ", "video editor", "animator",
                           "illustrator", "creative", "3d", "photographer"]),
    ("Production, Supply Chain & Core Engg", [
        "production", "mechanical", "manufactur", "plant", "quality", "maintenance",
        "supply chain", "procurement", "purchase", "logistic", "warehouse", "store",
        "inventory", "cnc", "vmc", "machine operator", "process engineer", "industrial",
        "civil", "site engineer", "electrical", "electronics", "instrumentation", "hvac",
        "automobile", "automotive", "service engineer", "field engineer", "design engineer",
        "r&d", "piping", "welding", "tool", "operations executive", "planning engineer",
        "sourcing", "vendor", "dispatch", "fleet", "safety officer", "embedded", "electrician",
        "technician", "supervisor", "tender"]),
    ("Healthcare & Pharma", ["nurse", "doctor", "medical", "pharma", "clinical", "health",
                             "hospital", "physio", "dental", "lab technician", "radiolog",
                             "pharmacist", "patient"]),
    ("Teaching & Education", ["teacher", "faculty", "professor", "trainer", "tutor", "teaching",
                              "lecturer", "education", "counsellor", "counselor", "academic"]),
    ("Management & Operations", ["manager", "operations", "project", "program", "consultant",
                                 "analyst", "administrat", "admin", "executive assistant",
                                 "coordinator", "head", "director", "lead", "officer", "front office", "receptionist",
                                 "back office", "data entry", "office assistant", "computer operator",
                                 "personal assistant", "secretary", "office boy", "front desk",
                                 "management trainee"]),
]
OTHER_FAMILY = "Other"

# ----------------------------------------------------------------------
# Skill tags: merge different spellings of the same skill
# ----------------------------------------------------------------------
SKILL_SYNONYMS = {
    "ms excel": "excel", "advanced excel": "excel", "microsoft excel": "excel", "ms-excel": "excel",
    "excel sheet": "excel", "excel macros": "excel vba", "vba": "excel vba",
    "ms office": "ms office", "microsoft office": "ms office", "ms office suite": "ms office",
    "communication skills": "communication", "good communication": "communication",
    "communication skill": "communication", "verbal communication": "communication",
    "excellent communication skills": "communication", "english communication": "communication",
    "written communication": "communication",
    "react.js": "react", "reactjs": "react", "react js": "react",
    "node.js": "node.js", "nodejs": "node.js", "node js": "node.js", "node": "node.js",
    "angularjs": "angular", "angular js": "angular",
    "javascript": "javascript", "java script": "javascript", "js": "javascript",
    "typescript": "typescript",
    "power bi": "power bi", "powerbi": "power bi", "microsoft power bi": "power bi",
    "microsoft azure": "azure", "azure cloud": "azure",
    "amazon web services": "aws", "aws cloud": "aws",
    "google cloud platform": "gcp", "google cloud": "gcp",
    "ml": "machine learning", "machine learning algorithms": "machine learning",
    "ai": "artificial intelligence", "ai/ml": "artificial intelligence",
    "genai": "generative ai", "gen ai": "generative ai",
    "ci/cd": "ci/cd", "continuous integration": "ci/cd", "cicd": "ci/cd",
    "postgres": "postgresql", "my sql": "mysql", "ms sql": "sql server", "mssql": "sql server",
    "microsoft sql server": "sql server", "sql queries": "sql", "pl/sql": "plsql",
    "team handling": "team management", "team leading": "team management",
    "team lead": "team management", "people management": "team management",
    "customer care": "customer service", "customer support": "customer service",
    "auto cad": "autocad", "solid works": "solidworks",
    "lean manufacturing": "lean", "six sigma green belt": "six sigma",
    "b2b": "b2b sales", "boot": "spring boot", "spring boot": "spring boot", "springboot": "spring boot",
    "rest api": "rest", "restful": "rest", "rest apis": "rest", "web api": "rest",
    "c sharp": "c#", "dot net": ".net", "asp.net": ".net", "net": ".net",
    "software testing": "software testing", "manual testing": "manual testing",
    "selenium webdriver": "selenium", "hindi speaking": "hindi", "english speaking": "english",
    "fluent english": "english", "tally erp": "tally", "tally prime": "tally",
}

# Tags that are not real skills (too vague to be useful)
GENERIC_TAGS = {
    "senior", "development", "data", "process", "engineering", "design", "coding",
    "relationship", "it", "technology", "technical", "software", "fresher", "freshers",
    "graduate", "computer science", "management", "business", "consulting", "operations",
    "architecture", "analytical", "training", "administration", "documentation", "support",
    "application", "applications", "system", "systems", "skills", "work from home", "any graduate",
    "jobs", "job", "hiring", "immediate joiner", "immediate joiners", "service", "services", "ms",
    "good", "excellent", "knowledge", "experience",
}

# Pretty names for skills whose plain title-case looks wrong
SKILL_DISPLAY = {
    "sql": "SQL", "aws": "AWS", "gcp": "GCP", "sap": "SAP", "css": "CSS", "html": "HTML",
    "javascript": "JavaScript", "typescript": "TypeScript", "node.js": "Node.js", "c#": "C#",
    "c++": "C++", ".net": ".NET", "power bi": "Power BI", "ci/cd": "CI/CD", "erp": "ERP",
    "bpo": "BPO", "autocad": "AutoCAD", "solidworks": "SolidWorks", "mysql": "MySQL",
    "postgresql": "PostgreSQL", "plsql": "PL/SQL", "sql server": "SQL Server", "rest": "REST",
    "devops": "DevOps", "ms office": "MS Office", "excel vba": "Excel VBA", "ui": "UI", "ux": "UX",
    "seo": "SEO", "hr": "HR", "b2b sales": "B2B Sales", "tds": "TDS", "gst": "GST", "api": "API",
    "sap hana": "SAP HANA", "ios": "iOS", "php": "PHP", "nlp": "NLP", "qa": "QA", "crm": "CRM",
    "kpi": "KPI", "mis": "MIS", "iot": "IoT", "ml": "ML", "llm": "LLM", "hvac": "HVAC",
    "plc": "PLC", "cnc": "CNC", "gd&t": "GD&T", "fmea": "FMEA", "catia": "CATIA", "iso": "ISO",
    "tqm": "TQM", "hrsd": "HRSD", "abap": "ABAP", "fico": "FICO", "mm": "MM", "sd": "SD", "pp": "PP",
    "wm": "WM", "qm": "QM", "us": "US", "cad": "CAD", "uk": "UK", "hcm": "HCM", "bw": "BW", "s/4hana": "S/4HANA", "hana": "HANA", "etl": "ETL",
    "gcp": "GCP", "aws": "AWS", "ppc": "PPC", "sem": "SEM", "smo": "SMO", "kyc": "KYC", "aml": "AML",
}
