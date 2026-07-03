# Operation HealthGuard — Student Workspace

> **Your role.** Chief AI Risk Officer at Klariova Health. Pre-launch GRC review of HealthGuard, a diabetes-risk prediction model.
>
> **Target time.** approximately 5–6 hours. 
**Submission:** the populated `governance_portfolio.xlsx`, your completed `audit_notebook.ipynb`, and your `lab/` folder.

---

## Quick start

```bash
pip install -r requirements.txt
pytest lab/tests/ -v          # 34 tests will fail at first; that's expected
jupyter notebook audit_notebook.ipynb
```


## What's in this folder

```
starter/
├── README.md                   ← you are here
├── requirements.txt            Python dependencies
├── audit_notebook.ipynb        Orchestrator notebook — runs your lab/ code, writes to Excel
├── governance_portfolio.xlsx   THE single deliverable Excel — 8 tabs to populate
├── data/                       Inputs (DO NOT MODIFY)
│   ├── case_study_brief.md       Read this first — your scenario
│   ├── healthguard_model.joblib  Pre-trained scikit-learn pipeline
│   ├── train.csv  /  test.csv    Synthetic patient datasets
│   └── data_flow_diagram.png     Pre-drawn DFD — annotate this in D2
├── lab/                        Code modules YOU implement
│   ├── fairness_metrics.py        TODO scaffolding
│   ├── audit.py                   TODO scaffolding
│   ├── shap_analysis.py           TODO scaffolding
│   ├── kri.py                     TODO scaffolding
│   └── tests/                     34 unit tests (read-only)
└── results/                    Where any chart PNGs you export should go
```

## How to do the project

1. **Read `data/case_study_brief.md`.** Refer back to it constantly — it is your scenario.
2. **Implement the four `lab/` modules.** Each TODO has a "why this matters" docstring; the unit tests tell you when you have it right. Run after each:
   ```
   pytest lab/tests/test_fairness_metrics.py -v
   ```
   All 34 tests must pass before you submit.
3. **Run `audit_notebook.ipynb`.** It calls your `lab/` code, runs the audit, and writes results to the Excel portfolio (Tab 04 D4 + Tab 07 D7 status).
4. **Fill in the [HAND] tabs in `governance_portfolio.xlsx`:** D1, D2, D3, the rest of D5, D6, D8.
5. **Sanity check:** open `../rubric.md` and walk down each criterion — anything you missed?

## The two parallel tracks

### Track A — `lab/` code (~2 hours)

| Module | What you implement | Why it matters |
|---|---|---|
| `fairness_metrics.py` | FNR, FPR, selection rate, demographic-parity diff, equalized-odds diff — from confusion-matrix primitives | Tests whether you understand what each metric *is*, not just how to call a library |
| `audit.py` | Subgroup grouping, intersectional pivot, statistical-power-aware most-harmed finder, proxy-pathway test | Tests the small-n trap and proxy-bias intuition |
| `shap_analysis.py` | SHAP explainer setup, programmatic `detect_proxy_features` | Tests whether you can encode "what a proxy feature looks like" in code |
| `kri.py` | Three KRIs translated from D7 written definitions | Tests whether you can move from a written KRI to a reproducible one |

### Track B — Excel portfolio (~3.5 hours)

8 tabs. Some tabs are filled by the notebook from your code (`[CODE]`); some you write by hand (`[HAND]`).

| Tab | Source | What you produce |
|---|---|---|
| 00 Cover | — | (no work needed) |
| 01 D1 EU AI Act | [HAND] | Risk-tier classification + all 11 Article 9–15 obligation rows |
| 02 D2 DFD Compliance | [HAND] | Annotate `data/data_flow_diagram.png`, build the obligation table |
| 03 D3 ATLAS | [HAND] | 3 well-analyzed TTPs |
| 04 D4 Technical Audit | [CODE] | Notebook writes performance + subgroup tables; you write the Proxy-Bias Hypothesis |
| 05 D5 Risk Register | [HAND] | You author all rows; draw 3-4 directly from your D4 findings |
| 06 D6 Model Card | [HAND] | You transcribe Performance & Subgroup from D4; you write the rest |
| 07 D7 KRI Dashboard | [CODE] | Notebook calls your `kri.py` to populate Status |
| 08 D8 Launch Decision | [HAND] | 1-page exec brief + detailed justification + RACI sign-off |

## Tips

- **Run unit tests after every TODO.** Implement one function, run the tests, fix, move on. Do NOT implement all five fairness functions then run tests — the test-driven loop is the pedagogy.
- **The proxy-bias hypothesis is the spine.** If your D4 surfaces it, it should reappear naturally in D5 (Fairness risk), D6 (Explainability), D7 (subgroup-FNR KRI), D8 (top residual risk). If it doesn't, your tabs aren't connected to your code findings.
- **Section A of D8 is what a CEO will read.** No jargon. If a non-technical reader needs Section B to understand Section A, you've failed the stand-alone test.
- **Don't modify the `data/` folder.** Every input is read-only.

## Submission checklist

- [ ] `pytest lab/tests/ -v` reports 34 passed
- [ ] `audit_notebook.ipynb` runs end-to-end without error
- [ ] All 8 tabs in `governance_portfolio.xlsx` are populated
- [ ] Tab 04 (D4) shows the proxy-hypothesis verdict
- [ ] Tab 05 (D5) has ≥10 risks across ≥4 source outputs
- [ ] Tab 07 (D7) shows three KRIs with Status populated by your code (not typed)
- [ ] Tab 08 (D8) Section A reads stand-alone for a non-technical CEO
- [ ] The `lab/` folder you submit contains your implementations, not the original TODOs
