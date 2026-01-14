# RadarRoster Retention Guard

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Open-source HR attrition risk scoring MVP built by **Daryoosh Dehestani** (RadarRoster).  
Designed as a **B2B accelerator**: fast to adopt, easy to extend, and practical
enough for real HR teams.

## Why this project exists

Most companies only learn about attrition risk in exit interviews. Retention
Guard helps teams **see risk early**, prioritize interventions, and build a
data-driven retention strategy without heavy setup.

## What you get

- **End-to-end pipeline** (CSV → risk scores → exportable report)
- **Baseline model** with clear, explainable signals
- **Streamlit dashboard** to demo and explore insights
- **Synthetic data generator** for safe internal demos
- **Data quality checks** and a high-risk watchlist

## Quick start (local)

1) Create a virtual environment and install:

```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

2) Run the pipeline on synthetic data:

```
python -m retention_guard.pipeline --generate-sample --output outputs/retention_scores.csv
```

3) Or use the provided sample CSV:

```
python -m retention_guard.pipeline --input data/sample_hr_data.csv --output outputs/retention_scores.csv
```

4) Launch the dashboard:

```
streamlit run app.py
```

## Data schema (minimum)

Required input columns:

- employee_id (string)
- dept (string)
- tenure_months (number)
- last_promotion_months (number)
- salary_band (string: A/B/C/D)
- manager_span (number)
- overtime_hours_month (number)
- engagement_score (0-100)
- absenteeism_days_month (number)
- peer_turnover_rate (0-1)
- performance_score (0-100)
- internal_mobility (0/1)

If your dataset lacks a target label, the pipeline generates a **synthetic
training label** to keep the demo usable. Replace this with your own exit label
when moving toward production.

## Outputs

The pipeline writes a scored CSV with:

- employee_id
- dept
- tenure_months
- last_promotion_months
- engagement_score
- overtime_hours_month
- absenteeism_days_month
- performance_score
- peer_turnover_rate
- internal_mobility
- risk_score (0-1)
- risk_band (Low/Medium/High)
- top_driver (overall model signal)

## Intended usage

Retention Guard is a **prototype** and showcase. It is perfect for:

- Internal demos with HR / People teams
- Data pipeline validation
- Early roadmap discussions with stakeholders

It is **not** a replacement for legal or compliance review. Do not use it to
automate decisions without human oversight.

## Screenshots

Add screenshots of the Streamlit dashboard here to make the project page pop.
Recommended: overview, risk distribution, and top drivers.

## Demo video

Add a short demo video or Loom link here to make the project more compelling:

- `https://your-demo-link`

## Project structure

```
.
├─ app.py                     # Streamlit demo UI
├─ data/
│  └─ sample_hr_data.csv       # Sample input file
├─ outputs/                    # Generated reports (gitignored)
└─ src/retention_guard/
   ├─ data.py                  # Data loading + synthetic generator
   ├─ model.py                 # Model pipeline + feature importance
   └─ pipeline.py              # CLI pipeline runner
```

## How to customize

Common extensions by teams who fork this repo:

- Add custom features (e.g., manager tenure, team growth)
- Swap baseline model with XGBoost or LightGBM
- Replace synthetic label with real `exit_flag`
- Ship the scores to a BI tool (Power BI / Tableau)

If you want help implementing any of these, reach out.

## Roadmap (ideas)

- Add per-employee driver explanations
- Integrate Power BI / Tableau exports
- Add role-based access and audit logs
- Optional data connector templates (Workday, SAP)

## Attribution

If you fork, reuse, or adapt this project, please keep the original attribution
to **Daryoosh Dehestani** and **RadarRoster**, and include a visible reference
in your derivative work or documentation.

## License

MIT License © 2026 Daryoosh Dehestani (RadarRoster)