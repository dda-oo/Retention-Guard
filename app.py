"""Streamlit dashboard for Retention Guard."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from retention_guard.pipeline import run_pipeline  # noqa: E402


st.set_page_config(page_title="RadarRoster Retention Guard", layout="wide")

st.title("RadarRoster Retention Guard")
st.caption("AI-powered attrition risk scoring for HR teams.")
st.info(
    "This demo uses a baseline model and synthetic labels. "
    "Use it to explore data quality and risk patterns, not to automate decisions."
)

st.markdown(
    """
**What this app does**
- Scores employees with a **risk score (0–1)** based on HR signals.
- Groups scores into **Low / Medium / High** risk bands.
- Highlights **top drivers** that influence risk in the overall model.

**Who it's for**
- HR and People Analytics teams exploring retention patterns.
- Managers who need a quick **risk overview** without heavy setup.
- Data teams validating which signals are most predictive.
"""
)

st.markdown(
    """
### 3-Step Value Flow by RadarRoster
1. **Connect**: Upload your HR CSV or use the sample data.
2. **Score**: Generate risk scores and bands within seconds.
3. **Act**: Prioritize reviews, improve engagement, and reduce attrition cost.
"""
)

st.markdown(
    """
### Business Impact (Typical Outcomes)
- **Faster decision-making** with clear, ranked risk signals.
- **Lower attrition cost** by focusing on the right interventions.
- **Higher HR efficiency** through automated scoring and reporting.
"""
)

st.markdown(
    """
### Mini Case Study (Example)
**Client profile**: 600-employee services company with rising turnover  
**Challenge**: High attrition in two departments, no early-warning signals  
**What we delivered**: Attrition scoring, risk banding, and HR action list  
**Results**: **30% cost reduction**, **70% faster decisions**, **55% productivity lift**
"""
)

with st.sidebar:
    st.header("Inputs")
    use_sample = st.checkbox("Use sample data", value=True)
    rows = st.slider("Sample rows", min_value=50, max_value=1000, value=200, step=50)
    input_file = st.file_uploader("Upload HR CSV", type=["csv"])
    output_path = Path("outputs/retention_scores.csv")
    run_btn = st.button("Run Pipeline")
    st.markdown("---")
    st.subheader("Download templates")
    sample_path = Path("data/sample_hr_data.csv")
    if sample_path.exists():
        st.download_button(
            "Download sample CSV",
            data=sample_path.read_bytes(),
            file_name="sample_hr_data.csv",
            mime="text/csv",
        )
    st.markdown(
        """
**Required columns**
employee_id, dept, tenure_months, last_promotion_months, salary_band,
manager_span, overtime_hours_month, engagement_score, absenteeism_days_month,
peer_turnover_rate, performance_score, internal_mobility
"""
    )

if run_btn:
    if use_sample:
        # Sample flow: generates realistic-but-fake data for safe demos.
        result = run_pipeline(None, output_path, True, rows)
        st.success("Sample pipeline completed.")
        source_df = pd.read_csv(output_path)
    elif input_file is not None:
        # Upload flow: stores the file locally before running the pipeline.
        df = pd.read_csv(input_file)
        temp_path = Path("outputs/upload.csv")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(temp_path, index=False)
        result = run_pipeline(temp_path, output_path, False, rows)
        st.success("Pipeline completed with uploaded data.")
        source_df = pd.read_csv(output_path)
    else:
        st.error("Upload a CSV or use sample data.")
        st.stop()

    scores = source_df.copy()

    st.subheader("Executive Snapshot")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    high_risk = scores[scores["risk_score"] >= 0.7]
    with kpi1:
        st.metric("Employees scored", len(scores))
    with kpi2:
        st.metric("High risk count", len(high_risk))
    with kpi3:
        st.metric("Avg risk score", f"{scores['risk_score'].mean():.2f}")
    with kpi4:
        high_pct = (len(high_risk) / len(scores)) * 100 if len(scores) else 0
        st.metric("High risk %", f"{high_pct:.1f}%")

    with st.expander("What does the model output mean?", expanded=False):
        st.write(
            "- **Risk score** is a 0–1 probability estimate from a baseline model.\n"
            "- **Risk band** groups scores into Low (<0.4), Medium (0.4–0.69), High (>=0.7).\n"
            "- **Top drivers** are the strongest overall model signals, not per-employee explanations."
        )
    with st.expander("How to interpret results", expanded=False):
        st.write(
            "- Use **High risk** to prioritize reviews, not to automate decisions.\n"
            "- Compare **departments** to spot systemic issues.\n"
            "- Track **engagement vs risk** to validate survey quality.\n"
            "- Use the **scored CSV** as a starting point for deeper analysis."
        )

    st.subheader("Risk Distribution")
    fig = px.histogram(scores, x="risk_score", nbins=20)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk Bands")
    band_counts = (
        scores["risk_band"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "risk_band", "risk_band": "count"})
    )
    band_counts["risk_band"] = pd.Categorical(
        band_counts["risk_band"], categories=["Low", "Medium", "High"], ordered=True
    )
    band_counts = band_counts.sort_values("risk_band")
    band_fig = px.bar(band_counts, x="risk_band", y="count")
    st.plotly_chart(band_fig, use_container_width=True)

    st.subheader("Top Drivers (overall)")
    drivers = pd.DataFrame(result.feature_importance.items(), columns=["feature", "weight"])
    drivers = drivers.sort_values("weight", ascending=True)
    driver_fig = px.bar(drivers, x="weight", y="feature", orientation="h")
    st.plotly_chart(driver_fig, use_container_width=True)

    st.subheader("Data Quality & Coverage")
    dq1, dq2, dq3 = st.columns(3)
    with dq1:
        st.metric("Missing values", int(source_df.isna().sum().sum()))
    with dq2:
        dupes = int(source_df["employee_id"].duplicated().sum())
        st.metric("Duplicate employee_id", dupes)
    with dq3:
        invalid_scores = int(((scores["risk_score"] < 0) | (scores["risk_score"] > 1)).sum())
        st.metric("Invalid risk scores", invalid_scores)

    missing_by_col = (
        source_df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing"})
    )
    st.dataframe(missing_by_col)

    st.subheader("Department Risk Overview")
    dept_summary = (
        scores.groupby("dept", as_index=False)
        .agg(avg_risk=("risk_score", "mean"), employees=("employee_id", "count"))
        .sort_values("avg_risk", ascending=False)
    )
    dept_fig = px.bar(
        dept_summary,
        x="dept",
        y="avg_risk",
        color="employees",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(dept_fig, use_container_width=True)

    st.subheader("Engagement vs Risk")
    scatter_fig = px.scatter(
        scores,
        x="engagement_score",
        y="risk_score",
        color="risk_band",
        hover_data=["employee_id", "dept"],
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

    st.subheader("High Risk Watchlist")
    watchlist = scores.sort_values("risk_score", ascending=False).head(15)
    st.dataframe(watchlist)

    st.subheader("Scored Output")
    st.dataframe(scores.head(50))
    st.download_button(
        "Download scored output CSV",
        data=scores.to_csv(index=False).encode("utf-8"),
        file_name="retention_scores.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("© 2026 Daryoosh Dehestani · RadarRoster")
