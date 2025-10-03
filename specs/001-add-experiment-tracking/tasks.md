# Task Breakdown: Experiment Tracking and Baseline Comparison

## 1. Environment Setup
- [ ] Ensure Python 3.13 is installed and used for all development.
- [ ] Add MLflow to `requirements.txt` and install dependencies.

## 2. MLflow Integration
- [ ] Import and initialize MLflow in [`api/model.py`](api/model.py:1).
- [ ] Wrap each asset/model fit in an MLflow run.
- [ ] Log model parameters, metrics, and artifacts (plots, .pkl, .csv) to MLflow.

## 3. Baseline Comparison Dashboard
- [ ] Design and implement a summary table or dashboard for structured comparison of baseline fits.

## 4. Documentation
- [ ] Document the experiment tracking workflow for onboarding and reproducibility.
- [ ] Add usage instructions for MLflow and the dashboard.

## 5. Error Handling & Performance
- [ ] Implement graceful handling of MLflow failures.
- [ ] Ensure experiment tracking does not significantly degrade model fitting performance.
