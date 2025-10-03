# Implementation Plan: Experiment Tracking and Baseline Comparison

**Python Version:** 3.13  
**Experiment Tracking Framework:** MLflow

## Steps

1. **Update Environment**
   - Ensure Python 3.13 is used for all development and deployment.
   - Add MLflow to `requirements.txt`.

2. **Integrate MLflow**
   - Import and initialize MLflow in [`api/model.py`](api/model.py:1).
   - Wrap each asset/model fit in an MLflow run.
   - Log model parameters, metrics, and artifacts (plots, .pkl, .csv) to MLflow.

3. **Comparison Dashboard**
   - Create a summary table or dashboard for structured comparison of baseline fits.

4. **Documentation**
   - Document the experiment tracking workflow for onboarding and reproducibility.

## Notes

- All experiment tracking and logging must be compatible with Python 3.13.
- MLflow is the required tracking solution.
- Handle MLflow failures gracefully and document any limitations.
