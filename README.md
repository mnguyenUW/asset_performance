# Asset Performance: Air Filter Restriction Monitoring

This project provides a data-driven solution for monitoring air filter health in industrial assets. It models the relationship between hydraulic horsepower and air filter restriction to detect clogging, quantify filter health, and enable predictive maintenance.

## Project Structure

- [`api/`](api/): Core backend logic, including FastAPI endpoints, model training, prediction utilities, and experiment tracking.
- [`notebook/`](notebook/): Jupyter notebooks for data exploration, baseline modeling, and algorithm demonstration.
- [`data/`](data/): Datasets and analysis outputs for model training and validation.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Key packages: `fastapi`, `mlflow`, `scikit-learn`, `joblib`, `pandas`, `matplotlib`, `numpy`.

2. **(Optional) Docker**:
   ```bash
   docker build -t asset-performance .
   docker run -p 8000:8000 asset-performance
   ```

3. **Prepare data**:
   - Place `air_filter_data.csv` and `asset_limits.csv` in the `data/` directory.

## Usage

### Run the API server

```bash
uvicorn api.app:app --reload
```

### API Endpoints

- **POST `/predict_clean_restriction`**  
  Predicts the expected air filter restriction for a given asset type and horsepower.
- **POST `/estimate_clog`**  
  Estimates the percent clogged and maximum achievable horsepower based on current readings.

### Model Training & Experiment Tracking

- Model fitting and experiment tracking are handled in [`api/model.py`](api/model.py:1) using MLflow.
- Artifacts (models, plots, CSVs) are logged for each asset.
- Combined models are saved as `api/all_models.pkl`.

## Data

- [`data/air_filter_data.csv`](data/air_filter_data.csv): Historical operational data (asset, hydraulic horsepower, air filter restriction, timestamps).
- [`data/asset_limits.csv`](data/asset_limits.csv): Asset specifications (asset, Max_AirFilterRestriction, Max_Horsepower).
- Analysis images: Asset-specific performance plots.

## Notebooks

- [`notebook/exploration.ipynb`](notebook/exploration.ipynb:1): Data exploration, visualization, and motivation for baseline modeling.
- [`notebook/baseline_model.ipynb`](notebook/baseline_model.ipynb:1): Clean baseline curve fitting for each asset.
- [`notebook/calculations.ipynb`](notebook/calculations.ipynb:1): Clogging detection algorithm and scenario demonstrations.

