# GMV Forecasting System

> End-to-end ML pipeline for time series forecasting using hybrid SARIMAX-Prophet ensemble model.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-green.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org/)

---

## Overview

Production-ready forecasting system that combines **SARIMAX** (statistical) and **Facebook Prophet** (ML) models in an optimized ensemble, achieving **0.94-1.57% MAPE**.

**Key Features**:
- Hybrid ensemble with automatic weight optimization
- REST API for predictions
- MLflow experiment tracking
- Docker deployment
- Full MLOps infrastructure

---

## Performance

| City | Hybrid MAPE | SARIMAX | Prophet | Weights |
|------|-------------|---------|---------|---------|
| Ho Chi Minh | **1.57%** | 2.39% | 1.62% | 24% / 76% |
| Hanoi | **0.94%** | 2.15% | 1.20% | 45% / 55% |

---

## Architecture

### ML Pipeline Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Raw Data  │ -> │ Preprocessing │ -> │   Training  │ -> │  Deployment  │
│  (CSV file) │    │  & Features   │    │   Models    │    │   (Docker)   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
       |                   |                    |                   |
       v                   v                    v                   v
  data/raw/          Validation         SARIMAX + Prophet    FastAPI + MLflow
  data.csv          StandardScaler        + Hybrid          Monitoring
```

### System Components

```
┌────────────────────────────────────────────────────────────┐
│                    Docker Environment                       │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   FastAPI    │  │    MLflow    │  │  Prometheus  │    │
│  │ (Port 8000)  │  │ (Port 5001)  │  │ (Port 9090)  │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Grafana    │  │  PostgreSQL  │  │    Redis     │    │
│  │ (Port 3000)  │  │ (Port 5432)  │  │ (Port 6379)  │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└────────────────────────────────────────────────────────────┘
```

---

## Full ML Lifecycle

### 1. Data Pipeline

```python
# src/data/data_loader.py
┌─────────────────────────────────────────┐
│ 1. Load CSV                             │
│ 2. Validate schema & date format       │
│ 3. Handle missing values                │
│ 4. Feature scaling (StandardScaler)    │
│ 5. Train/test split by date            │
│ 6. City-wise data splitting             │
└─────────────────────────────────────────┘
```

**Input**: `data/raw/data.csv`
- Columns: `week_date`, `city_name`, `gmv`, `x1-x18`

**Output**: `data/processed/`
- `train_{city}.csv`
- `test_{city}.csv`

### 2. Model Training

```python
# src/models/train.py
┌─────────────────────────────────────────┐
│ For each city:                          │
│                                         │
│ 1. SARIMAX Training                     │
│    ├─ Auto-ARIMA parameter selection   │
│    ├─ Test m=12, 26, 52 seasonal       │
│    └─ Select best by AICc              │
│                                         │
│ 2. Prophet Training                     │
│    ├─ Add seasonalities                │
│    ├─ Add country holidays (VN)        │
│    └─ Add 18 regressors                │
│                                         │
│ 3. Hybrid Optimization                  │
│    ├─ Get predictions from both         │
│    ├─ Optimize weights on validation    │
│    └─ Save ensemble model               │
│                                         │
│ 4. MLflow Logging                       │
│    ├─ Parameters                        │
│    ├─ Metrics (MAPE, MAE, RMSE)        │
│    └─ Model artifacts                   │
└─────────────────────────────────────────┘
```

**Models**:
- **SARIMAX**: `statsmodels.tsa.statespace.sarimax.SARIMAX`
- **Prophet**: `facebook/prophet` with CmdStan backend
- **Hybrid**: Weighted combination with scipy.optimize

### 3. Model Serving

```python
# src/api/main.py
┌─────────────────────────────────────────┐
│ FastAPI Endpoints:                      │
│                                         │
│ GET  /health                            │
│      → Health check                     │
│                                         │
│ GET  /models                            │
│      → List available models            │
│                                         │
│ POST /predict                           │
│      → Generate forecasts               │
│      Request: {                         │
│        "city": "Ho Chi Minh",           │
│        "periods": 12,                   │
│        "model_type": "hybrid"           │
│      }                                  │
└─────────────────────────────────────────┘
```

### 4. Monitoring & Tracking

```python
┌─────────────────────────────────────────┐
│ MLflow: Experiment tracking             │
│ ├─ Training runs                        │
│ ├─ Model versioning                     │
│ └─ Artifact storage                     │
│                                         │
│ Prometheus: System metrics              │
│ ├─ API latency                          │
│ ├─ Prediction count                     │
│ └─ Error rates                          │
│                                         │
│ Grafana: Visualization                  │
│ └─ Real-time dashboards                 │
└─────────────────────────────────────────┘
```

---

## Project Structure

```
gmv_forecasting_mlops/
├── config/
│   └── config.yaml              # All configurations
│
├── data/
│   ├── raw/                     # Original data
│   ├── processed/               # Train/test splits
│   └── predictions/             # Forecast outputs
│
├── src/
│   ├── data/
│   │   ├── data_loader.py      # Data loading & validation
│   │   └── prepare_data.py     # Preprocessing pipeline
│   │
│   ├── models/
│   │   ├── base_model.py       # Abstract base class
│   │   ├── sarimax_model.py    # SARIMAX implementation
│   │   ├── prophet_model.py    # Prophet implementation
│   │   ├── hybrid_model.py     # Ensemble model
│   │   └── train.py            # Training pipeline
│   │
│   ├── api/
│   │   ├── main.py             # FastAPI app
│   │   ├── schemas.py          # Request/response models
│   │   └── model_manager.py    # Model loading
│   │
│   ├── evaluation/
│   │   └── evaluate.py         # Metrics calculation
│   │
│   └── utils/
│       ├── config_loader.py    # Config management
│       └── logger.py           # Logging setup
│
├── deployment/
│   ├── Dockerfile              # Container definition
│   ├── docker-compose.yml      # Multi-service setup
│   └── docker-compose.train.yml # Training only
│
├── tests/                       # Unit & integration tests
├── models/                      # Trained model artifacts
├── mlruns/                      # MLflow experiments
└── outputs/                     # Plots & reports
```

---

## Quick Start

### Prerequisites
- Docker Desktop installed
- OR Python 3.10+ (for local development)

### Option 1: Docker (Recommended)

#### 1. Build Image
```bash
docker compose -f deployment/docker-compose.yml build
```

#### 2. Prepare Data

Place your data file in `data/raw/data.csv`:

**Required format**:
```csv
week_date,city_name,gmv,x1,x2,x3,...,x18
2020-01-01,Ho Chi Minh,1000000,0.5,0.3,0.8,...,0.6
2020-01-08,Hanoi,800000,0.6,0.4,0.7,...,0.5
```

**Columns**:
- `week_date`: Date (YYYY-MM-DD format)
- `city_name`: City identifier
- `gmv`: Target variable (Gross Merchandise Value)
- `x1` to `x18`: Feature columns

#### 3. Train Models
```bash
docker compose -f deployment/docker-compose.train.yml up
```

**Output Sample**:
```
INFO - Training SARIMAX component
INFO - Best SARIMAX: m=12, AIC=2425.78
INFO - Training Prophet component
INFO - Prophet fitted successfully ✓
INFO - Optimizing hybrid weights
INFO - Weights: SARIMAX=0.24, Prophet=0.76
INFO - Hybrid MAPE: 1.57%
```

#### 4. Start Services
```bash
docker compose -f deployment/docker-compose.yml up -d
```

**Access**:
- API: http://localhost:8000/docs
- MLflow: http://localhost:5001
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

### Option 2: Local Python

#### 1. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. Prepare Data

```bash 
python src/data/prepare_data.py --config config/config.yaml # (Optional) Copy your latest data file into the raw data directory first
```

#### 3. Train Models
```bash
python src/models/train.py --config config/config.yaml
```

#### 4. Evaluate
```bash
python src/evaluation/evaluate.py --config config/config.yaml
```

#### 5. Start API
```bash
uvicorn src.api.main:app --reload --port 8000
```

---

## Usage Examples

### Make Predictions

**Via API**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Ho Chi Minh",
    "periods": 12,
    "model_type": "hybrid"
  }'
```

**Response**:
```json
{
  "city": "Ho Chi Minh",
  "predictions": [
    {"date": "2024-01-01", "value": 952341.23},
    {"date": "2024-02-01", "value": 978234.56},
    ...
  ],
  "model_info": {
    "sarimax_weight": 0.24,
    "prophet_weight": 0.76,
    "mape": 1.57
  }
}
```

**Via Python**:
```python
from src.api.model_manager import ModelManager

# Load model
manager = ModelManager()
model = manager.get_model("hybrid_ho_chi_minh")

# Predict
predictions = model.predict(periods=12, exog_future=None)
print(predictions)
```

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLETE ML WORKFLOW                          │
└─────────────────────────────────────────────────────────────────┘

1. DATA INGESTION
   └─ CSV → Validation → Cleaning
      ↓
2. PREPROCESSING
   └─ Missing values → Feature scaling → Train/test split
      ↓
3. MODEL TRAINING
   ├─ SARIMAX (Auto-ARIMA) ──┐
   ├─ Prophet (with holidays) ┤→ Ensemble Optimization
   └─ Hybrid weights          │
      ↓                        │
4. MODEL EVALUATION           │
   └─ MAPE, MAE, RMSE metrics ┘
      ↓
5. MODEL REGISTRY
   └─ MLflow → Version → Store artifacts
      ↓
6. DEPLOYMENT
   └─ Docker → FastAPI → Load model
      ↓
7. SERVING
   └─ REST API → Predictions
      ↓
8. MONITORING
   └─ Prometheus → Grafana → Alerts
```

---

## Configuration

All settings in `config/config.yaml`:

```yaml
data:
  raw_path: "data/raw/data.csv"
  date_column: "week_date"
  target_column: "gmv"
  city_column: "city_name"
  cities: ["Ho Chi Minh", "Hanoi"]

models:
  sarimax:
    seasonal_periods: [12, 26, 52]
    trend: "c"
  
  prophet:
    seasonality_mode: "multiplicative"
    country_holidays: "VN"
  
  hybrid:
    optimization_metric: "mape"

api:
  host: "0.0.0.0"
  port: 8000
```

---

## Model Details

### SARIMAX
- **Auto-ARIMA** parameter selection
- **Seasonal periods**: 12, 26, 52 weeks tested
- **Selection criterion**: AICc (Corrected Akaike Information Criterion)
- **Exogenous features**: 18 scaled variables

### Prophet
- **Trend**: Piecewise linear with changepoint detection
- **Seasonalities**: Yearly, monthly, quarterly (Fourier series)
- **Holidays**: Vietnam-specific
- **Regressors**: 18 external features
- **Backend**: CmdStan (compiled in Docker)

### Hybrid Ensemble
- **Method**: Weighted average
- **Optimization**: scipy.minimize on validation set
- **Metric**: MAPE minimization
- **Weights**: City-specific (not global)

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html

# Specific test
pytest tests/test_models.py -v
```

**Coverage**: 85%+

---

## Retraining

When new data is available:

```bash
# 1. Update data
cp new_data.csv data/raw/data.csv

# 2. Retrain
docker compose -f deployment/docker-compose.train.yml up

# 3. Restart API
docker compose -f deployment/docker-compose.yml restart api
```

Models are automatically versioned in MLflow.

---

## Monitoring

### MLflow UI
```
http://localhost:5001
```
- View all training runs
- Compare model metrics
- Download artifacts

### Prometheus Metrics
```
http://localhost:9090
```
- API latency
- Prediction count
- Error rates

### Grafana Dashboards
```
http://localhost:3000
```
- Real-time visualization
- Custom alerts
- Performance tracking

---

## Tech Stack

### Data Science & ML
- **Python**: 3.10+
- **Time Series**: statsmodels (SARIMAX), pmdarima (Auto-ARIMA)
- **ML**: Facebook Prophet, scikit-learn
- **Data Processing**: pandas, numpy
- **Optimization**: scipy

### MLOps & Infrastructure
- **Experiment Tracking**: MLflow
- **API Framework**: FastAPI, Pydantic, Uvicorn
- **Containerization**: Docker, Docker Compose
- **Monitoring**: Prometheus, Grafana
- **Database**: PostgreSQL (optional)
- **Caching**: Redis (optional)
- **Testing**: pytest, pytest-cov
- **CI/CD**: GitHub Actions

---

## Repository & Version Control

### What's Included in Git
✅ Source code (`src/`)
✅ Configuration files (`config/`)
✅ Tests (`tests/`)
✅ Docker setup (`deployment/`)
✅ Documentation (README, LICENSE)
✅ Dependencies (`requirements.txt`)
✅ CI/CD (`.github/workflows/`)

### What's Excluded (.gitignore)
❌ Data files (`data/raw/`, `data/processed/`)
❌ Trained models (`models/`)
❌ MLflow experiments (`mlruns/`)
❌ Virtual environments (`venv/`)
❌ Logs and outputs (`logs/`, `outputs/`)
❌ Python cache (`__pycache__/`, `*.pyc`)

**Why**: Large files, generated files, and sensitive data should not be in version control.

### For Users Cloning This Repo

After cloning, you need to:
1. Add your own data to `data/raw/data.csv`
2. Build Docker image (first time: 10-15 min)
3. Train models with your data (5-10 min)
4. Start services

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## Author

**Thinh Nguyen**
- GitHub: [@hungthinh005](https://github.com/hungthinh005)
- Project: [GMV Forecasting MLOps](https://github.com/hungthinh005/gmv-forecasting-mlops)

---

*This project demonstrates a complete ML lifecycle from data processing to production deployment with MLOps best practices.*