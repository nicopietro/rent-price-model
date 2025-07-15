# Rent Price Model

This project intention is create an ML model for predicting house rent prices in India.

It is composed of an EDA analysis, which includes outlier removal, ML model training, fine tuning and selection and a interactive app to visualize the Analysis and to be used for predicting house prices of user-defined data.

## Quick Start

First ensure Docker is running. To run all required container run:

```bash
docker compose up --build
```

Services will be available in the following links:

- **Streamlit UI**: [http://localhost:8501](http://localhost:8501)  
- **MLflow Tracking UI**: [http://localhost:5000](http://localhost:5000)  
- **MinIO Console**: [http://localhost:9001](http://localhost:9001)  
  - Username: `minioadmin`  
  - Password: `minioadmin`

### Evaluation Metrics

**RMSE (Root Mean Squared Error)**  
Measures average prediction error (penalizes large errors). Lower is better.

RMSE = √(1/n ∑ (ŷ − y)²)

---

**MAE (Mean Absolute Error)**  
Average absolute difference between predicted and actual values. Lower is better.

MAE = (1/n) ∑ |ŷ − y|

---

**R² (R-squared / Coefficient of Determination)**  
Explains how much variance is captured by the model. Closer to 1 is better.

R² = 1 − (∑ (ŷ − y)²) / (∑ (y − ȳ)²)