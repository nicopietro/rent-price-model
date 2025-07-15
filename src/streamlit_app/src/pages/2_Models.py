import mlflow
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import os
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from plotly.subplots import make_subplots

# Load MinIO/MLflow config
load_dotenv()

selected_exp = 'rent-price-prediction'

# Get MLflow tracking URI from environment variable, with fallback to mlflow service
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
client = MlflowClient()

st.set_page_config(layout='wide')
st.title('🔎 MLflow Experiments & Models Viewer')

selected_exp = "rent-price-prediction"

exp = client.get_experiment_by_name(selected_exp)

if exp is None:
    st.error(f"🚫 Experiment '{selected_exp}' not found. Please make sure train file was executed.")
    st.stop()

runs = client.search_runs(experiment_ids=[exp.experiment_id], order_by=['start_time DESC'])

# Format run info
data = []
for run in runs:
    run_info = {
        'Run Name': run.data.tags.get('mlflow.runName', '[no name]'),
        'RMSE': run.data.metrics.get('RMSE'),
        'MAE': run.data.metrics.get('MAE'),
        'R2': run.data.metrics.get('R2'),
        'Model Type': run.data.params.get('model'),
    }
    data.append(run_info)

df = pd.DataFrame(data).sort_values(by='Run Name', ascending=True).reset_index(drop=True)
st.subheader('📊 Experiment Runs')
st.dataframe(df, use_container_width=True)


# Ensure all values are numeric
numeric_df = df.copy()
numeric_df['RMSE'] = pd.to_numeric(numeric_df['RMSE'], errors='coerce')
numeric_df['MAE'] = pd.to_numeric(numeric_df['MAE'], errors='coerce')
numeric_df['R2'] = pd.to_numeric(numeric_df['R2'], errors='coerce')

# Subplots layout
fig = make_subplots(
    rows=1,
    cols=3,
    shared_yaxes=False,
    subplot_titles=('RMSE', 'MAE', 'R2'),
    horizontal_spacing=0.15,
)

# RMSE - auto-scaled
fig.add_trace(
    go.Bar(x=numeric_df['Run Name'], y=numeric_df['RMSE'], name='RMSE', marker_color='skyblue'),
    row=1,
    col=1,
)

# MAE - auto-scaled
fig.add_trace(
    go.Bar(x=numeric_df['Run Name'], y=numeric_df['MAE'], name='MAE', marker_color='blue'),
    row=1,
    col=2,
)

# R2 - manually scaled to [0, 1]
fig.add_trace(
    go.Bar(x=numeric_df['Run Name'], y=numeric_df['R2'], name='R2', marker_color='tomato'),
    row=1,
    col=3,
)

# Only modify y-axis for R2
fig.update_yaxes(range=[0, 1], row=1, col=3)

# Final layout
fig.update_layout(
    height=450,
    width=1200,
    showlegend=False,
    title_text='📊 Performance Metrics per Run',
)

st.plotly_chart(fig, use_container_width=True)
