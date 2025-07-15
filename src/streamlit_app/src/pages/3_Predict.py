import mlflow
import pandas as pd
import streamlit as st
import os
from mlflow.tracking import MlflowClient

# Get MLflow tracking URI from environment variable, with fallback to localhost
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
client = MlflowClient()

st.set_page_config(layout="wide")
st.title("🏠 Predict Rent Price")

selected_exp = "rent-price-prediction"

exp = client.get_experiment_by_name(selected_exp)

if exp is None:
    st.error(f"🚫 Experiment '{selected_exp}' not found. Please make sure train file was executed.")
    st.stop()

all_models = []

runs = client.search_runs([exp.experiment_id])
for run in runs:
    run_info = {
        "Run Name": run.data.tags.get("mlflow.runName", "[no name]"),
        "RMSE": run.data.metrics.get("RMSE"),
        "MAE": run.data.metrics.get("MAE"),
        "R2": run.data.metrics.get("R2"),
        "Model Type": run.data.params.get("model"),
    }
    all_models.append(run_info)

if not all_models:
    st.warning("No models found across experiments.")
    st.stop()

st.markdown("### 📦 Available Models (from all runs)")
st.dataframe(all_models, use_container_width=True)

st.markdown("### ✍️ Enter Features")

with st.form("prediction_form"):
    size = st.number_input("Size (sq ft)", min_value=100, max_value=10000, value=1000)
    bhk = st.selectbox("BHK", options=[1, 2, 3, 4, 5], index=1)
    bathroom = st.selectbox("Bathroom", options=[1, 2, 3, 4], index=0)
    city = st.selectbox("City", options=["Mumbai", "Delhi", "Bangalore", "Hyderabad"])
    furnishing = st.selectbox("Furnishing Status", options=["Furnished", "Semi-Furnished", "Unfurnished"])
    tenant = st.selectbox("Tenant Preferred", options=["Family", "Bachelors", "Bachelors/Family", ])
    floor_group = st.selectbox("Floor Group", options=["Lower Floor", "Upper Floor", "Ground Floor/Basement"])

    submitted = st.form_submit_button("🔮 Predict with All Models")

if submitted:
    with st.spinner("⏳ Running predictions... Please wait."):
        input_dict = [{
        "Size": size,
        "BHK": bhk,
        "Bathroom": bathroom,
        "City": city,
        "Furnishing Status": furnishing,
        "Tenant Preferred": tenant,
        "Floor Group": floor_group
        }]
        input_data = pd.DataFrame(input_dict)
        
        models = {}
        for model in client.search_registered_models():
            try:
                models[model.name] = mlflow.pyfunc.load_model(f"models:/{model.name}/latest")
            except Exception as e:
                print(f"❌ Failed to load {model.name}: {e}")
        print("models in client: ",client.search_registered_models())
        print("loaded models: ",models)
        
        predictions = pd.DataFrame(columns=["Model", "Predicted Rent Price"])
        for name, model in models.items():
            
            new_row = pd.DataFrame([{"Model": name, "Predicted Rent Price": float(model.predict(input_data))}])

            predictions = pd.concat([predictions, new_row], ignore_index=True)

    st.success("✅ Prediction complete!")
    
    # Enhanced predictions display
    st.markdown("### 🎯 Model Predictions")
    
    # Add ranking and performance info to predictions
    enhanced_predictions = predictions.copy()
    enhanced_predictions['Rank'] = range(1, len(enhanced_predictions) + 1)
    
    # Add model performance metrics
    model_performance = {}
    for model in all_models:
        model_performance[model['Run Name']] = {
            'RMSE': model['RMSE'],
            'R2': model['R2'],
            'MAE': model['MAE']
        }
    
    # Add performance columns
    enhanced_predictions['RMSE'] = enhanced_predictions['Model'].apply(
        lambda x: next((model_performance[name]['RMSE'] for name in model_performance if name in x), None)
    )
    enhanced_predictions['R² Score'] = enhanced_predictions['Model'].apply(
        lambda x: next((model_performance[name]['R2'] for name in model_performance if name in x), None)
    )
    
    # Sort by RMSE (best first)
    enhanced_predictions = enhanced_predictions.sort_values('RMSE').reset_index(drop=True)
    enhanced_predictions['Rank'] = range(1, len(enhanced_predictions) + 1)
    
    # Format the price column
    enhanced_predictions['Predicted Rent Price'] = enhanced_predictions['Predicted Rent Price'].apply(
        lambda x: f"₹{x:,.0f}"
    )
    
    # Format performance metrics
    enhanced_predictions['RMSE'] = enhanced_predictions['RMSE'].apply(
        lambda x: f"{x:.4f}" if x is not None else "N/A"
    )
    enhanced_predictions['R² Score'] = enhanced_predictions['R² Score'].apply(
        lambda x: f"{x:.4f}" if x is not None else "N/A"
    )
    
    # Reorder columns for better display
    enhanced_predictions = enhanced_predictions[['Rank', 'Model', 'Predicted Rent Price', 'RMSE', 'R² Score']]
    
    # Display with custom styling
    st.dataframe(
        enhanced_predictions,
        use_container_width=True,
        column_config={
            "Rank": st.column_config.NumberColumn(
                "🏆 Rank",
                help="Model ranking based on RMSE (lower is better)",
                format="%d"
            ),
            "Model": st.column_config.TextColumn(
                "🤖 Model",
                help="Machine learning model name"
            ),
            "Predicted Rent Price": st.column_config.TextColumn(
                "💰 Predicted Price",
                help="Predicted rent price in Indian Rupees"
            ),
            "RMSE": st.column_config.TextColumn(
                "📊 RMSE",
                help="Root Mean Square Error (lower is better)"
            ),
            "R² Score": st.column_config.TextColumn(
                "📈 R² Score",
                help="R-squared score (higher is better)"
            )
        },
        hide_index=True
    )
    
    # Add summary statistics
    if not predictions.empty:
        st.markdown("### 📊 Prediction Summary")
        
        # Find the best model (lowest RMSE)
        best_model_info = min(all_models, key=lambda x: x['RMSE'] if x['RMSE'] is not None else float('inf'))
        best_model_name = best_model_info['Run Name']
        
        # Find the prediction for the best model
        best_prediction = None
        for _, row in predictions.iterrows():
            if best_model_name in row['Model']:
                best_prediction = row['Predicted Rent Price']
                break
        
        # If best model prediction not found, use the first available
        if best_prediction is None and not predictions.empty:
            best_prediction = predictions.iloc[0]['Predicted Rent Price']
            best_model_name = predictions.iloc[0]['Model']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average", f"₹{predictions['Predicted Rent Price'].mean():,.0f}")
        
        with col2:
            st.metric("Maximum", f"₹{predictions['Predicted Rent Price'].max():,.0f}")
        
        with col3:
            st.metric("Minimum", f"₹{predictions['Predicted Rent Price'].min():,.0f}")
        
        with col4:
            st.metric("Range", f"₹{predictions['Predicted Rent Price'].max() - predictions['Predicted Rent Price'].min():,.0f}")
        
        # Additional statistics
        col5, col6 = st.columns(2)
        
        with col5:
            st.metric("Median", f"₹{predictions['Predicted Rent Price'].median():,.0f}")
        
        with col6:
            st.metric("Standard Deviation", f"₹{predictions['Predicted Rent Price'].std():,.0f}")
        
        # Highlight best model prediction
        if best_prediction is not None:
            st.markdown("---")
            st.markdown("### 🏆 Best Model Prediction")
            
            # Create a highlighted box for the best prediction
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin: 10px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            ">
                <h3 style="margin: 0; color: white;">🎯 Recommended Rent Price</h3>
                <h2 style="margin: 10px 0; color: white; font-size: 2.5em;">₹{best_prediction:,.0f}</h2>
                <p style="margin: 0; color: white; opacity: 0.9;">Based on {best_model_name} (Best RMSE: {best_model_info['RMSE']:.4f})</p>
            </div>
            """, unsafe_allow_html=True)
