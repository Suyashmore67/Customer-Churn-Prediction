import pandas as pd
import numpy as np
import joblib
import os

def main():
    print("Starting data scoring process...")
    
    # Define paths
    raw_data_path = os.path.join("data", "raw", "Telco-Customer-Churn.csv")
    output_data_path = os.path.join("data", "scored_customer_churn.csv")
    
    model_path = os.path.join("model", "churn_model.pkl")
    scaler_path = os.path.join("model", "scaler.pkl")
    features_path = os.path.join("model", "feature_names.pkl")
    threshold_path = os.path.join("model", "threshold.txt")
    
    # Load assets
    if not all(os.path.exists(p) for p in [raw_data_path, model_path, scaler_path, features_path, threshold_path]):
        print("Error: Missing required files. Ensure raw data and model files exist in their respective directories.")
        return
        
    df = pd.read_csv(raw_data_path)
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)
    
    with open(threshold_path, "r") as f:
        threshold = float(f.read().strip())
        
    print(f"Loaded model, scaler, and threshold ({threshold}) successfully.")
    
    # Prepare features
    # Drop customerID and Churn target if they exist
    X = df.drop(columns=["customerID", "Churn"], errors="ignore")
    
    # Preprocess TotalCharges (convert empty spaces to NaN and fill with median)
    X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
    total_charges_median = X['TotalCharges'].median()
    X['TotalCharges'].fillna(total_charges_median, inplace=True)
    
    # One-hot encode categorical features
    categorical_cols = X.select_dtypes(include='object').columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Align columns to match the trained features list
    X_encoded = X_encoded.reindex(columns=feature_names, fill_value=0)
    
    # Scale numerical features
    X_scaled = scaler.transform(X_encoded)
    
    # Generate predictions
    churn_probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # Create scored columns
    df_scored = df.copy()
    
    # Clean up TotalCharges in the original data as well so it's numeric in Power BI
    df_scored['TotalCharges'] = pd.to_numeric(df_scored['TotalCharges'], errors='coerce')
    df_scored['TotalCharges'].fillna(total_charges_median, inplace=True)
    
    df_scored["Churn_Probability"] = churn_probabilities
    df_scored["Predicted_Churn"] = np.where(churn_probabilities >= threshold, "Yes", "No")
    
    # Classify Risk Segments
    df_scored["Risk_Segment"] = np.select(
        condlist=[
            churn_probabilities >= 0.70,
            (churn_probabilities >= 0.30) & (churn_probabilities < 0.70)
        ],
        choicelist=["High Risk", "Medium Risk"],
        default="Low Risk"
    )
    
    # Save output
    df_scored.to_csv(output_data_path, index=False)
    print(f"Scored dataset saved to: {output_data_path}")
    print(f"Total scored records: {len(df_scored)}")
    print(f"Predicted Churn Rate: {df_scored['Predicted_Churn'].value_counts(normalize=True).get('Yes', 0):.2%}")

if __name__ == "__main__":
    main()
