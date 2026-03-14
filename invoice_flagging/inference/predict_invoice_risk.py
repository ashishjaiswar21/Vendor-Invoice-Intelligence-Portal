import joblib
import pandas as pd
import os

def predict_invoice_risk(input_data):
    # Path logic to find models from the subfolder
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "models", "invoice_risk_model.pkl")
    scaler_path = os.path.join(base_dir, "models", "invoice_scaler.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    features = ['invoice_quantity', 'invoice_dollars', 'Freight', 
                'total_item_quantity', 'total_item_dollars']
    
    df = pd.DataFrame([input_data])[features]
    df_scaled = scaler.transform(df)
    
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[:, 1][0]
    return prediction, probability

if __name__ == "__main__":
    # --- THIS LOOP MAKES THE OUTPUT CHANGE ---
    test_scenarios = [
        {
            "name": "Clean (Matches)", 
            "data": {'invoice_quantity': 100, 'invoice_dollars': 1000.0, 'Freight': 50.0, 
                     'total_item_quantity': 100, 'total_item_dollars': 1000.0}
        },
        {
            "name": "EXTREME FRAUD ($45k Mismatch)", 
            "data": {'invoice_quantity': 500, 'invoice_dollars': 50000.0, 'Freight': 150.0, 
                     'total_item_quantity': 50, 'total_item_dollars': 5000.0} # Billed for $50k, only received $5k!
        },
        {
            "name": "Quantity Mismatch (Missing Items)", 
            "data": {'invoice_quantity': 1000, 'invoice_dollars': 10000.0, 'Freight': 100.0, 
                     'total_item_quantity': 10, 'total_item_dollars': 100.0} # Billed for 1000 items, received 10!
        }
    ]

    print(f"\n{'INVOICE SCENARIO':<30} | {'RESULT':<12} | {'RISK SCORE'}")
    print("-" * 60)
    for scenario in test_scenarios:
        pred, prob = predict_invoice_risk(scenario["data"])
        status = "FLAGGED 🚨" if pred == 1 else "APPROVED ✅"
        print(f"{scenario['name']:<30} | {status:<12} | {prob:.2%}")