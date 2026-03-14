import joblib
import pandas as pd
import os

# Notice: We removed the invoice import completely!

def predict_freight(order_dollars):
    """
    Predicts the freight cost based on the invoice dollar amount.
    """
    # PATH LOGIC: Go up two levels to find the root /models folder
    current_file_path = os.path.abspath(__file__)
    inference_dir = os.path.dirname(current_file_path)
    freight_dir = os.path.dirname(inference_dir)
    project_root = os.path.dirname(freight_dir)
    
    model_path = os.path.join(project_root, "models", "predict_freight_model.pkl")
    
    if not os.path.exists(model_path):
        return f"Error: Model file not found at {model_path}"
        
    # Load the model
    model = joblib.load(model_path)

    # Prepare the input (The model expects a DataFrame with the column 'Dollars')
    X_new = pd.DataFrame([[order_dollars]], columns=["Dollars"])

    # Make prediction
    prediction = model.predict(X_new)

    return prediction[0]

if __name__ == "__main__":
    # --- FREIGHT TERMINAL TEST BLOCK ---
    # We only need single dollar amounts to test shipping costs
    test_order_values = [150.00, 1500.00, 25000.00]

    print(f"\n{'ORDER VALUE':<20} | {'ESTIMATED FREIGHT'}")
    print("-" * 40)

    for val in test_order_values:
        cost = predict_freight(val)
        
        # If it returns an error string instead of a number, print it
        if isinstance(cost, str):
            print(cost)
            break
            
        print(f"${val:<19,.2f} | ${cost:,.2f}")
    
    print("-" * 40 + "\n")