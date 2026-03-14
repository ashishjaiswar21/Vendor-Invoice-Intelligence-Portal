import joblib
import os
# Import your custom modules
from data_preprocessing import load_invoice_data, apply_labels, split_data, scale_features
from modeling_evaluation import train_random_forest, evaluate_classifier

def main():
    # 1. Ensure the models directory exists ONE LEVEL UP
    # ../ tells Python to go up to the 'In Project' folder
    model_path = '../models'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # 2. Load Data (Path is handled inside the function in data_preprocessing.py)
    print("Step 1: Loading Data...")
    df = load_invoice_data() 

    # 3. Preprocess
    print("Step 2: Preprocessing and Labeling...")
    df = apply_labels(df)
    features = ['invoice_quantity', 'invoice_dollars', 'Freight', 
                'total_item_quantity', 'total_item_dollars']
    target = 'flag_invoice'

    # 4. Split & Scale
    print("Step 3: Splitting and Scaling...")
    X_train, X_test, y_train, y_test = split_data(df, features, target)
    
    # Updated path to save the scaler in the central models folder
    scaler_file = os.path.join(model_path, 'invoice_scaler.pkl')
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test, scaler_file) 

    # 5. Train
    print("Step 4: Training Optimized Random Forest (this will take a few minutes)...")
    grid_search = train_random_forest(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_

    # 6. Evaluate
    print("Step 5: Evaluating Performance...")
    evaluate_classifier(best_model, X_test_scaled, y_test, "Optimized Random Forest")

    # 7. Save Final Model to the central models folder
    model_file = os.path.join(model_path, 'invoice_risk_model.pkl')
    joblib.dump(best_model, model_file)
    
    print("-" * 30)
    print(f"SUCCESS: Pipeline finished.")
    print(f"Files saved in: {os.path.abspath(model_path)}")
    print("-" * 30)

if __name__ == "__main__":
    main()