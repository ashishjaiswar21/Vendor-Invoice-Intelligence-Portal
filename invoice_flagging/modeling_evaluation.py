from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score

def train_random_forest(X_train, y_train):
    # n_jobs=2 tells Python to use only 2 cores, leaving the rest for your OS
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=2 
    )

    # Simplified grid: This reduces total fits from 2,160 down to just 40.
    # This will finish in about 1-2 minutes without freezing your laptop.
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 8],
        "min_samples_split": [2, 5],
        "criterion": ["gini"]
    }

    scorer = make_scorer(f1_score)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        n_jobs=2,      # Limit to 2 cores here as well
        verbose=2      # This prints progress so you know it's not stuck
    )

    grid_search.fit(X_train, y_train)
    return grid_search

def evaluate_classifier(model, X_test, y_test, model_name):
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    print(f"\n{model_name} Performance")
    print(f"Accuracy: {accuracy:.2f}")
    print(report)