import joblib
from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_train, y_train, C=1.0, max_iter=1000):
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver='lbfgs',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath="models/logistic_regression_model.joblib"):
    joblib.dump(model, filepath)
    print("Model saved to", filepath)

def load_model(filepath="models/logistic_regression_model.joblib"):
    return joblib.load(filepath)

if __name__ == "__main__":
    import numpy as np
    
    X_dummy = np.array([[0.1, 0.8], [0.2, 0.9], [0.8, 0.1], [0.9, 0.2]])
    y_dummy = np.array([0, 0, 1, 1])
    
    dummy_model = train_logistic_regression(X_dummy, y_dummy)
    predictions = dummy_model.predict(X_dummy)
    probabilities = dummy_model.predict_proba(X_dummy)[:, 1]
    
    print("True Labels:", y_dummy)
    print("Predicted Labels:", predictions)
    print("Positive Probabilities:", probabilities)
