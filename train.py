import csv
from sklearn.model_selection import train_test_split
from src.preprocess import clean_text, encode_label
from src.features import create_tfidf_vectorizer, save_vectorizer
from src.model import train_logistic_regression, save_model
from src.evaluate import compute_metrics, plot_confusion_matrix, analyze_misclassifications

def load_data(filepath="data/IMDB Dataset.csv"):
    reviews = []
    labels = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reviews.append(row["review"])
            labels.append(row["sentiment"])
    return reviews, labels

def main():
    print("Loading dataset...")
    reviews, labels = load_data()
    print(f"Total reviews loaded: {len(reviews)}")

    print("Cleaning text...")
    cleaned_reviews = [clean_text(r) for r in reviews]
    encoded_labels = [encode_label(l) for l in labels]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        cleaned_reviews, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    print(f"Train size: {len(X_train_text)}, Test size: {len(X_test_text)}")

    print("Vectorizing with TF-IDF...")
    vectorizer = create_tfidf_vectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = train_logistic_regression(X_train, y_train, C=1.0, max_iter=1000)

    print("\nEvaluating model on test set...")
    y_pred = model.predict(X_test)
    acc, prec, rec, f1 = compute_metrics(y_test, y_pred)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    plot_confusion_matrix(y_test, y_pred, save_path="reports/figures/confusion_matrix.png")
    analyze_misclassifications(X_test_text, y_test, y_pred, n_samples=3)

    save_vectorizer(vectorizer, "models/tfidf_vectorizer.joblib")
    save_model(model, "models/logistic_regression_model.joblib")

if __name__ == "__main__":
    main()
