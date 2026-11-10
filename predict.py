import sys
from src.preprocess import clean_text
from src.features import load_vectorizer
from src.model import load_model

def predict_sentiment(review_text, vectorizer, model):
    cleaned = clean_text(review_text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probability[1] if prediction == 1 else probability[0]
    return sentiment, confidence

def main():
    vectorizer = load_vectorizer("models/tfidf_vectorizer.joblib")
    model = load_model("models/logistic_regression_model.joblib")
    
    if len(sys.argv) > 1:
        user_review = " ".join(sys.argv[1:])
        sentiment, conf = predict_sentiment(user_review, vectorizer, model)
        print(f"\nReview: {user_review}")
        print(f"Sentiment: {sentiment} ({conf * 100:.2f}% confidence)\n")
    else:
        print("Interactive IMDb Sentiment Predictor (Type 'exit' to quit):")
        while True:
            user_input = input("\nEnter a movie review: ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            if not user_input:
                continue
            sentiment, conf = predict_sentiment(user_input, vectorizer, model)
            print(f"Predicted Sentiment: {sentiment} ({conf * 100:.2f}% confidence)")

if __name__ == "__main__":
    main()
