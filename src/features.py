from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def create_tfidf_vectorizer(max_features=10000, ngram_range=(1, 2)):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        stop_words='english'
    )
    return vectorizer

def save_vectorizer(vectorizer, filepath="models/tfidf_vectorizer.joblib"):
    joblib.dump(vectorizer, filepath)
    print("Vectorizer saved to", filepath)

def load_vectorizer(filepath="models/tfidf_vectorizer.joblib"):
    return joblib.load(filepath)

if __name__ == "__main__":
    sample_texts = [
        "the movie was brilliant and stunning",
        "the movie was terrible and not good",
        "an absolute masterpiece with great acting"
    ]
    
    vec = create_tfidf_vectorizer(max_features=10)
    matrix = vec.fit_transform(sample_texts)
    
    print("Features extracted:")
    print(vec.get_feature_names_out())
    print("Matrix shape:", matrix.shape)
