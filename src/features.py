from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from sklearn.feature_extraction.text import TfidfTransformer

def create_tfidf_vectorizer(max_features=10000, ngram_range=(1,2)):
    """
        Creates and configures a TF-IDF vecotrizer:
        - max_functions: keeps the top 10,000 most informativ words/phases
        - ngram_range=(1,2): Extracts both single words and 2-word pairs
        - sublinear_tf=True: Scales term frequencey logorithmically to prevent words repeated 10 times what from dominating the score.
    """

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        stop_words='english'
    )

    return vectorizer

def save_vectorizer(vectorizer, filepath="models/tfidf_vectorizer.joblib"):
    """
    Save the fitted vectorizer to disk so we can reuse it later.
    """
    joblib.dump(vectorizer,filepath)
    print(f"Vectorizer successfully saved to {filepath}")

def load_vectorizer(filepath="models/tfidf_vectorizer.joblib"):
    """
        Loads a previously saved vectorizer from disk.
    """
    return joblib.load(filepath)


if __name__ == "__main__":
    sample_texts = [
        "the movie was brilliant and stunning",
        "the movie was terrible and not good",
        "an absolute   with great acting"
    ]
    
    vec = create_tfidf_vectorizer(max_features=10)
    matrix = vec.fit_transform(sample_texts)
    
    print("--- TF-IDF Feature Extraction Test ---")
    print("Learned Vocabulary Features:")
    print(vec.get_feature_names_out())
    print("\nShape of Output Matrix (3 reviews, 10 features):")
    print(matrix.toarray())


