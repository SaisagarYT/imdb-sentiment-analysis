import re

def clean_text(text):
    """
        Cleans raw review text:
        1. Removes HTML tags like <br />
        2. Keeps only letters (removes special symobls, numbers, punctuation)
        3. Converts to lowercase
        4. Trims excess whitespace
    """

    text = re.sub(r'<[^>]*>',' ',text)
    text = re.sub(r'[^a-zA-Z\s]', ' ',text)
    text = text.lower()
    text = ' '.join(text.split())

    return text

def encode_label(sentiment):
    """
        Converts text sentiment into a binary number:
        'positive' -> 1
        'negative -> 0
    """

    return 1 if sentiment.lower() == 'positive' else 0

if __name__ == "__main__":
    sample = "<br /><br /> The ACTING was Brilliant!!! A must-watch masterpiece"
    print("---Testing Text Preprocessing---")
    print("Raw:  ",sample)
    print("Cleaned:  ", clean_text(sample))
    print("\n--- Testing Label Encoding ---")
    print("Positive ->", encode_label("positive"))
    print("Negative ->", encode_label("negative"))