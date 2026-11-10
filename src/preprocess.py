import re

def clean_text(text):
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text

def encode_label(sentiment):
    if sentiment.lower() == 'positive':
        return 1
    return 0

if __name__ == "__main__":
    sample = "<br /><br /> The ACTING was Brilliant!!! A must-watch masterpiece"
    print("Original text:", sample)
    print("Cleaned text: ", clean_text(sample))
    print("Positive ->", encode_label("positive"))
    print("Negative ->", encode_label("negative"))
