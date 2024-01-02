import string
import nltk
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')

def analyze_text(file_path):
    word_count = {}
    sentiment_analyzer = SentimentIntensityAnalyzer()
    nlp = spacy.load("en_core_web_sm")

    with open(file_path, 'r') as file:
        for line in file:
            words = line.lower().translate(str.maketrans("", "", string.punctuation)).split()
            for word in words:
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1

    with open(file_path, 'r') as file:
        text = file.read()
        sentiment_scores = sentiment_analyzer.polarity_scores(text)
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

    return word_count, sentiment_scores, text, entities

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def main():
    file_path = input("Enter the path to the text file: ")

    try:
        word_count, sentiment_scores, text, entities = analyze_text(file_path)

        print("\nWord count result:")
        for word, count in word_count.items():
            print(f"{word}: {count}")

        print("\nSentiment Analysis:")
        print(f"Compound Score: {sentiment_scores['compound']}")
        print(f"Positive Score: {sentiment_scores['pos']}")
        print(f"Neutral Score: {sentiment_scores['neu']}")
        print(f"Negative Score: {sentiment_scores['neg']}")

        print("\nNamed Entity Recognition:")
        for entity, label in entities:
            print(f"{entity} ({label})")

        generate_wordcloud(text)

    except FileNotFoundError:
        print("File not found. Please check the file path.")

if __name__ == "__main__":
    main()
