import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import bigrams
from collections import Counter

df = pd.read_csv("feedbacks.csv")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df["clean_review"] = df["review"].apply(clean_text)

print("Кількість рядків:", len(df))
avg_length = df["clean_review"].apply(lambda x: len(x.split())).mean()
print("Середня довжина відгуку (у словах):", round(avg_length, 2))

all_tokens = []
clean_tokens = []

stop_words = set(stopwords.words("english"))

for rev in df["clean_review"]:
    tokens = word_tokenize(rev)
    all_tokens.extend(tokens)

    filtered = [
        word for word in tokens
        if word not in stop_words and len(word) >= 3
    ]
    clean_tokens.extend(filtered)

print("Кількість токенів до очищення:", len(all_tokens))
print("Кількість токенів після очищення:", len(clean_tokens))

word_freq = Counter(clean_tokens)
top_words = word_freq.most_common(15)

print("\nTop 15 Frequent Words:")
for word, count in top_words:
    print(word, "-", count)

words, counts = zip(*top_words)

plt.figure(figsize=(10, 6))
plt.barh(words, counts)
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.title("Top 15 Frequent Words")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feedback_word_freq.png")
plt.show()

bigram_list = list(bigrams(clean_tokens))
bigram_freq = Counter(bigram_list)

top_bigrams = bigram_freq.most_common(10)

bigram_df = pd.DataFrame(
    [(" ".join(bigram), count) for bigram, count in top_bigrams],
    columns=["Bigram", "Frequency"]
)

print("\nTop 10 Bigrams:")
print(bigram_df)

bigram_df.to_csv("feedback_bigrams.csv", index=False)
