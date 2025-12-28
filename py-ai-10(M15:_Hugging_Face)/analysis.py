import pandas as pd
import numpy as np
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
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["clean_review"] = df["review"].apply(clean_text)

print("Number of reviews:", len(df))
avg_length = df["clean_review"].apply(lambda x: len(x.split())).mean()
print("Average review length (words):", round(avg_length, 2))

stop_words = set(stopwords.words("english"))

all_tokens = []
clean_tokens = []

for clean in df["clean_review"]:
    tokens = word_tokenize(clean)
    all_tokens.extend(tokens)

    filtered = [
        token for token in tokens
        if token not in stop_words and len(token) >= 3
    ]
    clean_tokens.extend(filtered)

print("Total tokens before cleaning:", len(all_tokens))
print("Total tokens after cleaning:", len(clean_tokens))

word_freq = Counter(clean_tokens)
top_words = word_freq.most_common(15)

print("\nTop 15 Frequent Words:")
for word, freq in top_words:
    print(word, ":", freq)

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
    [(" ".join(bigram), freq) for bigram, freq in top_bigrams],
    columns=["Bigram", "Frequency"]
)

print("\nTop 10 Bigrams:")
print(bigram_df)

bigram_df.to_csv("feedback_bigrams.csv", index=False)
