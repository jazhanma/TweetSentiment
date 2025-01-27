import pandas as pd
import re
import emoji
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from langdetect import detect, LangDetectException
import nltk

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

# ----------------------------
# IMPROVED TEXT CLEANING
# ----------------------------
def clean_tweet(tweet):
    # Extract emojis
    emojis = ''.join(c for c in tweet if c in emoji.EMOJI_DATA or emoji.is_emoji(c))
    
    # Remove URLs completely
    tweet = re.sub(r"http\S+|www\S+", "", tweet)
    
    # Remove mentions/hashtags but keep text content
    tweet = re.sub(r"@\w+|#\w+", "", tweet)
    
    # Language detection
    try:
        lang = detect(tweet) if len(tweet) > 10 else 'en'
    except LangDetectException:
        lang = 'en'
    
    # Text cleaning based on language
    if lang == 'en':
        tweet = re.sub(r"[^a-zA-Z\s]", "", tweet)
        contraction_map = {
            "im": "i am", "dont": "do not", "cant": "cannot",
            "wont": "will not", " ur ": " your ", " u ": " you "
        }
        for term, replacement in contraction_map.items():
            tweet = tweet.replace(term, replacement)
    else:
        tweet = re.sub(r"[^\w\s]", "", tweet)
    
    # Process words
    words = tweet.lower().split()
    cleaned_words = []
    for word in words:
        if len(word) > 3 and word not in stop_words:
            corrected = spell.correction(word) or word
            lemmatized = lemmatizer.lemmatize(corrected)
            cleaned_words.append(lemmatized)
    
    return " ".join(cleaned_words), emojis, lang

# ----------------------------
# LOAD ORIGINAL DATASET
# ----------------------------
try:
    print("⏳ Loading original dataset...")
    dataset = pd.read_csv(
        r"C:\Users\jaska\XAI\archive\training.1600000.processed.noemoticon.csv",
        encoding='latin-1',
        header=None
    )
    dataset.columns = ["target", "id", "date", "flag", "user", "text"]
    print("✅ Original dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: File not found. Check the path!")
    exit()

# Create a balanced subset for faster testing (1000 of each class)
print("⚙️  Creating balanced dataset...")
positive = dataset[dataset["target"] == 4].head(1000)
negative = dataset[dataset["target"] == 0].head(1000)
neutral = dataset[dataset["target"] == 2].head(1000) if 2 in dataset["target"].unique() else pd.DataFrame()
if neutral.empty:
    print("⚠️  No neutral tweets found, skipping neutral analysis!")
dataset = pd.concat([positive, negative, neutral])

# Clean text and process
print("🧹 Cleaning text...")
dataset["clean_text"], dataset["emojis"], dataset["language"] = zip(
    *dataset["text"].apply(clean_tweet)
)

# ----------------------------
# ANALYSIS
# ----------------------------
print("\n📊 Analyzing results...")

# Improved word analysis
custom_stopwords = {"get", "like", "go", "got", "today", "know", "back", "want"}
def get_top_words(target_value):
    class_data = dataset[dataset["target"] == target_value]
    words = " ".join(class_data["clean_text"]).split()
    filtered = [word for word in words if word not in custom_stopwords]
    return Counter(filtered).most_common(5)

print("\n🔝 Top Positive Words:", get_top_words(4))
print("🔝 Top Negative Words:", get_top_words(0))
if not neutral.empty:
    print("🔝 Top Neutral Words:", get_top_words(2))
else:
    print("🔝 Top Neutral Words: [No neutral data]")

# Improved emoji analysis
def get_top_emojis(target_value):
    emojis = "".join(dataset[dataset["target"] == target_value]["emojis"])
    return Counter(emojis).most_common(3) if emojis else [("None found", 0)]

print("\n😊 Top Positive Emojis:", get_top_emojis(4))
print("😠 Top Negative Emojis:", get_top_emojis(0))

# ----------------------------
# VISUALIZATION
# ----------------------------
print("\n🎨 Generating report...")
plt.figure(figsize=(15, 6))

# Sentiment distribution
plt.subplot(1, 2, 1)
dataset["target"].replace({0: "Negative", 2: "Neutral", 4: "Positive"}).value_counts().plot(
    kind="bar", color=["red", "gray", "green"], edgecolor="black"
)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment Category")
plt.ylabel("Number of Tweets")

# Emoji visualization
plt.subplot(1, 2, 2)
all_emojis = "".join(dataset["emojis"])
top_emojis = Counter(all_emojis).most_common(5) if all_emojis else []
if top_emojis:
    plt.bar([e[0] for e in top_emojis], [e[1] for e in top_emojis])
    plt.title("Top 5 Overall Emojis")
else:
    plt.text(0.5, 0.5, "No emojis found", ha='center', va='center')
    plt.title("Emoji Distribution")

plt.tight_layout()
plt.savefig("sentiment_report.png")
print("✅ Report saved as 'sentiment_report.png'!")