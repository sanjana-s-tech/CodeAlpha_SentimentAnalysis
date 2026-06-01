import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import os

# Create outputs folder
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# Load dataset
df = pd.read_csv(r"C:\Users\s sanjana\OneDrive\Desktop\CodeAlpha_SentimentAnalysis\reviews.csv")

# Function to classify sentiment
def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df["Sentiment"] = df["Review"].apply(get_sentiment)

# Display results
print(df)

# Save output
df.to_csv("outputs/sentiment_results.csv", index=False)

# Count sentiments
sentiment_counts = df["Sentiment"].value_counts()

print("\nSentiment Counts:")
print(sentiment_counts)

# Visualization
plt.figure(figsize=(6,6))
sentiment_counts.plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("Sentiment Analysis Results")
plt.ylabel("")
plt.tight_layout()

plt.savefig("outputs/sentiment_chart.png")
plt.show()

print("\nTask Completed Successfully!")