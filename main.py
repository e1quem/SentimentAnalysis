import pandas as pd
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=SyntaxWarning)

df_tweets = pd.read_csv("data/Tweet/Tweet.csv")
df_ticker = pd.read_csv("data/Tweet/Company_Tweet.csv")

df_ticker['ticker_symbol'] = df_ticker['ticker_symbol'].astype(str) + ', '
df_ticker_agg = df_ticker.groupby('tweet_id', as_index=False).sum()
df_ticker_agg['ticker_symbol'] = df_ticker_agg['ticker_symbol'].str.rstrip(', ')

df = pd.merge(df_tweets, df_ticker_agg, on='tweet_id', how='left')
dates = pd.to_datetime(df['post_date'], unit='s')

print(f"Total tweets: {len(df)}")

# Removing tweets with spam keywords
keywords = ["JOIN NOW", "subscriber", "#AppAdvice", "Free course",
            "Free courses", "Competition", "free trial", "sign up",
            "join our", "free trial", "sign up", "join our", "Sign up",
            "In 14 days", "Algo Trading", "Algorithmic Trading",
            "Premium Package", "Stock Picking by Algorithms", "Top 10 Stock Picks",
            "AI-Based Algorithms""#Stocks Trend", "#Stocks Performance", "12Stocks.com",
            "ow.ly", "SUBSCRIBE", "subscribers", "#howtotrade"]

spam_pattern = "|".join(keywords)
df_spam = df[~df['body'].str.contains(spam_pattern, case=True, na=False)]
print(f"{len(df)-len(df_spam)} tweets removed after spam-keywords cleaning")

# Remove tweets with more than 4 cashtags
cashtag_regex = r'\$[a-zA-Z]{2,4}(?=\s|,|$)'
ticker_count = df['body'].str.findall(cashtag_regex).str.len()
df_cashtag = df[ticker_count <= 4].copy()
print(f"{len(df)-len(df_cashtag)} tweets removed after cashtag filtering")

df = pd.merge(df_spam, df_cashtag, how='inner')
df = df.reset_index(drop=True)

# Removing lengthy links (http://, https://, www.) from the body of the tweets
df['body'] = df['body'].str.replace(r'https?://\S+|(?:\w+\.)+\w+/\S*', '', regex=True)

dates = pd.to_datetime(df['post_date'], unit='s')



from transformers import pipeline
import torch

classifier = pipeline(
    "text-classification", 
    model="ProsusAI/finbert", 
    device=torch.device("mps"), # Apple Silicon M2 GPU
    dtype=torch.float16, 
    batch_size=512 # 16go RAM
)


from tqdm import tqdm

begin = 200000
to = 300000

df_slice = df.iloc[begin:to].copy()
tweets = df_slice['body'].fillna("").tolist()

results = []

for tweet in tqdm(tweets, desc="FinBERT"):
    out = classifier(tweet)[0]  
    results.append({
        'label': out['label'],
        'score': out['score']
    })

df_results = pd.DataFrame(results, index=df_slice.index)  
df_slice = pd.concat([df_slice, df_results], axis=1)

df_slice.to_csv(f"data/output/finBERT_{begin}_to_{to}.csv", index=True)

df.head(100)
