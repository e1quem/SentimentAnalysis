import pandas as pd

df = pd.read_csv("all_articles_cleaned.csv", sep=",", engine='python')

df = df[~df['headline'].str.contains('ê', na=False)]

# Save the cleaned dataframe (optional)
df.to_csv("all_articles_cleaned.csv", index=False)


#How to: full of bullshit
# Deleted all "How to", infos related to the Amazon rain forest