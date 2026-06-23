import pandas as pd

reuters = pd.read_csv("reuters_results_scraping.csv")
theguardian = pd.read_csv("theguardian_results_scraping.csv")
wsj = pd.read_csv("wsj_results_scraping.csv")
nyt = pd.read_csv("nyt_results_scraping.csv")
bloomberg = pd.read_csv("bloomberg_results_scraping.csv")
cnbc = pd.read_csv("cnbc_results_scraping.csv")
bbc = pd.read_csv("bbc_results_scraping.csv")


df_combined = pd.concat([
    reuters, 
    theguardian, 
    wsj, 
    nyt, 
    bloomberg, 
    cnbc, 
    bbc
], ignore_index=True)


print(df_combined.tail(10))

df_combined.to_csv("all_articles_merged.csv", index=False)