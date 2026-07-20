# SentimentAnalysis

This project investigates the predictive power of social media (3M Tweets) and news (11k headlines) sentiment on stock price movements of five major technology firms (Apple, Amazon, Google, Microsoft, and Tesla) over the period January 2015 to December 2019. We address the next-day and next-week stock price movement prediction task by combining textual sentiment data with traditional financial features within a unified Bayesian forecasting framework. 

Daily sentiment of tweets and headlines is computed using lexicon-based approaches (VADER, Bag of Words, TextBlob) and pre-trained language models (FinBERT, RoBERTa). Financial features include lagged returns, rolling volatility, and SPX & VIX indicators. Following Shao et al. (2025), our core model is a *Heterogeneous Dynamic Seemingly Unrelated Regression with Dynamic Linear* (HD-SURDLM) model, a Bayesian state-space framework capable of modeling multiple assets while capturing cross-sectional spillover effects. We benchmark this approach against naive (always-up), machine learning (Random Forest, Lasso, SVR) and deep learning models (MLP, RNN, LSTM, CNN-LSTM). We further design a simple long-only backtest to translate model predictions into trading decisions and compare the results against an equally-weighted buy-and-hold benchmark.

HD-SURDLM outperforms benchmarks on hit-rate, but fails to beat an always-up baseline on recall and F1, suggesting limited sentiment edge during this bull market period. However, a strategy based on the model's predictions still achieves a higher Sharpe and PnL than simple buy-and-hold. Limitations include uneven headline coverage across firms despite various scraping methods and the lack of follower-level data for tweet weighting.

*References*

Shao, Zhiqi et al. (2025). “Revisiting time-varying dynamics in stock market forecasting: A multi-source sentiment analysis approach with large language model”. In: Decision Support Systems 190, p. 114362. issn: 0167-9236. doi: 10.1016/j.dss.2024.114362.
