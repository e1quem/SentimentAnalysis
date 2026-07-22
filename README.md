# SentimentAnalysis

This project investigates the predictive power of social media (3M tweets) and news (11k headlines) sentiment on stock price movements of five major technology firms (Apple, Amazon, Google, Microsoft, and Tesla) over the period January 2015 to December 2019. We address the next-day and next-week stock price movement prediction task by combining textual sentiment data extracted via NLP techniques with traditional financial features within a unified Bayesian forecasting framework. 

Daily sentiment of tweets and headlines is computed using lexicon-based approaches (VADER, Bag of Words, TextBlob) and pre-trained large language models (FinBERT, RoBERTa). Financial features include lagged returns, rolling volatility, and SPX & VIX indicators. Following Shao et al. (2025), our core model is a *Heterogeneous Dynamic Seemingly Unrelated Regression with Dynamic Linear* (HD-SURDLM) model, a Bayesian state-space framework capable of modeling multiple assets while capturing cross-sectional spillover effects. We benchmark this approach against naive (always-up), machine learning (Random Forest, Lasso, SVR) and deep learning models (MLP, RNN, LSTM, CNN-LSTM). We further design a simple long-only backtest to translate model predictions into trading decisions and compare the results against an equally-weighted buy-and-hold benchmark.

HD-SURDLM outperforms benchmarks on hit-rate, but fails to beat an always-up baseline on recall and F1, suggesting limited sentiment edge during this bull market period. However, a strategy based on the model's predictions still achieves a higher Sharpe and PnL than simple buy-and-hold. Limitations include uneven headline coverage across firms despite various scraping methods and the lack of follower-level data for tweet weighting.

*References*

Araci, Dogu (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. arXiv: 1908.10063 [cs.CL]. url: https://arxiv.org/abs/1908.10063.

Bacco, Luca et al. (2024). “Investigating Stock Prediction Using LSTM Networks and Sentiment Analysis of Tweets Under High Uncertainty: A Case Study of North American and European Banks”. In: IEEE Access 12, pp. 122239–122248. doi: 10.1109/ACCESS.2024.3450311.

Bollen, Johan, Huina Mao, and Xiaojun Zeng (2011). “Twitter mood predicts the stock market”. In:
Journal of Computational Science 2.1, pp. 1–8. issn: 1877-7503. doi: 10.1016/j.jocs.2010.12.007.
url: https://www.sciencedirect.com/science/article/pii/S187775031100007X.

Gite, S. et al. (2021). “Explainable stock prices prediction from financial news articles using sentiment analysis”. In: PeerJ Computer Science 7, e340. doi: 10.7717/peerj-cs.340.

Gupta, Tapas, Shridev Devji, and Ashish Kumar Tripathi (2025). “Investigating the impact of sentiments on stock market using digital proxies: Current trends, challenges, and future directions”. In: Expert Systems with Applications 285, p. 127864. issn: 0957-4174. doi: 10.1016/j.eswa.2025.127864. url: https://www.sciencedirect.com/science/article/pii/S0957417425014861.

Nguyen, Thien Hai and Kiyoaki Shirai (July 2015). “Topic Modeling based Sentiment Analysis on Social
Media for Stock Market Prediction”. In: Proceedings of the 53rd Annual Meeting of the Associa-
tion for Computational Linguistics and the 7th International Joint Conference on Natural Language
Processing (Volume 1: Long Papers). Ed. by Chengqing Zong and Michael Strube. Beijing, China:
Association for Computational Linguistics, pp. 1354–1364. doi: 10.3115/v1/P15-1131. url: https:
//aclanthology.org/P15-1131/.

Shao, Zhiqi et al. (2025). “Revisiting time-varying dynamics in stock market forecasting: A multi-source sentiment analysis approach with large language model”. In: Decision Support Systems 190, p. 114362. issn: 0167-9236. doi: 10 . 1016 / j . dss . 2024 . 114362. url: https://www.sciencedirect.com/science/article/pii/S0167923624001957.
