# ResNLS

Code for [ResNLS: An Improved Model for Stock Price Forecasting](https://arxiv.org/abs/2312.01020v2), accepted by Computational Intelligence 2023.

Stock prices forecasting has always been a challenging task. Although many research projects adopt machine learning and deep learning algorithms to address the problem, few of them pay attention to the varying degrees of dependencies between stock prices. 
In this paper we introduce a hybrid model that improves stock price prediction by emphasizing the dependencies between adjacent stock prices. 
The proposed model, ResNLS, is mainly composed of two neural architectures, ResNet and LSTM. 
ResNet serves as a feature extractor to identify dependencies between stock prices across time windows, while LSTM analyses the initial time-series data with the combination of dependencies which considered as residuals. 
Our experiment reveals that when the stock price data for the previous 5 consecutive trading days is used as the input, the performance is optimal. 
Furthermore, the proposed model outperforms vanilla ResNet and LSTM models in terms of prediction accuracy. 
It also demonstrates a 20% improvement over the current state-of-the-art baselines.

![image](https://github.com/Yuanzhe-Jia/ResNLS/assets/104203996/0c159c6c-3dd4-451c-aa4d-2277cc8a7ae5)
