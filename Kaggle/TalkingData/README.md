# TalkingData AdTracking Fraud Detection Challenge

Fraud risk is everywhere, but for companies that advertise online, click fraud can happen at an overwhelming volume, resulting in misleading click data and wasted money. Ad channels can drive up costs by simply clicking on the ad at a large scale. With over 1 billion smart mobile devices in active use every month, China is the largest mobile market in the world and therefore suffers from huge volumes of fradulent traffic.

[TalkingData](https://www.talkingdata.com/), China’s largest independent big data service platform, covers over 70% of active mobile devices nationwide. They handle 3 billion clicks per day, of which 90% are potentially fraudulent. Their current approach to prevent click fraud for app developers is to measure the journey of a user’s click across their portfolio, and flag IP addresses who produce lots of clicks, but never end up installing apps. With this information, they've built an IP blacklist and device blacklist.

While successful, they want to always be one step ahead of fraudsters and have turned to the Kaggle community for help in further developing their solution. In their 2nd competition with Kaggle, you’re challenged to build an algorithm that predicts whether a user will download an app after clicking a mobile app ad. To support your modeling, they have provided a generous dataset covering approximately 200 million clicks over 4 days!

In this project, the challenged is to build an algorithm that predicts whether a user will download an app after clicking a mobile app ad. The evaluation metric will be auc-roc.

Some points noted:

- All features are encoded, which seems convenient to us, but it seems we can do less feature engineering at this point.
- This is highly imbalanced dataset, with true download rate of 0.251% for 1 million user records.
- Data is of very large size, 7.54 GB for training set (200 million observations). Will train a random 10k observations with local laptop, then train on Google Cloud.
- Shrink data by remove year/month for time column
- Shrink data by convert int64 to int32

# Ranking & LB Score

## Final Submission Score

| Rank | Public LB Score | Private LB Score |
|------|-----------------|------------------|
| 212  | 0.9812383 | 0.9820713 |
