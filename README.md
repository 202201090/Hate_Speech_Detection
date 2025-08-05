# Hate Speech Detection
A robust NLP project that uses Classical ML and GloVe based embeddings to classify text(tweet/comment) as Hateful or Non-Hateful

## Goal
Hate speech is a serious and growing issue on the internet. Our goal is to built a system that can reliably detect hateful content in text. To do this, we used different types of models — from basic machine learning to advanced transformer models — and combined their strengths into one powerful system that works well even in real-world situations.

## Tech-Stack & Libraries
- Data Handling: pandas, numpy, joblib, pickle
- Text Cleaning: nltk, re, html, emoji
- Visualization: matplotlib, seaborn, wordcloud
- ML Models: sklearn (Logistic Regression, Naive Bayes)
- Deployment: Hugging Face Spaces

## Files in this repo
- Hate_Speech_Detector.ipynb → Main notebook with code & models
- app.py → Web app backend logic
- hs_logreg.joblib → 	Logistic Regression model
- hs_naivebayes.joblib → 	Naive Bayes model
- tfidf_vectorizer.joblib → Saved TF-IDF vectorizer
- tokenizer.joblib → 	Tokenizer for Ensemble model

## How it works? 
1) Input text (tweet/comment/statement) 
2) We process it and run through different DL models and Transformers
3) Combine predictions using a weighted ensemble
4) Show the ouput (Hate Speech or Not Hate Speech)

## Model Scroes
| Model                      | Type                | Accuracy (%) |
| -------------------------- | ------------------- | ------------ |
| Logistic Regression        | Classical ML        | 94.08        |
| Naive Bayes                | Classical ML        | 86.83        |
| Ensemble                   | Hybrid              | **Best**     |

## Deployment
This project is live on Hugging Face Spaces using FastAPI
- Try it here: https://aaryan24-hate-speech-detector.hf.space/?text=
- Original Dataset Link: https://www.kaggle.com/datasets/waalbannyantudre/hate-speech-detection-curated-dataset/data
- Cleaned Datset Link: https://www.kaggle.com/datasets/h202201075/hate-speech?select=finalhatefull.csv

## Acknowledgements
- Kaggle
- Hugging Face
- GloVe Embeddings

## Disclaimer
This project and all accompanying materials are provided **solely for educational and research purposes**. The models and code herein are **not intended** for production use in real-world content moderation without further validation and testing. The authors **do not** assume any liability for decisions made based on the outputs of these models. Users are responsible for understanding the limitations of automated hate‑speech detection systems and for complying with all applicable laws and platform policies when deploying or using similar tools.

## License
MIT License
