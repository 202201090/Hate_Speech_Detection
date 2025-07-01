# Hate Speech Detection
A robust NLP project that uses Classical ML, Deep Learning (GRU + GloVe), and Transformers (RoBERTa, Toxigen-hateBERT) to classify text(tweet/comment) as Hateful or Non-Hateful

## Team
- Aaryan Sinh Shaurya (SID: 202201075)  
- Denil Antala (SID: 202201090)
- Jami Sidhava (SID: 202201038)


## Goal
Hate speech is a serious and growing issue on the internet.
Our goal is to built a system that can reliably detect hateful content in text.
To do this, we used different types of models — from basic machine learning to advanced transformer models — and combined their strengths into one powerful system that works well even in real-world situations.

## Tech-Stack & Libraries
- Data Handling: pandas, numpy, joblib, pickle
- Text Cleaning: nltk, re, html, emoji
- Visualization: matplotlib, seaborn, wordcloud
- ML Models: sklearn (Logistic Regression, Naive Bayes, SVM)
- Deep Learning: TensorFlow, Keras (GRU + GloVe)
- Transformers: HuggingFace Transformers (RoBERTa, Toxigen-hateBERT)
- Deployment: Hugging Face Spaces
- Interface: Flask

## Files in this repo
- Dockerfile → Setup for containerized deployment
- Hate_Speech_Detector_v2.ipynb → Main notebook with code & models
- app.py → Web app backend logic
- hatespeech_detection.pdf → Project report and explanation
- hs_gru.h5 → GRU deep learning model (GloVe)
- hs_logreg.joblib → 	Logistic Regression model
- hs_naivebayes.joblib → 	Naive Bayes model
- hs_svm.joblib → 	SVM model
- index.html → 	Web UI interface (frontend)
- requirements.txt → 	List of dependencies
- tfidf_vectorizer.joblib → Saved TF-IDF vectorizer
- tokenizerpkl_gru.pkl → 	Tokenizer for GRU model

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
| SVM                        | Classical ML        | 94.52        |
| GloVe - GRU                | Deep Neural Network | 95.80        |
| RoBERTa / Toxigen-hateBERT | Transformer         | –            |
| Ensemble                   | Hybrid              | **Best**     |

## Deployment
This project is live on Hugging Face Spaces using FastAPI
- Try it here: https://aaryan24-hate-speech-detector.hf.space/?text=
- Original Dataset Link: https://www.kaggle.com/datasets/waalbannyantudre/hate-speech-detection-curated-dataset/data
- Cleaned Datset Link: https://www.kaggle.com/datasets/h202201075/hate-speech?select=finalhatefull.csv

## Acknowledgements
- Kaggle
- Hugging Face
- Transformers Library
- GloVe Embeddings

## Demo
![WhatsApp Image 2025-07-01 at 22 07 12](https://github.com/user-attachments/assets/dd79b289-6ccc-41e5-ba46-16f51f353529)
![WhatsApp Image 2025-07-01 at 22 07 39](https://github.com/user-attachments/assets/51da2eb0-3121-47c2-ac21-e55fd72e6a44)

## Disclaimer
This project and all accompanying materials are provided **solely for educational and research purposes**. The models and code herein are **not intended** for production use in real-world content moderation without further validation and testing. The authors **do not** assume any liability for decisions made based on the outputs of these models. Users are responsible for understanding the limitations of automated hate‑speech detection systems and for complying with all applicable laws and platform policies when deploying or using similar tools.

## License
MIT License
