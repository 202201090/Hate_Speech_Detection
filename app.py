import nltk
# Prepend your bundled path so NLTK finds it first:
nltk.data.path.insert(0, "nltk_data")
import streamlit as st
import joblib
import re, string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ——— Load Artifacts ———
vectorizer   = joblib.load("tfidf_vectorizer.joblib")
logreg_model = joblib.load("hs_logreg.joblib")
nb_model     = joblib.load("hs_naivebayes.joblib")
nn_model     = load_model("embeddings_NN.h5")
tokenizer    = joblib.load("tokenizer.joblib")

# Hyperparameters
max_seq_len      = 100
ensemble_weights = (0.2, 0.2, 0.6)  # adjust if you’ve tuned
threshold        = 0.5              # adjust if you’ve threshold‑tuned

# ——— Text Cleaner ———
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
def clean_text(text: str) -> str:
    # replicate your notebook’s cleaning
    text = text.lower()
    text = re.sub(r'\[.*?\]|https?://\S+|www\.\S+|<.*?>+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]|\n|\w*\d\w*", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t)
              for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

# ——— Streamlit UI ———
st.title("🛡️ Hate Speech Detector")
st.write("Enter a comment and click **Analyze** below.")

user_input = st.text_area("Your Comment", height=150)
if st.button("Analyze") and user_input.strip():
    # 1) Clean
    cleaned = clean_text(user_input)

    # 2) TF–IDF models
    tfidf_vec = vectorizer.transform([cleaned])
    p_lr = logreg_model.predict_proba(tfidf_vec)[:,1][0]
    p_nb = nb_model.predict_proba(tfidf_vec)[:,1][0]

    # 3) Neural net
    seq = tokenizer.texts_to_sequences([cleaned])
    seq_pad = pad_sequences(seq, maxlen=max_seq_len, padding="post")
    p_nn = float(nn_model.predict(seq_pad)[0])

    # 4) Ensemble
    w1, w2, w3 = ensemble_weights
    p_ens = w1*p_lr + w2*p_nb + w3*p_nn
    label = "🚫 Hate Speech" if p_ens >= threshold else "✅ Not Hate Speech"

    # 5) Display
    st.subheader("Model Probabilities")
    st.write(f"• Logistic Regression: **{p_lr:.4f}**")  
    st.write(f"• Naive Bayes        : **{p_nb:.4f}**")
    st.write(f"• Neural Network     : **{p_nn:.4f}**")
    st.markdown("---")
    st.write(f"**Ensemble Score**    : **{p_ens:.4f}** (threshold = {threshold})")
    st.markdown(f"### {label}")
