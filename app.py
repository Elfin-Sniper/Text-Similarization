from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pickle
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

app = Flask(__name__)

# Load model and tokenizer
model = load_model('model/text_similarity_model.h5')
with open('model/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

STOPWORDS = set(stopwords.words('english'))
max_len = 5  # Replace with your actual max_len from training

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text1 = preprocess_text(data['text1'])
    text2 = preprocess_text(data['text2'])

    # Convert texts to sequences
    seq1 = pad_sequences(tokenizer.texts_to_sequences([text1]), maxlen=max_len, padding='post')
    seq2 = pad_sequences(tokenizer.texts_to_sequences([text2]), maxlen=max_len, padding='post')

    # Make prediction
    prediction = model.predict([seq1, seq2])[0][0]
    result = "Similar" if prediction > 0.5 else "Not Similar"
    
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True))
