import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
import pickle
import re
import nltk
nltk.download('stopwords')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Lambda
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
data = {
    "text1": [
        "How are you?",
        "What is your name?",
        "Hello, how are you?",
        "Where is the nearest restaurant?",
        "Can you help me?"
    ],
    "text2": [
        "How do you do?",
        "Tell me your name.",
        "Hi, how are you doing?",
        "Where can I find food?",
        "I need assistance."
    ],
    "label": [1, 1, 1, 1, 1]  # 1 means similar, 0 means not similar
}

df = pd.DataFrame(data)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    return text

df["text1"] = df["text1"].apply(preprocess_text)
df["text2"] = df["text2"].apply(preprocess_text)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["text1"].tolist() + df["text2"].tolist())

vocab_size = len(tokenizer.word_index) + 1
max_len = max(df["text1"].apply(lambda x: len(x.split())).max(), df["text2"].apply(lambda x: len(x.split())).max())

X1 = pad_sequences(tokenizer.texts_to_sequences(df["text1"]), maxlen=max_len, padding='post')
X2 = pad_sequences(tokenizer.texts_to_sequences(df["text2"]), maxlen=max_len, padding='post')
Y = np.array(df["label"])
X1_train, X1_test, X2_train, X2_test, Y_train, Y_test = train_test_split(X1, X2, Y, test_size=0.2, random_state=42)
embedding_dim = 50

# Shared LSTM Model
input_text1 = Input(shape=(max_len,))
input_text2 = Input(shape=(max_len,))

embedding_layer = Embedding(vocab_size, embedding_dim, input_length=max_len)

lstm = LSTM(50)

encoded_text1 = lstm(embedding_layer(input_text1))
encoded_text2 = lstm(embedding_layer(input_text2))

# Compute Similarity
lambda_layer = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
merged_vector = lambda_layer([encoded_text1, encoded_text2])

# Fully Connected Layer
dense = Dense(10, activation="relu")(merged_vector)
output = Dense(1, activation="sigmoid")(dense)

model = Model(inputs=[input_text1, input_text2], outputs=output)
model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

model.summary()
history = model.fit(
    [X1_train, X2_train], Y_train,
    batch_size=4,
    epochs=50,
    validation_data=([X1_test, X2_test], Y_test)
)
def predict_similarity(text1, text2):
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)

    seq1 = pad_sequences(tokenizer.texts_to_sequences([text1]), maxlen=max_len, padding='post')
    seq2 = pad_sequences(tokenizer.texts_to_sequences([text2]), maxlen=max_len, padding='post')

    prediction = model.predict([seq1, seq2])[0][0]
    return "Similar" if prediction > 0.5 else "Not Similar"

print(predict_similarity("Hello, how are you?", "Hi, how are you doing?"))
print(predict_similarity("What is your name?", "Tell me your name."))

