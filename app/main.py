from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the FastAPI app
app = FastAPI()

# Define the input data schema
class TextData(BaseModel):
    text: str

# Load the tokenizer and model
tokenizer = Tokenizer()
tokenizer.fit_on_texts(pd.read_csv('Language Detection.csv')['Text'])

model = load_model('my_model.h5')

# Define the prediction endpoint
@app.post('/predict')
async def predict_language(data: TextData):
    # Convert the input text to a numerical sequence
    X = tokenizer.texts_to_sequences([data.text])
    X = pad_sequences(X, maxlen=100, padding='post')

    # Make a prediction using the loaded model
    prediction = model.predict(X)[0][0]
    if prediction >= 0.5:
        language = 'Italian'
    else:
        language = 'Not Italian'

    # Return the predicted language
    return {'language': language}