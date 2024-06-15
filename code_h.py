import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import time
import joblib

#reading a txt file
with open("youtube.txt","r") as f:
  text=f.read()

#calling the tokenizer
tokenizer=Tokenizer()
tokenizer.fit_on_texts([text])

# Load your trained model for next word prediction
# def load_next_word_model():
#     return load_model('yt.h5')  # Replace with your model path

model = load_model('yt.h5')


# Function to predict next word
def predict_next_word(text,model):

    # tokenize
    token_text = tokenizer.texts_to_sequences([text])[0]
    # padding
    padded_token_text = pad_sequences([token_text], maxlen=330, padding='pre')
    predicted_word=tokenizer.sequences_to_texts([[np.argmax(model.predict(padded_token_text))]])

    return predicted_word[0]

# Main Streamlit app
def main():
    st.title('Next Word Predictor')

    # Input text box for user input
    user_input = st.text_input('Enter text:', '')
    num_lines = st.slider('Select number of lines:', min_value=1, max_value=100, value=10)

    if user_input:

        for i in range(num_lines):
            
            # Predict next word
            word = predict_next_word(user_input, model)

            user_input = user_input + " " + word
            st.write(user_input)
            time.sleep(2)
        # Display predicted next word
        # st.subheader('Predicted Next Word:')
        # st.write(next_word_index)  # Display the predicted next word index or token

# Run the app
if __name__ == '__main__':
    main()
