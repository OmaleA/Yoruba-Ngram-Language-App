import streamlit as st
import pickle
from ngram_model import YorubaNgram


# Assuming YorubaNgram class is defined with a method Predict that takes the last word(s) and returns predictions
# Create an instance of the class
ngram_instance = YorubaNgram()

# Load your models (assuming they are saved as .pkl files)
bigram_model = ngram_instance.load_from_file('models/yoruba_bigram.pkl')
trigram_model = ngram_instance.load_from_file('models/yoruba_trigram.pkl')

# Set up your Streamlit interface
st.title('Yoruba N-Gram Language Model Prediction')

# User selects the n-gram model
model_option = st.selectbox('Choose the N-Gram model', ['Bigram', 'Trigram'])

# Initialize the model based on the selection
model = None
if model_option == 'Bigram':
    model = bigram_model
else:
    model = trigram_model

# Function to get predictions based on the last word(s)
def get_predictions(last_words):
    # Assuming the Predict method returns a list of tuples (word, probability)
    predictions_with_prob = model.Predict(last_words)
    # We only want the words, not the probabilities
    return [word for word, _ in predictions_with_prob]

# Text input for the word
input_word = st.text_input('Enter a word to get predictions', key='input_word')

# If the user enters a word, get new predictions
if input_word:
    # Get new predictions
    predictions = get_predictions(input_word)
    # Display the predictions
    if predictions:
        st.write('Predictions:')
        st.write(predictions)
