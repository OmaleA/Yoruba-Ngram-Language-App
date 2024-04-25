import streamlit as st
import pickle
from ngram_model import YorubaNgram

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
model = bigram_model if model_option == 'Bigram' else trigram_model

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
    # Check if the input word is likely to be Yoruba
    if not all(char in "abcdefghijklmnopqrstuvwxyzàèéìíòóùúẹọ̣́" for char in input_word.lower()):
        st.error("Please enter a Yoruba word. English words or words with special characters are not supported.")
    else:
        # Get new predictions
        predictions = get_predictions(input_word)
        # Display the predictions
        if predictions:
            st.write('Predictions:')
            st.write(predictions)
        else:
            st.info("No predictions found for the entered word. Try a different Yoruba word.")
