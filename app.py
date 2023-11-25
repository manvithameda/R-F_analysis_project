#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from scipy.sparse import hstack
import pandas as pd  # Assuming 'pd' is imported for DataFrame operations
import pickle

fake_sample = pd.read_csv("Fake.csv",encoding='latin1')
true_sample = pd.read_csv("True.csv",encoding='latin1')

# Load the model and vectorizer files
with open('model.pkl', 'rb') as model_file, open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    nb_model = pickle.load(model_file)
    nb_vectorizers = pickle.load(vectorizer_file)
    
with open('model.pkl', 'rb') as model_file, open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    nb_model = pickle.load(model_file)
    nb_vectorizer = pickle.load(vectorizer_file)


# Function to predict the class (real or fake) for user input
def predict_article_class(nb_model, nb_vectorizer, user_input):
    # Vectorize the user input using the loaded vectorizers
    user_input_vectorized = [nb_vectorizer.transform([user_input]) for nb_vectorizer in nb_vectorizers]
    user_input_vectorized = hstack(user_input_vectorized)
    # Predict the class
    prediction = nb_model.predict(user_input_vectorized)
    return prediction[0]

# Streamlit app interface
st.title("Article Class Prediction")

user_input = st.text_area("Enter the article text here:")

if st.button("Predict"):
    if user_input:
        prediction = predict_article_class(nb_model, nb_vectorizers, user_input)
        prediction_text = 'Real' if prediction == 1 else 'Fake'
        st.write(f"The predicted class for the article is: {prediction_text}")
        if (prediction_text == "Fake"):
            st.error(prediction_text, icon="⚠️")
            
        else:
            st.success(prediction_text, icon="✅")
            st.snow()
    else:
        st.warning("Please enter some text for prediction.")
  
    

st.write("""## Sample Articles to Try:""")
  
st.write('''#### Fake News Article''')
st.write('''Click the box below and copy/paste.''')
st.dataframe(fake_sample['text'].sample(1), hide_index = True)
  
st.write('''#### Real News Article''')
st.write('''Click the box below and copy/paste.''')
st.dataframe(true_sample['text'].sample(1), hide_index = True)
          
        

