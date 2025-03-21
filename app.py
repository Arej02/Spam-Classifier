import streamlit as st
import pickle
import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('punkt')
nltk.download('stopwords')

from wordcloud import WordCloud
from collections import Counter


#Opening the required files:
with open ('model/model.pkl','rb') as file:
    model=pickle.load(file)
with open ('model/vectorizer.pkl','rb') as file:
    tfidf=pickle.load(file)

#Reading the cleaned dataframe:
df=pd.read_csv('model/spam_cleaned.csv',encoding="latin-1")

snow_ball=SnowballStemmer('english')
#Function1: Text Preprocessing:
def text_preprocessing(text):
    #Lowercasing the text:
    text=text.lower()
    #Tokenizing the document into words:
    text=nltk.word_tokenize(text)
    #Removing Special characters:
    new_text=[]
    for word in text:
        if word.isalnum():
            new_text.append(word)
    text=new_text[:]
    new_text.clear()
    
    #Removing Stop Words:
    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            new_text.append(word)

    text=new_text[:]
    new_text.clear()
    
    #Stemming:
    snow_ball=SnowballStemmer('english')
    for word in text:
        new_text.append(snow_ball.stem(word))

    return " ".join(new_text)
#Function 2:Calculate word contribution:
def get_word_contribution(text,model,tfidf):
    words=text_preprocessing(text).split()
    log_prob_spam=model.feature_log_prob_[1]
    log_prob_ham=model.feature_log_prob_[0]

    feature_names = tfidf.get_feature_names_out()
   
    word_contributions = {}
    for word in words:
        if word in feature_names:
            idx = list(feature_names).index(word)
            contribution = log_prob_spam[idx] - log_prob_ham[idx]
            word_contributions[word] = contribution
    return word_contributions

#Title:
st.title("Spam Classifier:")

#Text Input:
user_input=st.text_area('Enter the message to be classified')


#Text Button:
if st.button("Classify"):
    #1)Text Preprocessing:
    preprocessed_text=text_preprocessing(user_input)
    #2)Vectorize:
    vectorized_text=tfidf.transform([preprocessed_text])
    #3)Prediction:
    prediction=model.predict(vectorized_text)[0]
    #4)Displaying the output:
    if prediction==1:
        st.write("This is a spam email")
    else:
        st.write("This is not a spam message")
    #5)Get word Contribution:
    word_contributions=get_word_contribution(user_input,model,tfidf)
    contributions_df=pd.DataFrame(word_contributions.items(),columns=['Word','Contribution'])
    contributions_df = contributions_df.sort_values(by='Contribution', ascending=False)

    # Plot word contributions
    if not contributions_df.empty:
        st.write("### Word Contributions to Spam Probability")
        plt.figure(figsize=(10, 6))
        plt.bar(contributions_df['Word'], contributions_df['Contribution'], color=np.where(contributions_df['Contribution'] > 0, 'red', 'green'))
        plt.xticks(rotation=45)
        plt.xlabel('Word')
        plt.ylabel('Contribution to Spam Probability')
        plt.title('Word Contributions to Spam Probability')
        st.pyplot(plt)
    else:
        st.write("No significant word contributions found.")



