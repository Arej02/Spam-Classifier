import streamlit as st
import pickle
import string
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('punkt')
nltk.download('stopwords')

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns


st.markdown(
    """
    <style>
    body {
        background-color: #000000;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)



df1=pd.read_csv('spam_cleaned.csv',encoding="latin-1")
snow_ball=SnowballStemmer('english')


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
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

st.title('Spam Classifier')

input_sms=st.text_area('Enter a message to classify:')


if st.button('Classify',key="green"):
    #Preprocess:
    transformed_sms=text_preprocessing(input_sms)

    #Vectorize:
    vector_input=tfidf.transform([transformed_sms])

    #Predict:
    spam_probability=model.predict(vector_input)[0]


    #Display:
    st.write(f"**Result:** {'Spam' if spam_probability ==1 else 'Not Spam'} (Probability: {spam_probability:.2f})")

def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    st.pyplot(plt)

def plot_top_words(text, title, top_n=10):
    words = text.split()
    word_counts = Counter(words)
    top_words = word_counts.most_common(top_n)
    words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])

    plt.figure(figsize=(10, 5))
    sns.barplot(x='Word', y='Frequency', data=words_df, palette='viridis')
    plt.title(title)
    plt.xticks(rotation=45)
    st.pyplot(plt)

def plot_pie_chart(data, labels, title):
    plt.figure(figsize=(6, 6))
    plt.pie(data, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
    plt.title(title)
    st.pyplot(plt)

# Check for missing values in 'Transformed_Text'
if df1['Transformed_Text'].isnull().sum() > 0:
    df1['Transformed_Text'] = df1['Transformed_Text'].fillna('')

# Ensure all values in 'Transformed_Text' are strings
df1['Transformed_Text'] = df1['Transformed_Text'].astype(str)


# Verify the correct column name for spam/ham labels
label_column = 'Type'  # Replace with the correct column name
if label_column not in df1.columns:
    st.error(f"Column '{label_column}' not found in the dataset. Please check the column names.")
else:
    # Sidebar for visualization options
    st.sidebar.title("Visualization Options")
    visualization_option = st.sidebar.selectbox(
        "Choose a Visualization",
        ["Word Clouds", "Top 10 Words", "Spam vs. Ham Distribution"]
    )

    # Main app
    st.title("Spam Classifier Visualizations")

    if visualization_option == "Word Clouds":
        if st.button('Show Word Clouds'):
            spam_text = " ".join(df1[df1[label_column] == 1]['Transformed_Text'])
            ham_text = " ".join(df1[df1[label_column] == 0]['Transformed_Text'])

            st.write("### Word Cloud for Spam Messages")
            generate_wordcloud(spam_text, "Spam Messages")

            st.write("### Word Cloud for Ham Messages")
            generate_wordcloud(ham_text, "Ham Messages")

    elif visualization_option == "Top 10 Words":
        if st.button('Show Top 10 Words'):
            spam_text = " ".join(df1[df1[label_column] == 1]['Transformed_Text'])
            ham_text = " ".join(df1[df1[label_column] == 0]['Transformed_Text'])

            st.write("### Top 10 Words in Spam Messages")
            plot_top_words(spam_text, "Top 10 Spam Words")

            st.write("### Top 10 Words in Ham Messages")
            plot_top_words(ham_text, "Top 10 Ham Words")

    elif visualization_option == "Spam vs. Ham Distribution":
        if st.button('Show Spam vs. Ham Distribution'):
            spam_count = df1[df1[label_column] == 1].shape[0]
            ham_count = df1[df1[label_column] == 0].shape[0]

            st.write("### Spam vs. Ham Distribution")
            plot_pie_chart([spam_count, ham_count], ['Spam', 'Ham'], "Spam vs. Ham Messages")