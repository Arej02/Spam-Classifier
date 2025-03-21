---
title: Spam Classifier
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.0"
app_file: app.py
pinned: false
---

## Introduction to the website:

A powerful and efficient spam classification web application built using **Streamlit**. This allows the user to input messages that gets categorized into either spam or ham messages. This model was trained from the dataset "SMS Spam Collection" from Kaggle which consisted of 5,574 different messages. This model uses techniques like Natural Processing Language(NLP) for text preprocessing, TF-IDF for feature selection and Bernoulli Naive Bayes for model training.

## Demo 

Checkout the live demo: [Spam Classifier]()

## Features:

1) Instant Prediction: Categorizes messages into spam or ham instantly
2) Interactive User Interface: Easy-to-use interface built in Streamlit
3) Highly Precise and Accurate: Model trained using Bernoulli Naive Based with accuracy of 0.98 and Precision of 0.99
4) Simple Deployment: Hosted on Hugging Spaces

a) What's different?

The website displays a bar plot of how much each words from the input text contributes on categorizing the message into either spam or ham. This provides a clear and intuitive way for the user to understand why the model was classified as either spam or ham.

b) How did I achieve this?

This comes from the core idea of Naive Bayes, where it makes a naive assumption that each features are independent from one another and calculates probabilities for each word independently. Since the probability can become very small as we are multiplying numerous decimals, this might lead to a problem know as underflow and to avoid this we use the log probabilities to make the contribution of each probabilities additive.

c)How can I improve?

My model does not handle out-of-vocabulary-words meaning if any words that are given beyond the training set, it will completely be ignored. This is a strong loophole as the word regardless of how important it is, is neglected. Another issue is with overlapping words, which I messaged during my EDA where words like "call" was present in both spam and ham messages which means such kinds of words provide a small or ambiguous contribution. The last issue I could see was when I received large amount of text the bar plot will clutter with too much words. 

## Technology Used:

- **Python** (Programming Language)
- **Streamlit** (Frontend Framework for Web App Deployment)
- **Scikit-Learn** (Machine Learning Library)
- **Pandas** (Data Manipulation and Analysis)
- **NumPy** (Numerical Computing)
- **Matplotlib & Seaborn** (Data Visualization)
- **NLTK (Natural Language Toolkit)** (Text Preprocessing - Stopword Removal, Tokenization)
- **SnowballStemmer** (Text Stemming)
- **WordCloud** (Visualizing Common Words)
- **Pickle** (Saving and Loading Machine Learning Models)

## Installation

1.**Clone the Repository**
