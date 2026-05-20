# Emotion-Detection-From-Text-With-LSTM

A Deep Learning Project for Detecting Emotion from Text with LSTM. It classifies emotions into 6 categories (joy, fear, sadness, anger, love, surprise) built with TensorFlow and keras and deployed as an interactive web app using streamlit

## Author:
Hamza Elsheikh (AAST Student) | LinkedIn: [Hamza-Elsheikh](www.linkedin.com/in/hamza-elsheikh)

## The Problem?
Computers don't understand human emotions naturally so emotion detection tries to identify these feelings from data such as text messages, video subtitles, tweets, etc...

## Why use LSTM?
Machine Learning models would naturally have trouble with context, word order, different meanings with the same word and long sentences. LSTM fixes that by remembering parts of a sentence and connect them to later words and being able to learn patterns.

## The Dataset
The dataset we used was "Emotions dataset for NLP" by Praveen containing three text files, test.txt, train.txt, val.txt. Theres a total of 20,000 samples each sentence is flagged with an emotion, there is 6 emotion classes (Joy, Fear, Sadness, Anger, Love, Surprise). **Important Note**: in this dataset the emotion Joy is dominating the dataset, found 9.4x more than Surprise this was handled with data augmentation, class weighting during training and F-1 score based evaluation. 

Link: [Emotions Dataset for NLP - Kaggle](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)

## Contents
```
DL_Project.ipynb # Main Project Notebook
DL_App.py # streamlit web app
------------------------------------------
Dataset Files:
train.txt
val.txt
test.txt
------------------------------------------
Model Artifacts:
lstm_scratch.h5 # trained scratch model
lstm_glove.h5 # trained G1ove model
tokenizer.pkl # saved tokenizer 
label_encoder.pkl # saved label encoder
------------------------------------------
glove.6B.100d.txt # too large to upload download from official link
```
## How to run?
**1. clone / download the repo**

[git clone]

**2. install required libraries** 

[pip install]

**3. run the main notebook**

**4. run the web app**

[streamlit run]

>**note**: make sure you have streamlit installed on your python environment before running the web app
```bash
activate env_name
streamlit run DL_App.py
```
>it is recommended to have a virtual environment available to avoid any issues in trying to run the web app
```bash
venv/Scripts/activate
cd [repo name]
-m streamlit run [path to app]
```

**bonus**:
to make a virtual environment:

## windows
```bash
python -m venv venv
venv\Scripts\activate
```


## macOS or linux
```bash
python3 -m venv venv
source venv/bin/activate
```

## Models used:
**1. LSTM From Scratch:** 
This model starts with no knowledge and then starts to read the text from left to right and right to left to recognize patterns in words. This helps in preventing overfitting by using the dropout layers and batch normalization

**2. G1ove Model:**
This model is already trained on word vectors from news articles and wikipedia pages. Instead of learning from scratch it starts with already known relationships and fine tunes them

## Web App
This is the visual app that provides a text box for the user to provide their input and choose which model to use then predicts the emotion from the input alongside a confidence score and a confidence score for the remaining classes. The web app runs on local host 8501

## Demo
<img width="1535" height="833" alt="WhatsApp Image 2026-05-16 at 9 46 51 PM" src="https://github.com/user-attachments/assets/996732b4-5dab-48c7-ab64-3559ef9741e4" />

## Acknowledgements
Dataset: [Emotions Dataset for NLP - Kaggle](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)
G1ove source: [G1ove Embeddings](https://nlp.stanford.edu/projects/glove/)
