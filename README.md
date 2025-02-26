# Reviews Sentiment Analysis
This project focuses on analyzing customer reviews using Transformers (RoBERTa). The goal was to determine the sentiment of the reviews—positive, negative, or neutral—using Natural Language Processing (NLP) techniques and ML algorithms. 

![image](https://github.com/user-attachments/assets/d56c5a60-f0e9-48d1-bbdb-be911e3d061b)


The project is divided into 2 parts :

• Exploration and study part of the data and the NLP techniques.

• Web Application: multiple gradio webapps designed as tools that bring useful functionnalities based on the work done in the previous part.

## Repository Structure
• *Reviews Sentiment Analysis.ipynb*: Global exploration and study of all the reviews. Formating using NLP techniques.

• *Sentiment Analysis webapp.ipynb*: Divided into 3 tools :

1. Message sentiment retriever (Input : word / sentence. Output : sentiment and confidence score of the message.)
Gives the sentiment and confidence score of a message.

2. File converter (Input : csv file with reviews, desired sample size. Output : JSON file with reviews)
Convert a csv file into a json sample.

3. Wordcloud generator (Input : JSON file with reviews, desired sentiment. Output : wordcloud for the associated reviews and sentiment)
Generates wordcloud representations based on the selected sentiment.

## Dataset
This study was led on 33700 hotel reviews. Each entry is a different review that I wanted to analyze using a Transformer.

## Methodology
The analysis is structured as follows:

### Data Preprocessing:

**Text Cleaning**: Removing punctuation, numbers, and special characters.

**Tokenization**: Splitting text into individual words or tokens (with .

**Stop Words Removal:** Eliminating common words that do not contribute to sentiment.

**Stemming/Lemmatization:** Reducing words to their root forms.

### Tokenization and sentiment classification:

Use of pretrained RoBERTa for tokenization and to get the sentiment list.

### Feature Extraction and visualization:

Calculation of the most recurrent comments based on sentiment : calculating the importance score for each word to eliminate the words represented in all the reviews (for example 'room', 'hotel') and keep the ones related to a specific sentiment.

Bag of Words (BoW): Representing text as a frequency distribution of words.

## How to use ?

In the 'Data' folder you can find the data sources:

• *reviews.csv* is the original raw data used. It can be used to test the file converter webapp to get a review sample in JSON.

• *reviews sample 1.json* & *review sample 2.json* are two data samples that can be used to test the wordcloud generator tool.
