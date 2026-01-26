# Reviews Sentiment Analysis

[![Colab](https://img.shields.io/badge/Open%20in-Colab-blue?logo=googlecolab)](https://colab.research.google.com/github/theophile-bb/Reviews-sentiment/blob/main/Reviews_Sentiment.ipynb)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)]([https://www.kaggle.com/datasets/yelexa/spotify200](https://www.kaggle.com/datasets/datafiniti/hotel-reviews))


This repository contains the code and notebook to perfom **sentiment analysis on hotel reviews** using modern NLP tools and models. The goal is to classify reviews into sentiment categories and provide reusable functions for preprocessing, analysis, and inference.

---

## Project Structure

Reviews-sentiment/
├── 📂 data/
│   └── reviews.csv
│
├── 📂 src/
│   ├── __init__.py
│   └── utils.py
│
├── 📂 plots/
│   └── (visualizations)
│
├── 📂 notebooks/
│   └── Reviews_Sentiment.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt

---

## 📋 Prerequisites

This project requires:

- Python 3.7+
- A working Python environment (venv, conda, etc.)

---

## ⚙️ Installation

Clone the repository and install dependencies:

```
$ git clone https://github.com/theophile-bb/Reviews-sentiment.git
$ cd Reviews-sentiment
$ pip install -r requirements.txt
```
---

## Getting the data

The data used for this project are available on kaggle at this address : https://www.kaggle.com/datasets/datafiniti/hotel-reviews.

The repository contains a csv file with the data.

---

## Notebook

The main notebook Reviews_Sentiment.ipynb walks through:

- Reading and cleaning text data

- Preprocessing and tokenization

- Model inference and evaluation

- Visualization of results

- Example of predictions

---

## Visualizations

Example of visualizations made :

*Wordcloud for positive reviews*
![Wordcloud for positive reviews](plots/plot_3_20260102_183439.png)

*Evolution of reviews sentiment over time*
![Evolution of reviews sentiment of time](plots/plot_6_20260102_183439.png)

*Mean rate over time*
![Mean rate over time](plots/plot_8_20260102_183439.png)


