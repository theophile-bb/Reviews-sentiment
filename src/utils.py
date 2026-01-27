import pandas as pd
import numpy as np
import os 
from datetime import datetime
from typing import List, Dict, Tuple, Union, Optional, Any

import nltk
nltk.download('punkt_tab')
from nltk import word_tokenize
from nltk.probability import FreqDist
import spacy
from spacy.tokens import Doc, Token
import urllib.request
from wordcloud import WordCloud
import string
import gensim.parsing.preprocessing as gp
from sklearn.preprocessing import StandardScaler

from transformers import pipeline

import matplotlib.pyplot as plt
import seaborn as sns


def data_processing(df: pd.DataFrame, columnDict: Dict[str, str]) -> pd.DataFrame:
    df = df[list(columnDict.keys())].copy()
    df.columns = list(columnDict.values())
    df.dropna(subset=['text'], inplace=True)
    return df

def data_conversion(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['date'])
    df['text'] = df['text'].astype(str)
    df['rating'] = df['rating'].round()
    return df

def text_processing(df: pd.DataFrame, textcol: str, toReplace: str) -> pd.DataFrame:
    df[textcol] = df[textcol].str.replace(toReplace, '', regex=True)
    df[textcol] = df[textcol].str.strip()
    review_df = df[df[textcol] != '']
    return review_df


def remove_punctuation(text: str) -> str:
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def process_spacy_doc(doc: Doc) -> List[str]:
    return [lem_word.lemma_ for lem_word in doc if not lem_word.is_stop]

def text_cleaning(df: pd.DataFrame, col: str) -> pd.DataFrame:
    nlp = spacy.load('en_core_web_sm')
    
    df[col] = df[col].apply(remove_punctuation).str.lower()
    df[col] = df[col].map(gp.strip_multiple_whitespaces)
    
    df[col] = [process_spacy_doc(doc) for doc in nlp.pipe(df[col], n_process=-1)]
    
    df = df[df[col].apply(lambda x: len(x) > 0)].reset_index(drop=True)
    return df


def get_sentiment(df: pd.DataFrame, col: str, nb: Optional[int] = None) -> pd.DataFrame:
    reviewList: List[str] = df[col].head(nb).tolist()
    
    model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    model = pipeline(model=model_path, tokenizer=model_path, truncation=True)
    
    batch_size = 32
    sentiments = model(reviewList, truncation=True, max_length=512, batch_size=batch_size)
    
    df_sentiment = pd.DataFrame(sentiments) 
    reviews_sentiment = pd.concat([df.head(nb).reset_index(drop=True), df_sentiment], axis=1)
    return reviews_sentiment


def get_most_common_words(df: pd.DataFrame, col: str) -> List[Tuple[str, int]]:
    fdist_list = [FreqDist(text) for text in df[col]]
    
    wfrequence = FreqDist()
    for fdist in fdist_list:
        wfrequence.update(fdist)
        
    most_common_words = wfrequence.most_common()
    return most_common_words

def compute_importance_ratio(
    reference_frequency: Union[Dict[str, int], List[Tuple[str, int]]], 
    compared_frequency: Union[Dict[str, int], List[Tuple[str, int]]], 
    threshold: float = 0.5
) -> List[Tuple[str, float]]:
    
    ref_dict = dict(reference_frequency)
    comp_dict = dict(compared_frequency)
    
    word_importance_ratio: Dict[str, float] = {
        word: (freq + 1) / (comp_dict.get(word, 0) + 1)
        for word, freq in ref_dict.items()
    }
    
    specific_words = {word: score for word, score in word_importance_ratio.items() if score > threshold}
    
    specific_words_sorted = sorted(specific_words.items(), key=lambda x: x[1], reverse=True)
    return specific_words_sorted


def over_time(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
        df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
    
    df_copy.dropna(subset=['date'], inplace=True)
    df_copy['year_month'] = df_copy['date'].dt.strftime('%Y-%m')
    
    # Calcul des pourcentages de sentiments par mois
    time_df = df_copy.groupby(['year_month', 'label']).size().unstack(fill_value=0)
    time_df = time_df.div(time_df.sum(axis=1), axis=0) * 100
    
    # Ajout des moyennes
    time_df['mean_score'] = df_copy.groupby('year_month')['score'].mean()
    time_df['mean_rate'] = df_copy.groupby('year_month')['rating'].mean()
    return time_df

def scale_data(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[cols] = scaler.fit_transform(df[cols])
    return df_scaled


def plot_wc(words: Union[Dict[str, Union[int, float]], List[Tuple[str, Union[int, float]]]]) -> plt.Figure:
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate_from_frequencies(dict(words))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    fig.tight_layout()
    return fig

def plot_pie(df: pd.DataFrame, col: str, title: str) -> plt.Figure:
    sentiment_counts = df[col].value_counts()
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        colors=sns.color_palette('pastel')
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig

def plot_graph(
    df: pd.DataFrame, 
    title: str, 
    cols: List[str] = ['negative', 'neutral', 'positive']
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 7))
    
    for col in cols:
        if col in df.columns:
            sns.lineplot(
                data=df,
                x='year_month',
                y=col,
                marker='o',
                label=col.capitalize(),
                ax=ax
            )
            
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Percentage', fontsize=12)
    ax.tick_params(axis='x', rotation=90)
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    return fig

def plot_correlation(df: pd.DataFrame, cols: List[str] = ['rating', 'score']) -> None:
    c = df[cols].corr(method='pearson')
    print(f"Correlation : ", c.iloc[0, 1])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(c, annot=True, fmt=".2f", linewidth=.5)
    plt.title("Correlation Heatmap")
    plt.show()

def save_figure_png(fig, folder: str, title: str) -> None:
    title = title.replace(' ','_')
    img_path = os.path.join(folder, f"{title}.png")
    fig.savefig(img_path, dpi=300, bbox_inches="tight")    

def save_figs(figs: List[plt.Figure], folder: str = "plots") -> None:
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, fig in enumerate(figs, 1):
        fig.savefig(
            f"{folder}/plot_{i}_{timestamp}.png",
            dpi=300,
            bbox_inches="tight"
        )
        
    print(f"✅ Saved {len(figs)} figures to {folder}/")

