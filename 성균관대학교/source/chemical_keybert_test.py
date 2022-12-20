# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
# import dataiku
import pandas as pd, numpy as np
from tqdm import tqdm
# from dataiku import pandasutils as pdu
# import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# from nltk.corpus import stopwords
# import re
# from collections import Counter
from keybert import KeyBERT
from keybert.backend import BaseEmbedder
from sentence_transformers import SentenceTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
# import os
# import time

class CustomEmbedder(BaseEmbedder):
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model

        # all available CUDA devices will be used
        self.pool = self.embedding_model.start_multi_process_pool()

    def embed(self, documents, verbose=False):

        # Run encode() on multiple GPUs
        embeddings = self.embedding_model.encode_multi_process(documents, 
                                                               self.pool)
        return embeddings


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df = pd.read_csv('/data/hyeondori/chemicel_eng_filtered.csv', encoding='utf-8')

df.fillna(' ', inplace = True)

# 필요 없는 단어 제거
df['works_title'] = df['works_title'].str.replace('(sub|sup|pm|abstract|Abstract|ABSTRACT)', '')
df['works_title'] = df['works_title'].replace('(<|>|:)', '', regex = True)

df['works_abstract'] = df['works_abstract'].str.replace('(sub|sup|pm|abstract|Abstract|ABSTRACT|background|Background|BACKGROUND|using|used|results|result|study|showed)', '')
df['works_abstract'] = df['works_abstract'].replace('(<|>|:)', '', regex = True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Create custom backend and pass it to KeyBERT
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
custom_embedder = CustomEmbedder(embedding_model=model)
kw_model = KeyBERT(model=custom_embedder)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
result = pd.DataFrame(columns=['works_id', 'works_title', 'works_publication_year', 'works_abstract', 'keyword', 'score'])

for i in tqdm(df.index):
    key_results = kw_model.extract_keywords(df['works_abstract'][i], keyphrase_ngram_range=(1, 3),
                                            use_mmr=True, diversity=0.7)
    for key_result in key_results:
        keyword, score = key_result
        result = result.append({'works_id': df.loc[i, 'works_id'],
                                'works_title': df.loc[i, 'works_title'],
                                'works_publication_year': df.loc[i, 'works_publication_year'],
                                'works_abstract': df.loc[i, 'works_abstract'],
                                'keyword': keyword,
                                'score': score},
                               ignore_index=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# result = pd.DataFrame(columns=['works_id', 'works_title', 'works_publication_year', 'works_abstract', 'keyword', 'score'])

# for i in df_tmp.index:
#     key_results = kw_model.extract_keywords(df['works_abstract'][i], keyphrase_ngram_range=(1, 3),
#                                          use_mmr=True, diversity=0.5)
#     for key_result in key_results:
#         keyword, score = key_result
#         result = result.append({'works_id': df_tmp.loc[i, 'works_id'],
#                                 'works_title': df_tmp.loc[i, 'works_title'],
#                                 'works_abstract': df_tmp.loc[i, 'works_abstract'],
#                                 'keyword': keyword,
#                                 'score': score},
#                                ignore_index=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# result = pd.DataFrame(columns=['works_id', 'works_title', 'works_publication_year', 'works_abstract', 'keyword', 'score'])d

# for i in df_tmp.index:
#     key_results = kw_model.extract_keywords(df['works_abstract'][i], keyphrase_ngram_range=(1, 3),
#                                             use_maxsum=True, nr_candidates=20, top_n=5)
#     for key_result in key_results:
#         keyword, score = key_result
#         result = result.append({'works_id': df_tmp.loc[i, 'works_id'],
#                                 'works_title': df_tmp.loc[i, 'works_title'],
#                                 'works_abstract': df_tmp.loc[i, 'works_abstract'],
#                                 'keyword': keyword,
#                                 'score': score},
#                                ignore_index=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

result.to_csv('/data/hyeondori/chemical_keybert_result1.csv', encoding='utf-8')