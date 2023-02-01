# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
from tqdm import tqdm
from multiprocessing import freeze_support
from keybert import KeyBERT
from keybert.backend import BaseEmbedder
from sentence_transformers import SentenceTransformer
import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')

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

def lemmatize(model, words):
    word_list = words.split()
    lemmatized_words = []
    for word in word_list:
        new_word = model.lemmatize(word)
        lemmatized_words.append(new_word)
    result = ' '.join(lemmatized_words)

    return result


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df = pd.read_csv('/data/hyeondori/data/works_lv0_Engineering_python.csv', encoding='utf-8')

df.fillna(' ', inplace = True)

# 필요 없는 단어 제거
df['works_title'] = df['works_title'].str.replace('(sub|sup|pm|abstract|Abstract|ABSTRACT)', '')
df['works_title'] = df['works_title'].replace('(<|>|:)', '', regex = True)

df['works_abstract'] = df['works_abstract'].str.replace('(sub|sup|pm|abstract|Abstract|ABSTRACT|background|Background|BACKGROUND|using|used|results|result|study|showed)', '')
df['works_abstract'] = df['works_abstract'].replace('(<|>|:)', '', regex = True)

df['works_abstract'] = df['works_title'] + ' ' + df['works_abstract']

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Create custom backend and pass it to KeyBERT
if __name__ == '__main__':
    freeze_support()
    
    # KeyBert Model
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    custom_embedder = CustomEmbedder(embedding_model=model)
    kw_model = KeyBERT(model=custom_embedder)
    
    # lemmatizer
    wlem = nltk.WordNetLemmatizer()

    n = 258
    i = 0
    loop = 5000
    data = df

    while n <= len(data) // loop:
        result = pd.DataFrame(columns=['works_id',
                                    #    'works_concepts',
                                       'works_title',
                                       'works_publication_year',
                                       'works_abstract',
                                       'keyword',
                                       'score'])
        if n != len(data) // loop:
            for i in tqdm(range(loop)):
                key_results = kw_model.extract_keywords(data['works_abstract'][n * loop + i], keyphrase_ngram_range=(1, 2),
                                                        use_mmr=True, diversity=0.6, top_n=5, stop_words='english')
                for key_result in key_results:
                    keyword, score = key_result
                    keyword = lemmatize(wlem, keyword)
                    
                    result = result.append({'works_id': df.loc[n * loop + i, 'works_id'],
                                            # 'works_concepts': df.loc[n * loop + i, 'works_concepts'],
                                            'works_title': df.loc[n * loop + i, 'works_title'],
                                            'works_publication_year': df.loc[n * loop + i, 'works_publication_year'],
                                            'works_abstract': df.loc[n * loop + i, 'works_abstract'],
                                            'keyword': keyword,
                                            'score': score},
                                            ignore_index=True)            
        else:
            for i in tqdm(range(len(data) % loop)):
                key_results = kw_model.extract_keywords(df['works_abstract'][n * loop + i], keyphrase_ngram_range=(1, 2),
                                                        use_mmr=True, diversity=0.6, top_n=5, stop_words='english')
                for key_result in key_results:
                    keyword, score = key_result
                    keyword = lemmatize(wlem, keyword)

                    result = result.append({'works_id': df.loc[n * loop + i, 'works_id'],
                                            # 'works_concepts': df.loc[n * loop + i, 'works_concepts'],
                                            'works_title': df.loc[n * loop + i, 'works_title'],
                                            'works_publication_year': df.loc[n * loop + i, 'works_publication_year'],
                                            'works_abstract': df.loc[n * loop + i, 'works_abstract'],
                                            'keyword': keyword,
                                            'score': score},
                                            ignore_index=True)
        n += 1
        result.to_csv('/data/hyeondori/keybert_result/engineering_keybert_result/engineering_keybert_result_{}.csv'.format(n), encoding='utf-8', index=False)