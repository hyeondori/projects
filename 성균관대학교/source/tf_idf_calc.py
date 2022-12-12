# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import time

# Read recipe inputs
df = pd.read_csv('../data/skku_5years_bef_tfidf.csv')

# 빈 공간 채워야 밑에서 에러 안남
df.fillna(' ', inplace = True)

# 필요 없는 단어 제거
df['works_title'] = df['works_title'].str.replace('(sub|sup|pm|abstract|Abstract|ABSTRACT)', '')
df['works_title'] = df['works_title'].replace('(<|>|:)', '', regex = True)

df['works_abstract'] = df['works_abstract'].str.replace('(sub|sup|pm|abstract|Abstract|ABSTRACT|background|Background|BACKGROUND|using|used|results|result|study|showed)', '')
df['works_abstract'] = df['works_abstract'].replace('(<|>|:)', '', regex = True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))

nltk.download('averaged_perceptron_tagger')
abstract_extr = []
print('df.size:',df.size)
print(len(stop_words))
start_time = time.time()
for i in df.index:
    # abstract 불용어 제거
    df['works_abstract'][i] = str(df['works_abstract'][i])
    sen = re.sub('[^a-zA-Z]', ' ', df['works_abstract'][i]).lower().split()
    df['works_abstract'][i] = ' '.join([w for w in sen if w not in stop_words ])

    # title 불용어 제거
    df['works_title'][i] = str(df['works_title'][i])
    sen2 = re.sub('[^a-zA-Z]', ' ', df['works_title'][i]).lower().split()
    df['works_title'][i] =  ' '.join([w for w in sen2 if w not in stop_words])

    # title + abstract
    df['works_abstract'][i] = str(df['works_title'][i]) + ' ' + str(df['works_abstract'][i])
    # abstract 없는 데이터는 nan으로 변환됨.. -> nan 직접 없애주기
    # nanosheets...
    # 명사, 형용사만 추출
    a = nltk.pos_tag(nltk.word_tokenize(df['works_abstract'][i]))
    extr_words = []
    for word, pos in a:
        if pos in ['NN', 'JJ']:
            extr_words.append(word)
    abstract_extr.append(extr_words)
    #if i==2000 : break
print(time.time()-start_time)
#최초 : 11.139795303344727
# Memory usage
total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
print("1. RAM memory % used:", round((used_memory/total_memory) * 100, 2))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 단 - 복수 처리
wlem = nltk.WordNetLemmatizer()
print('abstract_extr length:', len(abstract_extr))
lemmatized_words = []
for word in abstract_extr:
    if type(word) is list:
        tmp_words = []
        for w in word:
            new_word = wlem.lemmatize(w)
            tmp_words.append(new_word)
        lemmatized_words.append(tmp_words)
    else:
        new_word = wlem.lemmatize(word)
        lemmatized_words.append(new_word)
# Memory usage
total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
print("2. RAM memory % used:", round((used_memory/total_memory) * 100, 2))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df['works_abstract'] = lemmatized_words
# del(result, result2, sen, sen2, abstract_noun, NN_words, a)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 단어 합치기
abstract_list = []
for i in df['works_abstract']:
    try:
        abstract = ' '.join(i)
        abstract_list.append(abstract)
    except:
        abstract = ' '
        abstract_list.append(abstract)
total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
print("3. RAM memory % used:", round((used_memory/total_memory) * 100, 2))
# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df['works_abstract'] = abstract_list
del(abstract, abstract_list)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
ab_list = df['works_abstract'].tolist()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# tf-idf 값  구하기
vectorizer = TfidfVectorizer(analyzer = 'word', ngram_range = (1,2), max_features = 100000)
# 단어 수 지정 옵션, 노말라이징
vectors = vectorizer.fit_transform(ab_list)
feature_names = vectorizer.get_feature_names_out()
dense = vectors.todense()
total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
print("8. RAM memory % used:", round((used_memory/total_memory) * 100, 2))
print("dense",dense.shape)
print("feature_names len", len(feature_names))
print("feature_names", feature_names)
#denselist = dense.tolist()
total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
print("9. RAM memory % used:", round((used_memory/total_memory) * 100, 2))
#df1 = pd.DataFrame(denselist, columns=feature_names)
df1 = pd.DataFrame(dense, columns=feature_names)
total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
print("10. RAM memory % used:", round((used_memory/total_memory) * 100, 2))


df_all = pd.DataFrame()
for i in range(len(df)):
    globals()['df_{}'.format(i)] = pd.DataFrame(df1.iloc[i]).reset_index()
    globals()['df_{}'.format(i)].columns = ['index', 'TF-IDF']
    globals()['df_{}'.format(i)] = globals()['df_{}'.format(i)][globals()['df_{}'.format(i)]['TF-IDF'] > 0].sort_values(by = ['TF-IDF'], ascending = False).reset_index(drop=True)
    globals()['df_{}'.format(i)]['id'] = df['works_id'][i]
    globals()['df_{}'.format(i)]['title'] = df['works_title'][i]
    globals()['df_{}'.format(i)]['title_abstract'] = df['works_abstract'][i]
    globals()['df_{}'.format(i)]['publication_year'] = df['works_publication_year'][i]
    df_all = pd.concat([df_all, globals()['df_{}'.format(i)]])
    del(globals()['df_{}'.format(i)])

df_all.reset_index(drop = True)
df_all = df_all[['id','title','title_abstract','publication_year','index','TF-IDF']].sort_values(by = 'TF-IDF', ascending = False)
df_all.columns = ['ID','Title','Title_Abstract','Publication_year','keyword','TF-IDF']



# Write recipe outputs
df_all.to_csv('../data/skku_5years_aft_tfidf.csv')