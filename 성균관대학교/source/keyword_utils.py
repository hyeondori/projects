import numpy as np
import itertools
import re

from konlpy.tag import Okt
from pexpect import ExceptionPexpect
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


SELECTED_KEYWORDS = ["사건번호", "가상계좌", "강제집행", "법적절차", "완제", "가정법원", "갚기", "경매", "계좌이체", "금융감독원", "금지", "기각", "납입", "면탈", "법무법인", "법무사", "변호사", "부동산", "부활", "사망", "상속", "선납금", "송금", "승계", "신용불량", "신청번호", "실효", "압류", "약속", "연체금", "워크아웃", "원리금", "원초본", "일시납", "임의상환", "입금", "전세자금", "전액", "접수번호", "정리", "정상화", "종결", "증명서", "지급명령", "지방법원", "판결", "판결채권", "포기", "한정상속", "할부", "할부금", "해결", "해제", "해지", "회복", "면제", "면책", "명령", "목돈", "배당", "소송", "신용회복", "재판", "결제", "망인", "매월", "비용", "취하", "프리워크", "감면", "금액", "물건지", "방문", "분납", "예금", "이달", "조정", "납부", "원금", "회수", "가능", "소득", "가압류", "특별", "매달", "변제", "분할", "결정", "문자", "회생", "대환", "상환", "초본", "승인", "개회", "파산", "내일"]
MODEL_CACHE = {}

def max_sum_sim(doc_embedding, candidate_embeddings, words, top_n, nr_candidates):
    # 문서와 각 키워드들 간의 유사도
    distances = cosine_similarity(doc_embedding, candidate_embeddings)

    # 각 키워드들 간의 유사도
    distances_candidates = cosine_similarity(candidate_embeddings, 
                                            candidate_embeddings)

    # 코사인 유사도에 기반하여 키워드들 중 상위 top_n개의 단어를 pick.
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [words[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # 각 키워드들 중에서 가장 덜 유사한 키워드들간의 조합을 계산
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]

def mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):

    # 문서와 각 키워드들 간의 유사도가 적혀있는 리스트
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)

    # 각 키워드들 간의 유사도
    word_similarity = cosine_similarity(candidate_embeddings)

    # 문서와 가장 높은 유사도를 가진 키워드의 인덱스를 추출.
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # keywords_idx = [2]
    keywords_idx = [np.argmax(word_doc_similarity)]

    # 가장 높은 유사도를 가진 키워드의 인덱스를 제외한 문서의 인덱스들
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # ==> candidates_idx = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10 ... 중략 ...]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    # 최고의 키워드는 이미 추출했으므로 top_n-1번만큼 아래를 반복.
    # ex) top_n = 5라면, 아래의 loop는 4번 반복됨.
    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # MMR을 계산
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # keywords & candidates를 업데이트
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

def find_keywords(input_doc, verbose=False):

    okt = Okt()
    tokenized_doc = okt.pos(input_doc)
    tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])

    if verbose:
        print('품사 태깅 10개만 출력 :',tokenized_doc[:10])
        print('명사 추출 :',tokenized_nouns)

    n_gram_range = (2,2)

    count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
    candidates = count.get_feature_names_out()

    if verbose:
        print('trigram 개수 :',len(candidates))
        print('trigram 다섯개만 출력 :',candidates[:5])

    if "SentenceTransformer" not in MODEL_CACHE:
        print("SentenceTransformer model is not in MODEL CACHE => Load model")
        MODEL_CACHE['SentenceTransformer'] = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_PATH)

    model = MODEL_CACHE['SentenceTransformer']
    doc_embedding = model.encode([input_doc])
    candidate_embeddings = model.encode(candidates)


    # max_sum_sim_10 = max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=5, nr_candidates=10)
    max_sum_sim_30 = max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=5, nr_candidates=30)

    # mmr_02 = mmr(doc_embedding, candidate_embeddings, candidates, top_n=5, diversity=0.2)
    mmr_07 = mmr(doc_embedding, candidate_embeddings, candidates, top_n=5, diversity=0.7)

    # if verbose:
    #     print(f"max_sum_sim_10: {max_sum_sim_10}, max_sum_sim_30: {max_sum_sim_30}, mmr_02: {mmr_02}, mmr_07: {mmr_07}")

    return candidates[:5], max_sum_sim_30, mmr_07

# Not use
# def search_selected_keyword(total_conversation):
#     keyword_list = []
#     for word in SELECTED_KEYWORDS:
#         if word in total_conversation:
#             keyword_list.append(word)
#     return keyword_list

# Not use
# def search_selected_keyword_re(total_conversation, limit_word_num=4):
#     keyword_list = []
#     for word in SELECTED_KEYWORDS:
#         if word in total_conversation:
#             keyword_list.append(word)

#             if len(keyword_list) > limit_word_num:
#                 break

#     keyword_sent_list = []
#     for word in keyword_list:

#         # sub = f'\w*\W*\w*\W*{word}\W*\w*\W*\w*'
#         sub = f'\w*\W*{word}\W*\w*'
#         for i in re.findall(sub, total_conversation, re.I):
#             i=i.strip(" .")
#             keyword_sent_list.append(word + " - " + i.replace("KBCred", "").replace("Debtor", "").replace(":", "").replace("\n", ""))

#     return keyword_list, keyword_sent_list

# Not use
# def search_selected_keyword_re_dict(total_conversation_dict, limit_word_num=4):

#     keyword_dict = {}
#     keyword_sent_dict = {}

#     for call_id, total_conversation in total_conversation_dict.items():

#         keyword_list = []
#         for word in SELECTED_KEYWORDS:
#             if word in total_conversation:
#                 keyword_list.append(word)

#                 if len(keyword_list) > limit_word_num:
#                     break

#         keyword_sent_list = []
#         for word in keyword_list:

#             # sub = f'\w*\W*\w*\W*{word}\W*\w*\W*\w*'
#             sub = f'\w*\W*{word}\W*\w*'
#             for i in re.findall(sub, total_conversation, re.I):
#                 i=i.strip(" .")
#                 keyword_sent_list.append(word + " - " + i.replace("KBCred", "").replace("Debtor", "").replace(":", "").replace("\n", ""))

#         keyword_dict[call_id] = keyword_list
#         keyword_sent_dict[call_id] = keyword_sent_list

#     return keyword_dict, keyword_sent_dict


def find_keywords_dict(total_conversation_dict):

    okt = Okt()
    n_gram_range = (2,2)

    candidates_dict = {}
    max_sum_sim_dict = {}
    mmr_dict = {}
    for call_id, input_doc in total_conversation_dict.items():

        try:
            tokenized_doc = okt.pos(input_doc)
            tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])

            count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
            candidates = count.get_feature_names_out()

            if "SentenceTransformer" not in MODEL_CACHE:
                print("SentenceTransformer model is not in MODEL CACHE => Load model")
                MODEL_CACHE['SentenceTransformer'] = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_PATH)

            model = MODEL_CACHE['SentenceTransformer']
            doc_embedding = model.encode([input_doc])
            candidate_embeddings = model.encode(candidates)

            max_sum_sim_30 = max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=5, nr_candidates=30)

            mmr_07 = mmr(doc_embedding, candidate_embeddings, candidates, top_n=5, diversity=0.7)

            candidates_dict[call_id] = list(candidates[:5])
            max_sum_sim_dict[call_id] = max_sum_sim_30
            mmr_dict[call_id] = mmr_07
        except Exception as e:
            candidates_dict[call_id] = []
            max_sum_sim_dict[call_id] = []
            mmr_dict[call_id] = []
            print("Keyword Exception", str(e))


    if "SentenceTransformer" in MODEL_CACHE:
        del MODEL_CACHE['SentenceTransformer']

    return candidates_dict, max_sum_sim_dict, mmr_dict