# 챗봇
from telegram.ext import CommandHandler, Updater, MessageHandler, Filters, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram import ChatAction

# 모델
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import stylecloud


# 크롤링
import requests
from bs4 import BeautifulSoup

# ENG CRAWL
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# selenium
from selenium import webdriver

# pickle
import pickle

# GUI
import PySimpleGUI as sg
import logging

# time
import time

#######################################
##### Telegram Chatbot Functions ######
#######################################

# Chatbot Function
def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="어서오세요."+'\n'+
                             '사용 방법울 보시려면 /help 를 입력하세요.')

def echo(update, context):
    echo_text = '잘못된 입력입니다' + '\n' + '사용 가능한 명령어를 보려면 /help 를 입력하세요.'
    context.bot.send_message(chat_id=update.effective_chat.id, text=echo_text)

def help_command(update, context):
    reply_text = '--------사용 방법--------' + '\n' + '/movie [영화이름] : 영화 리뷰 보고서 받기'+ '\n' + 'Ex) /movie 테넷, /movie xpspt 등'
    context.bot.send_message(chat_id=update.effective_chat.id, text=reply_text)


def buttons_callback(update, context):
    global eng_title
    global delete_message_id
    query = update.callback_query
    data = query.data

    context.bot.send_chat_action(chat_id=update.effective_chat.id,
                                 action=ChatAction.TYPING)

    if data == '2':
        context.bot.edit_message_text(text='준비한 서비스는 여기까지 입니다. 이용해 주셔서 감사합니다.',
                                      chat_id=query.message.chat_id,
                                      message_id=query.message.message_id)
    elif data == '1':
        context.bot.delete_message(chat_id=query.message.chat_id,
                                   message_id=query.message.message_id)
        context.bot.send_animation(animation=open('loading.gif', 'rb'),
                                   chat_id=query.message.chat_id)
        delete_message_id = query.message.message_id + 1

        eng_crawling_url(update, context, eng_title)    # 로튼 토마토 리뷰 분석 Start
        context.bot.send_message(text='준비한 서비스는 여기까지 입니다. 이용해 주셔서 감사합니다.',
                                 chat_id=update.effective_chat.id)

def start_bot():
    print("봇 시작합니다")
    TOKEN = '1295682694:AAEpZDUVBHLhq4YxuXrMFYvbRZs7N1YtRYE'
    updater = Updater(token=TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    start_handler = CommandHandler('start', start)
    movie_handler = CommandHandler('movie', crawling_url)
    help_handler = CommandHandler('help', help_command)
    buttons_callback_handler = CallbackQueryHandler(buttons_callback)
    echo_handler = MessageHandler(Filters.text, echo)

    dispatcher.add_handler(start_handler)
    dispatcher.add_handler(movie_handler)
    dispatcher.add_handler(help_handler)
    dispatcher.add_handler(buttons_callback_handler)
    dispatcher.add_handler(echo_handler)

    updater.start_polling(clean = True)
    updater.idle()

#########################
##### GUI Fuctions ######
#########################

# def gui():
#     layout = [[sg.Text('Bot Status: '), sg.Text('Stopped', key='status')],
#               [sg.Button('Start'), sg.Button('Stop', disabled=True), sg.Exit()]]
#     window = sg.Window('Finxter Bot Tutorial', layout)
#     while True:
#         event, _ = window.read()
#
#         if event == 'Start':
#             if updater is None:
#                 start_bot()
#             else:
#                 updater.start_polling()
#                 window.FindElement('Start').Update(disabled=True)
#                 window.FindElement('Stop').Update(disabled=False)
#                 window.FindElement('status').Update('Running')
#         if event == 'Stop':
#             updater.stop()
#             window.FindElement('Start').Update(disabled=False)
#             window.FindElement('Stop').Update(disabled=True)
#             window.FindElement('status').Update('Stopped')
#         if event in (None, 'Exit'):
#             break
#     if updater is not None and updater.running:
#         updater.stop()
#     window.close()

# def gui():
#     layout = [[sg.Text('Bot Status: '), sg.Text('Stopped', key='status')],
#               [sg.Button('Start'), sg.Button('Stop'), sg.Exit()]]
#     window = sg.Window('Finxter Bot Tutorial', layout)
#     while True:
#         event, _ = window.read()
#
#         if event == 'Start':
#             if updater is None:
#                 start_bot()
#             else:
#                 updater.start_polling()
#             window.FindElement('status').Update('Running')
#         if event == 'Stop':
#             updater.stop()
#             window.FindElement('status').Update('Stopped')
#         if event in (None, 'Exit'):
#             break
#     if updater is not None and updater.running:
#         updater.stop()
#     window.close()

###############################
##### Crawling functions ######
###############################

# Naver Review Crawling Function
def crawling_url(update, context):
    global eng_title

    context.bot.send_animation(animation=open('loading.gif', 'rb'),
                               chat_id=update.message.chat_id)

    first_delete_message_id = update.message.message_id + 1

    movie_title = update.message.text[7:]

    # 드라이버 세팅
    path = ('chromedriver.exe')
    driver = webdriver.Chrome(path)
    driver.get('https://www.naver.com')

    #
    search_box = driver.find_element_by_name('query')
    search_box.send_keys(f'영화 {movie_title}')
    search_box.submit()

    try:
        a = driver.find_elements_by_xpath('//*[@id="main_pack"]/div[1]/div[1]/div[1]/h2/a')
        driver.get(a[0].get_attribute('href'))

        req = driver.page_source
        soup = BeautifulSoup(req, 'html.parser')

        selector = '#content > div.article > div.mv_info_area > div.mv_info > h3 > a'

        links = soup.select(selector)
        c = []
        for link in links:
            c.append(link['href'])

        code = c[0][-6:]
        if '=' in code:
            code = c[0][-5:]

        selector2 = '#content > div.article > div.mv_info_area > div.mv_info > strong'

        texts = []
        links2 = soup.select(selector2)
        for link in links2:
            texts.append(link.text)
        eng_title = texts[0]
        print(eng_title)

        total_count = 200  # int(result.replace(',', ''))
        sum_score = 0
        pos_num = 0
        neg_num = 0
        pos_dict = dict()
        neg_dict = dict()

        for i in range(1, int(total_count / 10) + 1):
            url = (f'https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code={code}&type=after&page=' + str(i))
            print(f'{url} is parsing...')
            resp = requests.get(url)
            html = BeautifulSoup(resp.content, 'html.parser')
            score_result = html.find('div', {'class': 'score_result'})
            lis = score_result.findAll('li')

            for li in lis:
                review_text = li.find('p').getText()  # span id = _filtered_ment_0
                review_text = review_text.replace("관람객", "")
                review_text = review_text.strip()
                score = int(li.find('em').getText())
                sum_score += score

                tokenized_sentence = okt.pos(review_text, stem=True)  # 토큰화
                exstopw_ts = [word for word in tokenized_sentence if not word[0] in kr_stopwords]  # 불용어 제거
                exst_tok_sentence = [word[0] for word in tokenized_sentence if not word[0] in kr_stopwords]
                encoded = tokenizer.texts_to_sequences([exst_tok_sentence])  # 정수 인코딩
                pad_new = pad_sequences(encoded, maxlen=max_len)  # 패딩
                pd_score = float(model.predict(pad_new))  # 예측
                if (pd_score > 0.5):
                    pos_num += 1
                    for word in exstopw_ts:
                        if word[1] in ['Noun', 'Adjective', 'Verb']:
                            if word[0] not in pos_dict:
                                pos_dict[word[0]] = 1
                            else:
                                pos_dict[word[0]] += 1
                else:
                    neg_num += 1
                    for word in exstopw_ts:
                        if word[1] in ['Noun', 'Adjective', 'Verb']:
                            if word[0] not in neg_dict:
                                neg_dict[word[0]] = 1
                            else:
                                neg_dict[word[0]] += 1

        avg_score = sum_score / total_count
        pos_dict = dict(sorted(pos_dict.items(), reverse=True, key=lambda item: item[1]))
        neg_dict = dict(sorted(neg_dict.items(), reverse=True, key=lambda item: item[1]))

        context.bot.delete_message(chat_id = update.effective_chat.id,
                                   message_id = first_delete_message_id)

        context.bot.send_message(chat_id=update.effective_chat.id, text=f'네이버 영화리뷰 검색 결과입니다.')
        context.bot.send_message(chat_id=update.effective_chat.id, text=f'관람객 평균 평점은 {avg_score}점 입니다.')
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text=f'긍정 리뷰는 {pos_num}개로 전체 리뷰 중 {round(pos_num / (pos_num + neg_num) * 100, 2)}%이며,' + '\n' + f'부정 리뷰는 {neg_num}개로 전체 리뷰 중 {round(neg_num / (pos_num + neg_num) * 100, 2)}%입니다.')
        context.bot.send_message(chat_id=update.effective_chat.id, text=f'잠시 후에 리뷰를 요약한 이미지가 표출됩니다')
        second_delete_message_id = update.message.message_id + 5
        # PNG file create --------------------------------------------
        wcstopwords = {'영화', '보다', '되다', '있다', '없다', '아니다', '이다', '좋다', '않다', '같다',
                       '많다', '때', '것', '바', '그', '수', '점', '중', '번', '내', '거', '난', '뭐', '편',
                       '진짜'}
        for w in wcstopwords:
            if w in pos_dict:
                pos_dict.pop(w)
            if w in neg_dict:
                neg_dict.pop(w)



# stylecloud part
        stylecloud.gen_stylecloud(text=pos_dict,    # 긍정 리뷰 사전
                                  font_path='C:/Windows/Fonts/BMJUA_ttf.ttf',   # 폰트
                                  icon_name="fas fa-carrot",    # 당근
                                  palette="cartocolors.sequential.Peach_5",     # 주황~분홍
                                  background_color='black',     # 배경
                                  output_name="results/positive.png"
                                  )
        stylecloud.gen_stylecloud(text=neg_dict,
                                  font_path='C:/Windows/Fonts/BMJUA_ttf.ttf',
                                  icon_name="fas fa-bomb",
                                  palette="colorbrewer.sequential.YlGn_4",
                                  background_color='black',
                                  output_name="results/negative.png"
                                  )

        context.bot.delete_message(chat_id=update.effective_chat.id,
                                   message_id=second_delete_message_id)

        context.bot.send_photo(chat_id=update.effective_chat.id, photo=open('results/positive.png', 'rb'))
        context.bot.send_photo(chat_id=update.effective_chat.id, photo=open('results/negative.png', 'rb'))

        buttons = [[InlineKeyboardButton('로튼 토마토 반응도 보고싶어', callback_data=1)],
                   [InlineKeyboardButton('여기까지 볼래', callback_data=2)]]

        reply_markup = InlineKeyboardMarkup(buttons)

        context.bot.send_message(chat_id=update.message.chat_id, text='이어서 해당 영화의 로튼 토마토 반응도 살펴보실 수 있습니다.', reply_markup=reply_markup)

    except (requests.exceptions.MissingSchema, IndexError):
        context.bot.delete_message(chat_id = update.effective_chat.id,
                                   message_id = first_delete_message_id)
        context.bot.send_message(chat_id=update.effective_chat.id, text='잘못된 입력입니다')

# Rotten Tomatoes Review Crawling
def eng_crawling_url(update, context, eng_title):
    global delete_message_id

    path = (r"C:\pythonProject\sentimental_analysis\chromedriver.exe")
    driver = webdriver.Chrome(path)


    # google
    driver.get('https://www.google.com')

    search_box = driver.find_element_by_name('q')
    search_box.send_keys(f'{eng_title}')
    search_box.submit()

    try:
        a = driver.find_elements_by_xpath('//*[@id="kp-wp-tab-overview"]/div[1]/div[2]/div/div/div[1]/div[1]/a[2]')
        driver.get(a[0].get_attribute('href'))

        a = driver.find_elements_by_xpath('//*[@id="criticHeaders"]/a[1]')
        driver.get(a[0].get_attribute('href'))

        url = driver.current_url

        model = load_model(r'C:\pythonProject\sentimental_analysis\models\eng_model_2.h5')

        eng_pos_num=0
        eng_neg_num=0
        eng_sum_score = 0
        p_eng = list()
        n_eng = list()
        eng_pos_dict = dict()
        eng_neg_dict = dict()

        max_len = 300

        wordnet = WordNetLemmatizer()

        with open(r'C:\pythonProject\sentimental_analysis\models\tokenizer_data_eng.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)

        for i in range(1, 6):
            new_url = url + f'?type=&sort=&page={i}'
            response = requests.get(new_url)
            html = response.text.strip()

            soup = BeautifulSoup(html, 'html.parser')
            selector = '#content > div > div > div > div.review_table > div > div.col-xs-16.review_container > div.review_area > div.review_desc > div.the_review'
            links = soup.select(selector)

            for link in links:
                eng_text = link.text.strip()
                new = re.sub('[^a-zA-Z]', ' ', eng_text)
                words = new.lower().split()
                stop_words = set(stopwords.words('english'))
                meaning_words = [w for w in words if not w in stop_words]
                words = [wordnet.lemmatize(w) for w in meaning_words]
                words = [w for w in words if not w in ['spanish', 'review']]
                words = [w for w in words if len(w) > 2]
                new = tokenizer.texts_to_sequences([words])
                eng_texts_test = pad_sequences(new, maxlen=max_len)
                eng_score = float(model.predict(eng_texts_test))  # 예측
                eng_sum_score += eng_score

                ### 영어 그대로 가져올 경우
                if (eng_score > 0.5):
                    eng_pos_num += 1
                    for word in words:
                        if word not in eng_pos_dict:
                            eng_pos_dict[word] = 1
                        else:
                            eng_pos_dict[word] += 1
                else:
                    eng_neg_num += 1
                    for word in words:
                        if word not in eng_neg_dict:
                            eng_neg_dict[word] = 1
                        else:
                            eng_neg_dict[word] += 1

### 한글로 전환할 경우
            #         if (eng_score > 0.5):
            #             eng_pos_num += 1
            #             p_eng.append(eng_text)
            #         else:
            #             eng_neg_num += 1
            #             n_eng.append(eng_text)
            #
            # kr_stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '을', '를', '으로', '자', '에', '와',
            #                 '한', '하다', '스페인어']
            #
            # driver.implicitly_wait(10)
            # driver.get('https://papago.naver.com/?sk=en&tk=ko&hn=0')
            #
            # new_list_p = []
            # new_list_n = []
            #
            # search = driver.find_element_by_xpath(
            #     "/html/body/div/div/div[1]/section/div/div[1]/div[1]/div/div[3]/label/textarea").send_keys(p_eng)
            # time.sleep(2)
            # # 번역 버튼
            # button = driver.find_element_by_css_selector("#btnTranslate > span.translate_pc___2dgT8").click()
            #
            # # 번역창에 뜬 결과물 가져오기
            # result = driver.find_element_by_css_selector("#txtTarget > span").text
            #
            # # 결과물 전처리
            # tokenized_sentence = okt.pos(result, stem=True)  # 토큰화
            # exst_tok_sentence = [word[0] for word in tokenized_sentence if not word[0] in kr_stopwords]  # 불용어 제거
            #
            # new_list_p.append(exst_tok_sentence)
            #
            # driver.implicitly_wait(10)
            # driver.get('https://papago.naver.com/?sk=en&tk=ko&hn=0')
            #
            # # 검색창에 영어 문장/단어 넣기 (부정적)
            # search_n = driver.find_element_by_xpath(
            #     "/html/body/div/div/div[1]/section/div/div[1]/div[1]/div/div[3]/label/textarea").send_keys(n_eng)
            # time.sleep(2)
            # # 번역 버튼
            # button_n = driver.find_element_by_css_selector("#btnTranslate > span.translate_pc___2dgT8").click()
            #
            # # 번역창에 뜬 결과물 가져오기
            # result_n = driver.find_element_by_css_selector("#txtTarget > span").text
            #
            # # 결과물 전처리
            # tokenized_sentence_n = okt.pos(result_n, stem=True)  # 토큰화
            # exst_tok_sentence_n = [word[0] for word in tokenized_sentence_n if
            #                        not word[0] in kr_stopwords]  # 불용어 제거
            #
            # new_list_n.append(exst_tok_sentence_n)
            #
            # for i in range(len(new_list_p)):
            #     for word in new_list_p[i]:
            #         if word in eng_pos_dict:
            #             eng_pos_dict[word] += 1
            #         else:
            #             eng_pos_dict[word] = 1
            #
            # for i in range(len(new_list_n)):
            #     for word in new_list_n[i]:
            #         if word in eng_neg_dict:
            #             eng_neg_dict[word] += 1
            #         else:
            #             eng_neg_dict[word] = 1

        context.bot.delete_message(chat_id = update.effective_chat.id,
                                   message_id = delete_message_id)

        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text='로튼 토마토 검색 결과입니다')

        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text=f'긍정 리뷰는 전체 리뷰 중 {round(eng_pos_num / (eng_pos_num + eng_neg_num) * 100, 2)}%이며,' + '\n' +
                                      f'부정 리뷰는 전체 리뷰 중 {round(eng_neg_num / (eng_pos_num + eng_neg_num) * 100, 2)}%입니다.')
        context.bot.send_message(chat_id=update.effective_chat.id, text=f'잠시 후에 리뷰를 요약한 이미지가 표출됩니다')
        third_delete_message_id = update.callback_query.message.message_id + 4

        eng_pos_dict = dict(sorted(eng_pos_dict.items(), reverse=True, key=lambda item: item[1]))
        eng_neg_dict = dict(sorted(eng_neg_dict.items(), reverse=True, key=lambda item: item[1]))

# wordcloud 전 공통 단어 제거
        wcstopwords = {'film', 'movie'}
        for w in wcstopwords:
            if w in eng_pos_dict:
                eng_pos_dict.pop(w)
            if w in eng_neg_dict:
                eng_neg_dict.pop(w)

### 한글로 변환 시 필요한 stopwords
        # wcstopwords = {'영화', '보다', '되다', '있다', '없다', '아니다', '이다', '좋다', '않다', '같다',
        #                '많다', '때', '것', '바', '그', '수', ',', '.', '[', ']', '...', '인', '그것', '적',
        #                '스럽다', '더', '로', '다', '중', '인데', '에서', '곳', '가장', '일', '못', '에게',
        #                '까지', '님', '수도', '정도', '"', "'"}
        # for w in wcstopwords:
        #     if w in eng_pos_dict:
        #         eng_pos_dict.pop(w)
        #     if w in eng_neg_dict:
        #         eng_neg_dict.pop(w)

# stylecloud part
        stylecloud.gen_stylecloud(text=eng_pos_dict,
                                  font_path='C:/Windows/Fonts/BMJUA_ttf.ttf',
                                  icon_name="fas fa-thumbs-up",
                                  palette="cartocolors.sequential.Peach_5",
                                  background_color='black',
                                  output_name="results/eng_positive.png"
                                  )
        stylecloud.gen_stylecloud(text=eng_neg_dict,
                                  font_path='C:/Windows/Fonts/BMJUA_ttf.ttf',
                                  icon_name="fas fa-thumbs-down",
                                  palette="colorbrewer.sequential.YlGn_4",
                                  background_color='black',
                                  output_name="results/eng_negative.png"
                                  )

        context.bot.delete_message(chat_id=update.effective_chat.id, message_id=third_delete_message_id)
        context.bot.send_photo(chat_id=update.effective_chat.id, photo=open('results/eng_positive.png', 'rb'))
        context.bot.send_photo(chat_id=update.effective_chat.id, photo=open('results/eng_negative.png', 'rb'))

    except (IndexError, AssertionError):
        context.bot.delete_message(chat_id=update.effective_chat.id,
                                   message_id=delete_message_id)
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text='로튼 토마토에 해당 영화의 리뷰가 등록되어 있지 않습니다!')

############################
##### 기본 Model 생성 ######
############################

with open('models/tokenizer_data.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

threshold = 3
total_cnt = len(tokenizer.word_index)  # 단어의 수
rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if (value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

model = load_model('models/model_1.h5')

checkpoint_path = 'models/ckpt/cp.ckpt'
model.load_weights(checkpoint_path)

vocab_size = total_cnt - rare_cnt + 2

# 형태소 분석기
okt = Okt()
kr_stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
max_len = 30

#################
##### main ######
#################
logging.basicConfig(format='%(levelname)s - %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)
updater = None

####################
##### .exe on ######
####################
start_bot()
