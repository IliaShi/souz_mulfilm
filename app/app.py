import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import hstack

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from pyyoutube import Api
import isodate
from pymystem3 import Mystem
import re
import string
import tqdm
import os
import joblib
import requests
from io import BytesIO

# Пайплайн обработки датасетов
# Обработка колонки 'yt_channel_id'
class ChannelIdProcessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.fillna({'yt_channel_id':'none'}, inplace=True)
        return X

# Обработка колонки 'seconds'
class SecondsProcessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.fillna({'seconds': 0}, inplace=True)
        X = X.infer_objects(copy=False)
        return X

# Добавляем параметр text_len - длина сниппета в символах
class TexLenProcessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['text_len'] = len(X.text)
        return X

# Обработка параметра reel_name
class ReelNameProcessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.fillna({'reel_name': ''}, inplace=True)
        X['reel_name_lem'] = text_lemmatizer(X['reel_name'])
        X['reel_name'] = X['reel_name_lem'].apply(text_cleaning)
        X.drop(columns='reel_name_lem', inplace=True)
        return X


def text_lemmatizer (texts):
    """ Функция лемматизирует текст.
        Для ускорения лемматизации использует объединение в один текст по 5000 текстов
        с последующей разбивкой после лемматизации.
        Использовал код отсюда: https://habr.com/ru/articles/503420/
    Args:
        texts - Series, содержащий тексты
    Retern:
        res - список лемматизированных текстов
    """
    lemmatizer = Mystem()
    lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
    txtpart = lol(texts, 5000)
    res = []
    for txtp in txtpart:
        alltexts = ' '.join([txt + ' brbrbr ' for txt in txtp])
        words = lemmatizer.lemmatize(alltexts)
        doc = []
        for txt in words:
            if txt != '\n' and txt.strip() != '':
                if txt == 'brbrbr':
                    res.append(" ".join(doc))
                    doc = []
                else:
                    doc.append(txt)
    return res


def text_cleaning(text):
    """Функция ощищает текст от знаков препинания, цифр, эмодзи, двойных пробелов
        и приводит текст к нижнему регистру.
        Использовал код https://habr.com/ru/articles/584506/
    Args:
        text - строка текста
    Retern:
        res - очищенная строка текста
    """
    # Определяем эмодзи
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Эмоции смайликов
        "\U0001F300-\U0001F5FF"  # Символы и пиктограммы
        "\U0001F680-\U0001F6FF"  # Транспорт и символы карт
        "\U0001F700-\U0001F77F"  # Разные символы
        "\U0001F780-\U0001F7FF"  # Специальные блоки
        "\U0001F800-\U0001F8FF"  # Сюжетные символы
        "\U0001F900-\U0001F9FF"  # Дополнительные символы эмодзи
        "\U0001FA00-\U0001FA6F"  # Декоративные символы эмодзи
        "\U0001FA70-\U0001FAFF"  # Другие символы эмодзи
        "\U00002600-\U000026FF"  # Разные символы (стрелки, символы зодиака и т.д.)
        "\U00002700-\U000027BF"  # Дополнительные символы
        "\U00002000-\U000023FF"  # Символы и стрелки
        "\U0001F1E0-\U0001F1FF"  # Флаги стран
        "]+", flags=re.UNICODE)

    # Определяем знаки препинания для удаления
    list_punct = string.punctuation
    list_punct_add = '«»'
    list_punct += list_punct_add

    res = ''.join([ch if ch not in list_punct else ' ' for ch in text])         # удаляем знаки препинания
    res = ''.join([i if not i.isdigit() else ' ' for i in res])                 # удаляем цифры
    res = re.sub(r'\s+', ' ', res)                                              # удаляем двойные пробелы
    res = emoji_pattern.sub(r'', res)                                           # удаляем эмодзи

    return res.lower()                                                          # к нижнему регистру


def matrix_preparation(df):
    """ Функция преобразует датасеты в формат матрицы для обучения модели
    Args:
        df - датасет
    Returns:
        data_result - данные в формате матрицы
    """
    # Если энкодер сохранен локально - загружаем
    path = 'model/ohe_yti.pkl'
    if os.path.exists(path):
        ohe_yti = joblib.load(path)
    # Если локального файла нет - загружаем енкодер из сети
    else:
        try:
            url = 'https://huggingface.co/ILIA-Shi/cartoon_classification_souzmultfilm/resolve/main/ohe_yti.pkl'
            resp = requests.get(url)
            ohe_yti = joblib.load(BytesIO(resp.content))
            # Сохраняем энкодер локально для последующего использования
            try:
                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))
                joblib.dump(ohe_yti, path)
            except Exception as e:
                print(f'Ошибка при сохранении энкодера: {e}')
        except Exception as e:
            print(f'Ошибка при загрузке энкодера: {e}')

    # Если векторайзер сохранен локально - загружаем
    path = 'model/tfidf.pkl'
    if os.path.exists(path):
        tfidf = joblib.load(path)
    # Если локального файла нет - загружаем векторайзер из сети
    else:
        try:
            url = 'https://huggingface.co/ILIA-Shi/cartoon_classification_souzmultfilm/resolve/main/tfidf.pkl'
            resp = requests.get(url)
            tfidf = joblib.load(BytesIO(resp.content))
            # Сохраняем векторайзер локально для последующего использования
            try:
                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))
                joblib.dump(tfidf, path)
            except Exception as e:
                print(f'Ошибка при сохранении векторайзера: {e}')
        except Exception as e:
            print(f'Ошибка при загрузке векторайзера: {e}')

    # Кодирование yt_channel_id
    data_yti = ohe_yti.transform(df[['yt_channel_id']])
    # Векторизуем reel_name
    data_tfidf = tfidf.transform(df.reel_name)
    # seconds и text_len
    data_sec = df[['seconds']]
    data_tl = df[['text_len']]

    # Объединяем разреженные матрицы
    data_result = hstack([data_yti,data_tfidf, data_sec, data_tl])

    return data_result


def reels_clf(api_key:str, id_list=[]):
    """ Основная функция модуля. Обрабатывает входящие данные, получает предсказания модели, возвращает ответ.
    Args:
        api_key - API-key для получения данных с YouTube
        id_list - список ID видеороликов
    Returns:
        res - словарь: ключ - ID видеороликов, значение - предсказание модели
    """
    pd.set_option('future.no_silent_downcasting', True)
    # Проверка передаваемого API-key
    try:
        api = Api(api_key=api_key)
        _ = api.get_channel_info(channel_id="UC_x5XG1OV2P6uZZ5FSM9Ttw")  # Пробный доступ к каналу
    except Exception as e:
        print(f'Ошибка {e}')
        return [], 'Ошибка доступа к API YouTube. Проверьте API-key.'

    # Инициализируем возвращаемые значения
    res = {}

    # Датафрейм для хранения спарсенных параметров ролика
    df = pd.DataFrame(columns = ['reel_name', 'yt_channel_id', 'seconds', 'text_len'])

    # Парсим данные с ютуб по id роликов
    for id in id_list:
        try:
            video_response = api.get_video_by_id(video_id=id)
            video_data = video_response.items[0].to_dict()
            res[id] = ''
        except:
            print(f'Не удалось получить данные для ролика с id: {id}')
            res[id] = 'wrong id'
            continue
        # Добавляем пустую строчку к датафрейму
        empty_row = pd.DataFrame([[np.nan] * len(df.columns)], columns=df.columns)
        df = pd.concat([df, empty_row], ignore_index=True)
        # Заполняем последнюю строку датафрейма имеющимися данными
        ind = len(df) - 1
        df.loc[ind, 'reel_name'] = video_data['snippet']['title']
        df.loc[ind, 'text'] = video_data['snippet']['title'] + ' ' + video_data['snippet']['description']
        df.loc[ind, 'yt_channel_id'] = video_data['snippet']['channelId']
        video_info = api.get_video_by_id(video_id=id)
        df.loc[ind, 'seconds'] = isodate.parse_duration(video_info.items[0].contentDetails.duration).total_seconds()

    # Готовим даннные для модели
    pipeline_feature_prep = Pipeline([
        ('reel_id_processor', ChannelIdProcessor()),
        ('seconds_processor', SecondsProcessor()),
        ('text_len_processor', TexLenProcessor()),
        ('reel_name_processor', ReelNameProcessor())])
    df = pipeline_feature_prep.fit_transform(df)

    # Создаем матрицу признаков
    df = matrix_preparation(df)

    # Если модель сохранена локально - загружаем
    path = 'model/model.pkl'
    if os.path.exists(path):
        model = joblib.load(path)
        print(f'Модель загружена из локального пути {path}')
    # Если локального файла нет - загружаем модель из сети
    else:
        try:
            url = 'https://huggingface.co/ILIA-Shi/cartoon_classification_souzmultfilm/resolve/main/model_rf.pkl'
            print(f'Модель загружается с {url}')
            resp = requests.get(url)
            model = joblib.load(BytesIO(resp.content))
            print('Модель загружена успешно')
            # Сохраняем модель локально для последующего использования
            try:
                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))
                joblib.dump(model, path)
                print(f'Модель сохранена локально: {path}')
            except Exception as e:
                print(f'Ошибка при сохранении модели: {e}')
        except Exception as e:
            print(f'Ошибка при загрузке модели: {e}')

    # Получаем предсказания модели
    answers = model.predict(df)

    # Заполняем возвращаемый словарь
    counter = 0
    for id in id_list:
        if res[id] == '':
            res[id] = answers[counter]
            counter += 1

    return res

# Описание интерфейса приложения
# Заголовок и лого
left, _, right = st.columns([1, 0.2, 0.4])
left.title('Классификация мультфильмов на YouTube')
right.image('app/logo.gif')

# Перечисление мультпроектов
left, right = st.columns([1, 0.2])
left.write('Детектируются 44 мультипликационных проекта разных студий, например "Малышарики", "Цветняшки", "Фиксики", "Симпсоны", "Жила-была Царевна"')

# Полный перечень мультпроектов (разворачивающийся список)
if 'show_text' not in st.session_state:
    st.session_state.show_text = False

if right.button('Полный список'):
    st.session_state.show_text = not st.session_state.show_text
    if st.session_state.show_text:
        st.write('Смешарики, Чуддики, Маша и медведь, Кошечки собачки, Зебра в клеточку, Сумка, Фиксики,  \
        Бэтмен, Енотки, Черепашки Ниндзя, Свинка Пеппа, ЖилаБыла Царевна, My little pony, Чик-Чирикино, Царевны,  \
        Котик Мормотик, Буба, Малышарики, Синий трактор, Говорящий Том, Чучело-Мяучело, Оранжевая корова,  \
        Трансформеры, Пороро, Акуленок, Простоквашино, Цветняшки, Кукутики, Приключения Пети и Волка, Губка Боб,  \
        Щенячий патруль, Enchantimals (Эншантималс), Барбоскины, Симпсоны, Чебурашка, Крутиксы, Финник,  \
        Cry babies magic tears, Вспыш, Лунтик, Мини-мишки, Бременские музыканты, Паровозики Чаттингтон,  \
        Ну_погоди каникулы')

# Поле ввода
inp = st.text_input('Скопируйте в поле ввода ссылку на видеоролик и нажмите Enter')

# Выделяем из введенного текста ID видеоролика
left, right = st.columns([1, 1])
match = re.search(r'\?v=(.*?)&', inp)
match_shorts = re.search(r'shorts/(.*)', inp)
if match:
    reel_id = match.group(1)
    list_id = [reel_id]
    answer = reels_clf(st.secrets['API_KEY'], list_id)
    if answer[reel_id] == 'wrong id':
        left.header('Некорректный ID')
    else:
        right.header(answer[reel_id])
        url = f'https://i.ytimg.com/vi/{reel_id}/mqdefault.jpg'
        response = requests.get(url, timeout=10)
        left.image(response.content)
elif match_shorts:
    reel_id = match_shorts.group(1)
    list_id = [reel_id]
    answer = reels_clf(st.secrets['API_KEY'], list_id)
    right.header(answer[reel_id])
    url = f'https://i.ytimg.com/vi/{reel_id}/default.jpg'
    response = requests.get(url, timeout=10)
    left.image(response.content)
else:
    st.write('ID видео не обнаружено. проверьте корректность ввода ссылки')

# Ссылки внизу страницы
st.write('')
_1, _2, _3, _4, _5 = st.columns([1.8, 1, 1, 1, 1])
_1.write('Автор: Илья Широких.  2024')
_2.markdown('[LinkedIN](https://www.linkedin.com/in/ilia-shirokikh-a5953030a)')
_3.markdown('[Kaggle](https://www.kaggle.com/competitions/animation-reels-classification/overview)')
_4.markdown('[GitHub](https://github.com/IliaShi/souz_mulfilm)')
_5.markdown('[HuggingFace](https://huggingface.co/ILIA-Shi/cartoon_classification_souzmultfilm/tree/main)')
