"""
Модуль reels_clf определяет принадлежность ролика с YouTube к одному из 44 мультипликационых проектов,
отслеживаемых студией.

При вызове функции reels_clf требуется передать параметры:
- api_key - API-key для получения данных с YouTube
- id_list - список ID видеороликов
Возвращаемое значение - словарь:
* ключ словаря - ID ролика
* значение словаря - результат предсказания. Возможны три варианта возвращаемого значения:
    * название проекта, к которму относится мультфильм
    * 'none' - если ролик не относится ни к одному из отслеживаемых проектов
    * 'wrong id' - если по указанному ID не удалось получить сведения о ролике на YouTube

Классы:
ChannelIdProcessor - обработка параметра yt_channel_id
SecondsProcessor - обработка параметра seconds
TexLenProcessor - обработка параметра text_len
ReelNameProcessor - обработка параметра reel_name

Функции:
text_lemmatizer - лемматизирует текст
text_cleaning - очищает текст от лишних символов
matrix_preparation - преобразует набор параметров в матрицу
reels_clf - принимает список ID роликов, парсит данные с YouTube, обрабатывает их и возвращает ответ.
"""

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
import joblib

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
        train - тренировочный датасет
        test - тестовый датасет
    Returns:
        X_train - тренировочный датасет в форме вектора
        X_test - тестовый датасет в форме вектора
        ohe_yti - энкодер для yt_channel_id
        tfidf - векторайзер для reel_name
    """
    # Загружаем энкодер и векторайзер
    try:
        ohe_yti = joblib.load('../models/final/ohe_yti.pkl')
    except Exception as e:
        print(f'Ошибка при загрузке кодировщика: {e}')
    try:
        tfidf = joblib.load('../models/final/tfidf.pkl')
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
    """ Функция принимает входящие в модуль значения, координирует код модуля и возвращает ответ.
    Args:
        api_key - API-key для получения данных с YouTube
        id_list - список ID видеороликов
    Returns:
        res - список возвращаемых значений
    """
    # Проверка передаваемого API-key
    try:
        api = Api(api_key=api_key)
        response = api.get_channel_info(channel_id="UC_x5XG1OV2P6uZZ5FSM9Ttw")  # Пример: канал Google Developers
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
            print(f'Не удалось получить данные для ролика: id {id}')
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

    # Загружаем модель
    try:
        model = joblib.load('../models/final/model_rf.pkl')
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
