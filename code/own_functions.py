import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from pymystem3 import Mystem
import stop_words

import time
import json
import re
import string
import tqdm

from pyyoutube import Api
import isodate


def data_review(df):
    """ Функция для первичного обзора данных. Выводит следующую информацию:
        - кол-во строк и столбцов датасета
        - кол-во явных дубликатов
        - таблицу с данными по каждому столбцу: тип данных, уникальных значений, долю уник.значений,
             кол-во пропусков, долю пропусков.
    Args:
        df - датафрейм для анализа
    Returns:
        нет.
    """
    print(f'Количество строк: {df.shape[0]} \nКоличество столбцов: {df.shape[1]} \nЯвных дубликатов: {df.duplicated().sum()}')

    res = pd.DataFrame(columns=['Имя столбца','Тип данных','Уникальных значений', 'Доля уник.знач., %',
                                'Число пропусков','Доля пропусков, %'])
    for k, i in enumerate(df.columns):
        res.loc[k] = [i,                                                    # Имя столбца
                    df.dtypes[i],                                           # Тип данных
                    len(df[i].unique()),                                    # Уникальных значений
                    round(len(df[i].unique()) / df[i].shape[0] * 100, 1),   # Доля уник.значений
                    df[i].isna().sum(),                                     # Число пропусков
                    round(df[i].isna().mean() * 100, 2)                     # Доля пропусков
                    ]
    display(res)


def describe_col_plot(df: pd.DataFrame, col: str, bins=80):
    '''Функция для визуализации распределения данных. Строит гистограмму и боксплот.
    Args:
        df - датафрейм
        col - название столбца датафрейма df
        bins - количество корзин для гистограммы
    Returns:
        Возвращаемых значений нет.
    '''
    # сброс индекса в датафрейме для boxplot
    df_ri = pd.DataFrame()
    df_ri = df.reset_index(drop=True)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 10))
    sns.histplot(data=df[col], ax=axes[0], bins=bins)  \
        .set_title(f'{col}: гистограмма распредления')
    sns.boxplot(data=df_ri, x=col, orient='h', ax=axes[1])


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
    for i, txtp in enumerate(tqdm.tqdm(txtpart, desc='Лемматизация')):
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
        Использовал код отсюда: https://habr.com/ru/articles/584506/
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


def data_enrichment(df):
    """ Функция парсит с YouTube и добавляет в датасет новые строки согласно списку
        из файла data_enrichment_dict.json. Удаляет дубликаты и строки с пропусками в 'reel_name'.
    Args:
        df - исходный датасет
    Retern:
        df - обогащенный датасет
    """
    # API youtube (изменен)
    api_key = 'AIzaSyCof87lFL4I_VvsQ0W3PsJop6bPfm*****'
    api = Api(api_key=api_key)

    # загружае словарь с id видеороликов и метками классов
    with open('data_enrichment_dict.json', 'r') as f:
        new_reels_dic = json.load(f)

    #Зафиксируем исходный размер датасета
    init_len = df.shape[0]

    # Парсим данные с ютуб по id роликов
    for k in new_reels_dic.keys():
        for i in new_reels_dic[k]:
            video_id = i
            try:
                video_response = api.get_video_by_id(video_id=video_id)
                video_data = video_response.items[0].to_dict()
            except:
                print(f'Не удалось получить данные: id {i} для {k}')
                continue
            # Добавляем пустую строчку к датафрейму
            empty_row = pd.DataFrame([[np.nan] * len(df.columns)], columns=df.columns)
            df = pd.concat([df, empty_row], ignore_index=True)
            # Заполняем последнюю строку датафрейма имеющимися данными
            ind = len(df) - 1
            df.loc[ind, 'cartoon'] = k
            df.loc[ind, 'reel_name'] = video_data['snippet']['title']
            df.loc[ind, 'yt_reel_id'] = i
            df.loc[ind, 'text'] = video_data['snippet']['title'] + ' ' + video_data['snippet']['description']
            df.loc[ind, 'yt_channel_id'] = video_data['snippet']['channelId']

            video_info = api.get_video_by_id(video_id=video_id)
            df.loc[ind, 'seconds'] = isodate.parse_duration(video_info.items[0].contentDetails.duration).total_seconds()

    # удаляем дубликаты
    df.drop_duplicates(subset='yt_reel_id', inplace=True)
    # удаляем строки с пропусками 'reel_name'
    df.dropna(subset='reel_name', inplace=True)

    print(f'Добавлено наблюдений: {df.shape[0] - init_len} \nРазмер датасета: {df.shape}')
    return df


def target_correction(df):
    """ Функция исправляет ошибки в таргете согласно словарю target_errors.json.
    Args:
        df - исходный датасет
    Retern:
        df - датасет с исправленными метками классов
    """
    # Загружаем словарь с корректными таргетами
    with open('target_errors.json', 'r') as f:
        errors_dict = json.load(f)

    for i in errors_dict.items():
        df.loc[df.yt_reel_id == i[0], 'cartoon'] = i[1]

    print(f'Исправлено меток классов: {len(errors_dict)}')
    return df


def matrix_preparation(train, test):
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
    # Засекаем время начала вычислений
    start_time = time.time()

    def log_time(stage_name, start_time):
        """ Внутренняя функция для отображения времени выполения операции.
        Args:
            stage_name - этап вычислений
            start_time - время начала вычислений
        Returns:
            end_time - время окончания этапа вычислений (старт следующего этапа)
        """
        end_time = time.time()
        print(f'{stage_name}: {end_time - start_time:.4f} секунд')
        return end_time

    # == Кодирование yt_channel_id ==
    ohe_yti = OneHotEncoder(handle_unknown = 'infrequent_if_exist', sparse_output=True)
    X_train_yti = ohe_yti.fit_transform(train[['yt_channel_id']])
    X_test_yti = ohe_yti.transform(test[['yt_channel_id']])
    start_time = log_time("Кодирование yt_channel_id", start_time)

    # == Преобразуем текст в вектор (reel_name) ==
    # Формируем словари стоп-слов
    ru_stopwords, en_stopwords = [], []
    ru_stopwords = stop_words.get_stop_words('ru')
    en_stopwords = stop_words.get_stop_words('en')
    ru_stopwords.extend(['серия', 'сериал', 'сборник', 'мультик', 'мультфильм', 'видео', 'выпуск'])
    en_stopwords.extend(['live', 'stream', 'series', 'cartoon', 'shorts', 'aren', 'can', 'couldn', 'didn',
                         'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'let', 'll', 'mustn', 're',
                         'shan', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn'])
    stopwords = ru_stopwords + en_stopwords
    # Определяем векторайзер
    tfidf = TfidfVectorizer(stop_words=stopwords, ngram_range=(1,1))
    # Векторизуем reel_name
    X_train_tfidf = tfidf.fit_transform(train.reel_name)
    start_time = log_time("Векторизация train", start_time)
    X_test_tfidf = tfidf.transform(test.reel_name)
    start_time = log_time("Векторизация test", start_time)

    X_train_sec = train[['seconds']]
    X_test_sec = test[['seconds']]

    X_train_tl = train[['text_len']]
    X_test_tl = test[['text_len']]

    # == Объединяем разреженные матрицы ==
    X_train = hstack([X_train_yti, X_train_tfidf, X_train_sec, X_train_tl])
    X_test = hstack([X_test_yti, X_test_tfidf, X_test_sec, X_test_tl])
    start_time = log_time("Объединение матриц", start_time)

    return X_train, X_test, ohe_yti, tfidf
