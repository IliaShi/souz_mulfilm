import streamlit as st
import re
from reels_clf import reels_clf
import requests

# Заголовок и лого
left, _, right = st.columns([1, 0.2, 0.4])
left.title('Классификация мультфильмов на YouTube')
right.image('../app/logo.gif')

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
    answer = reels_clf(APY_KEY, list_id)
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
    answer = reels_clf(api_key, list_id)
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
