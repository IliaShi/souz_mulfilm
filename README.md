# Классификация видеороликов на YouTube

## **Заказчик**  
Компания "Союзмультфильм". Основной вид деятельности Студии "Союзмультфильм" - производство контента, анимационных фильмов.  

**Описание проекта**  
Для оценки популярности проектов студия мониторит рынок и собирает статистику. И очень важно агрегировать статистику именно по проектам, а не отдельным роликам. Поэтому ролики классифицируются по принадлежности к проектам. Необходимо создать автоматизированное решение по определению принадлежности ролика к тому или иному проекту на основе анализа текстового описания видеороликов и другой доступной информации. 

## **Цель**  
Предложить решение для классификации роликов по проектам.
1. Максимизировать метрику f1
2. Создать модуль (класс, функцию) для определения проекта: 
    * на вход подается список уникальных идентификаторов yt_reel_id
    * на выходе словарь {‘yt_reel_id’: идентификатор, ‘cartoon’: название_проекта_или_none}
    * для роликов, которые не входят ни в один проект, в поле cartoon должна возвращаться строка ‘none’
  
## Структура репозитория:

| #    | Наименование файла                | Описание   |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1.   | [README.md](https://github.com/IliaShi/donor_search/blob/main/README.md) | Представлена основная информация по проекту и его результатах   |
| 2.   |  | Тетрадка с основным решением |
| 3.   | [data_preparation.ipynb](https://github.com/IliaShi/donor_search/blob/main/data_preparation.ipynb) | Тетрадка для подготовки датасета из исходных фотографий   |
| 4.   | [experiments.ipynb](https://github.com/IliaShi/donor_search/blob/main/experiments.ipynb) | Тетрадка с экспериментами и поиском решения   |
| 5.   | [own_functions.py](https://github.com/IliaShi/donor_search/blob/main/own_functions.py) | Собственные функции для файла rotation_angle_detection  |
| 6.   | [requirements.txt](https://github.com/IliaShi/donor_search/blob/main/requirements.txt) | Список всех библиотек и их версии, необходимых для установки в виртуальной среде для запуска кода проекта   |
| 7.   | [Dockerfile](https://github.com/IliaShi/donor_search/blob/main/app/Dockerfile) | Докер-файл для запуска приложения
| 8.   | [requirements.txt](https://github.com/IliaShi/donor_search/blob/main/app/requirements.txt) | Список всех библиотек и их версии, необходимых для установки в виртуальной среде для запуска приложения в Докере |
| 9.   | [app.py](https://github.com/IliaShi/donor_search/blob/main/app/scr/app.py) |Скрипт для запуска приложения |
|10.   | [model.py](https://github.com/IliaShi/donor_search/blob/main/app/scr/app.py) |Запуск сохраненной модели лучшего решения |


**Описание данных**
|Параметр|Описание|
|---|---|
|date|дата, когда ролик появился на ютубе  |
|reel_name|название ролика  |
|yt_reel_id  |уникальный идентификатор ролика на ютубе  |
|cartoon |название проекта, целевая переменная  |
|url |ссылка на ролик (включает идентификатор)  |
|text |текст сниппета, включает название ролика и описание  |
|seconds | длительность|
|is_shorts | вертикальные видеоролики продолжительностью <60сек  |
|broadcast | лайвы, прямые эфиры|
|yt_channel_id |идентификатор ютуб канала  |
|yt_channel_name |название ютуб канала  |
|yt_ch_url |ссылка на ютуб канал  |
|yt_channel_type |тип канала (Мультфильмы, Детские, Блогеры, Shorts…)  |
|flag_closed |ютуб канал закрыт, если 1  |
|international |метка международных каналов, каналов на иностранном языке (переведенный контент)  |
|language| язык|
