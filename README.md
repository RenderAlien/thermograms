![Application](https://img.shields.io/badge/Application-NDT-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![Status](https://img.shields.io/badge/Status-Active%20development-orange)


# TPU-TNDT-Thermograms-processing-toolkit

Набор инструментов для обработки и анализа термограмм с поддержкой различных форматов данных и расширенными возможностями фильтрации.

## Возможности

- [x] Медианный фильтр и фильтр по гауссу.
- [x] Шум по гауссу и шум методом "перец и соль".
- [x] Гибкий и расширяемый инструментарий для обработки термограмм.
- [x] Поиск аномалий на термограммах.
- [x] Обработка серии термограмм.
- [x] Гибкий инструментарий для работы с сериями термограмм.
- [x] Поиск аномалий на сериях термограмм.

## Установка

```bash
# Клонируйте репозиторий
git clone https://github.com/RenderAlien/thermograms.git
cd thermograms

# Создайте виртуальное окружение
python -m venv therm_env

# Активируйте окружение
# Windows:
therm_env\Scripts\activate
# Linux/macOS:
source therm_env/bin/activate

# Установите зависимости
pip install -r requirements.txt
```

## Структура проекта

- thermograms.py    # Библиотека thermograms
- requirements.txt  # Зависимости
- start.ipynb       # Примеры использования

## Архитектура

Основой инструментария являются классы Tseries и Timage. Все операции с термограммами выполняются через методы этих классов. Все методы являются чистыми - то есть не изменяют исходные данные. 

### Для быстрого старта стоит ознакомиться с start.ipynb