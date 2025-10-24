![Application](https://img.shields.io/badge/Application-NDT-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![Status](https://img.shields.io/badge/Status-Active%20development-orange)


# TPU-TNDT-Thermograms-processing-toolkit

Набор инструментов для обработки и анализа термограмм с поддержкой различных форматов данных и расширенными возможностями фильтрации.

## Возможности

- [x] Медианный фильтр и фильтр по гауссу.
- [x] Шум по гауссу и шум методом "перец и соль".
- [x] Изменение контраста изображения.
- [x] Гибкий и расширяемый инструментарий для обработки термогремм.
- [x] Поиск аномалий на термограммах.
- [ ] _Обработка серии термограмм._
- [ ] _Гибкий инструментарий для работы с сериями термограмм._
- [ ] _Поиск аномалий на сериях термограмм._

## Установка

```bash
# Клонируйте репозиторий
git clone https://github.com/your-username/thermograms.git
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

- Timage.py        # Основной класс Timage
- requirements.txt # Зависимости
- main.ipynb       # Примеры использования

## Архитектура

Основой инструментария является главная и единственная точка входа - класс Timage. Все операции с термограммами выполняются через методы этого класса. Все методы Timage возвращают новые объекты, что позволяет создавать элегантные цепочки вызовов.

## Быстрый старт

```python
from Timage import Timage
from PIL import Image

# Загрузка термограммы из файла
tim = Timage(image=Image.open("path/to/your/thermogram.png"))
```

Вместо использования Pillow можно задавать изображения напрямую через numpy array.

```python
import numpy as np

# Создание Timage из массива numpy
tim = Timage(array=np.array(...))
```

Теперь вы можете [использовать Timage для обработки термограмм](./main.ipynb).

## Основной инструментарий

Timage сам по себе предоставляет набор методов для обработки термограмм. Все методы являются чистыми - то есть не изменяют исходные данные. Ниже описаны основные инструменты работы с Timage

| Метод                                                                  | Описание                                                 |
| ---------------------------------------------------------------------- | -------------------------------------------------------- |
| `median_blur(radius:int) -> Timage`                                    | Медианный фильтр                                         |
| `gaussian_blur(blur:int = 1, radius:int = 3) -> Timage`                | Фильтр Гаусса                                            |
| `salt_and_pepper_noise(self, intensity:float) -> Timage`               | Шум методом соли и перца                                 |
| `gaussian_noise(self, mean:Int, stddev=32 -> Timage`                   | Шум по Гауссу                                            |
| `show(self, pallete:List[List[int]], contrast_level: int = 0) -> None` | Отобразить термограмму с заданными палитрой и контрастом |

