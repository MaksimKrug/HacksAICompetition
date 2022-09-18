import re

import numpy as np
import pandas as pd
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm


def preprocessing(data):
    # filter by "Год_поступления"
    data = data[data["Год_Поступления"] <= 2022]
    # "Пол" typos
    data["Пол"] = data["Пол"].replace(["муж", "жен"], ["Муж", "Жен"])
    # "Основания" typos
    data["Основания"] = data["Основания"].replace("ЛН", "ДН")
    # "Изучаемый_Язык"
    rep_vals = [
        "Иностранный язык (Английский)",
        "Англиийский",
        "Иностранный язык (Немецкий)",
    ]
    rep_to = ["Английский язык", "Английский язык", "Немецкий язык"]
    data["Изучаемый_Язык"] = data["Изучаемый_Язык"].replace(rep_vals, rep_to)
    # Calculate "Возраст_Зачисления"
    data["Дата_Рождения"] = pd.to_datetime(data["Дата_Рождения"]).dt.year
    data["Возраст_Зачисления"] = data["Год_Поступления"] - data["Дата_Рождения"]
    data = data[data["Возраст_Зачисления"] >= 12]
    # Calculate Возраст_Окончания_УЗ & Срок_Поступления
    data["Возраст_Окончания_УЗ"] = data["Год_Окончания_УЗ"] - data["Дата_Рождения"]
    data.loc[data["Возраст_Окончания_УЗ"] <= 15, "Возраст_Окончания_УЗ"] = np.nan
    # Общежитие
    data.loc[data["Общежитие"].notna(), "Общежитие"] = data.loc[
        data["Общежитие"].notna(), "Общежитие"
    ].astype(int)
    # Иностранец
    data.loc[data["Иностранец"].notna(), "Иностранец"] = data.loc[
        data["Иностранец"].notna(), "Иностранец"
    ].astype(int)
    # Село
    data.loc[data["Село"].notna(), "Село"] = data.loc[
        data["Село"].notna(), "Село"
    ].astype(int)
    # Наличие_Отца
    data["Наличие_Отца"] = data["Наличие_Отца"].astype(int)
    # КодФакультета
    data["КодФакультета"] = data["КодФакультета"].astype("object")
    # Опекунство
    data["Опекунство"] = data["Опекунство"].astype(int)

    """
    "Уч_Заведение"
    """
    data["Уч_Заведение"] = data["Уч_Заведение"].str.lower().copy()
    data["Уч_Заведение"] = data["Уч_Заведение"].str.replace("[^\w\s]", " ")
    rem = [
        "мбоу ",
        "фгбоу ",
        "мкоу ",
        "кгбоу ",
        "гоу ",
        "кгу ",
        "моу ",
        "кгбпоу ",
        "средняя ",
        "кгб ",
        "спо ",
        "фгоу ",
        "ноу ",
        "сош",
        "фбгоу ",
        "кгоу ",
        "кгбо ",
        "маоу ",
        "аноо ",
        "ано ",
        "соу ",
        "боу ",
        "сш ",
        "гу ",
        "фгкоу ",
        "фгаоу ",
        "чоу ",
        "гбоу ",
        "почу ",
        "бпоу ",
        "кгкп ",
        "фгбо ",
        "начоу ",
        "пу ",
        "во ",
        "впо ",
        "федеральное государственное бюджетное образовательное учреждение высшего образования ",
        "федеральное государственное бюджетное образовательное учреждение высшего профессионального образования",
    ]
    rem2 = ["им и и ползунова", "г барнаул"]
    data["Уч_Заведение"] = data["Уч_Заведение"].str.replace("|".join(rem), "")
    data["Уч_Заведение"] = data.loc[data["Уч_Заведение"].notna(), "Уч_Заведение"].apply(
        lambda x: re.sub("г\.?\s\w+", " ", x)
    )
    data["Уч_Заведение"] = data["Уч_Заведение"].str.strip()
    data["Уч_Заведение"] = data["Уч_Заведение"].replace("\s+", " ", regex=True)
    data["Уч_Заведение"] = data["Уч_Заведение"].str.replace("|".join(rem2), "")
    data["Уч_Заведение"] = data["Уч_Заведение"].str.strip()

    reps = {}
    counts = data["Уч_Заведение"].value_counts()
    candidates = counts[counts < 50].index
    for candidate in tqdm(candidates):
        temp = data.loc[
            (data["Уч_Заведение"].notna())
            & (data["Уч_Заведение"] != candidate)
            & (data["Уч_Заведение"].str.contains(candidate + " "))
        ]
        if temp.size == 0:
            continue
        rep_el = temp["Уч_Заведение"].value_counts().index[0]
        if (
            (levenshtein_distance(candidate, rep_el) / len(rep_el) <= 0.3)
            or (rep_el.startswith(candidate))
            or (rep_el.endswith(candidate))
            or (candidate.isdigit() and candidate in rep_el.split())
        ):
            reps[candidate] = rep_el

    data["Уч_Заведение"] = data.loc[
        data["Уч_Заведение"].notna(), "Уч_Заведение"
    ].replace(reps)

    """
    Страна_ПП & Страна_Родители
    """
    # replace
    rep_vals = [
        "РОССИЯ",
        "Киргизия",
        "Кыргызия",
        "Казахстан респ",
        "Кыргызская Республика",
        "Казахстан Респ",
        "Республика Казахстан",
        "Казахстан ВКО",
        "Росссия",
        "Таджикистан Респ",
        "Республика Таджикистан",
        "КИТАЙ",
        "КАЗАХСТАН",
        "КНР",
        "казахстан",
    ]
    rep_to = [
        "Россия",
        "Кыргызстан",
        "Кыргызстан",
        "Казахстан",
        "Кыргызстан",
        "Казахстан",
        "Казахстан",
        "Казахстан",
        "Россия",
        "Таджикистан",
        "Таджикистан",
        "Китай",
        "Казахстан",
        "Китай",
        "Казахстан",
    ]

    data["Страна_ПП"] = data["Страна_ПП"].replace(rep_vals, rep_to)
    data["Страна_Родители"] = data["Страна_Родители"].replace(rep_vals, rep_to)

    # nans
    data.loc[data["Страна_ПП"].isna(), "Страна_ПП"] = data.loc[
        data["Страна_ПП"].isna(), "Страна_Родители"
    ]
    # nans
    data.loc[data["Страна_Родители"].isna(), "Страна_Родители"] = data.loc[
        data["Страна_Родители"].isna(), "Страна_ПП"
    ]

    """
    Регион_ПП
    """
    data["Регион_ПП"] = data["Регион_ПП"].str.lower().copy()
    data["Регион_ПП"] = data["Регион_ПП"].str.replace("[^\w\s]", " ")
    rem = "республика|респ|провинция|пров| обл$| область$| ао$|ао "
    data["Регион_ПП"] = data["Регион_ПП"].str.replace(rem, " ")
    data["Регион_ПП"] = data["Регион_ПП"].str.strip()

    rep2pattern = {
        "алтайский край": ["алт", "алайский", "барнаул"],
        "восточно-казахстанская": ["вост", "в казах", "вко"],
        "южно-казахстанская": ["южно казахстанская"],
        "северо-казахстанская": ["северо казахстанская"],
        "западно-казахстанская": ["западно казахстанская"],
        "кемеровская область": ["кемер"],
        "якутия": ["якут"],
        "ханты-мансийский": ["манси"],
        "гуандун": ["гуан"],
        "хэйлунцзян": ["хэйл"],
        "хатлонская": ["хатлон"],
        "алматинская": ["алмат"],
        "ленинградская": ["ленинградская"],
        "ленинабадская": ["ленинабадская"],
        "синьцзян": ["синьц"],
        "свердловская": ["свердл"],
        "жалалабадская": ["жалал"],
        "новосибирская": ["новос"],
        "чеченская": ["чечен"],
        "сахалин": ["сахалин"],
        "иссыкульская": ["иссык"],
        "карагандинская": ["караган"],
        "павлодарская": ["павлод"],
        "томская": ["томск"],
        "омская": ["^омск"],
        "баткенская": ["баткен"],
        "цзянси": ["цзянси"],
        "цзянсу": ["цзянсу"],
        "нарынская": ["нар"],
        "волгоград": ["волгог"],
        "цзилинь": ["цзил"],
        "чуйская": ["чуй"],
        "бадахшанская": ["бадах"],
        "москва": ["москва"],
        "московская": ["московская", "московкая"],
        "ляолин": ["ляо"],
        "санкт-петербург": ["санк"],
        "аньхой": ["аньх"],
        "хэнань": ["нань"],
        "иркутская": ["иркутс"],
        "тыва": ["тыва"],
        "краснодарский": ["краснодар"],
        "акмолинская": ["акмолин"],
        "сычуань": ["сыч"],
        "курганская": ["курган"],
        "крым": ["крым"],
        "хакасия": ["хакас"],
        "хабаровская": ["хабаров"],
        "красноярская": ["краснояр"],
        "ненецкий": ["ненец"],
        "хубэй": ["хуб"],
        "самарская": ["самар"],
        "ганьсу": ["ганьс"],
        "лебапская": ["лебап"],
        "согдийская": ["согдий"],
        "талаская": ["талас"],
    }

    for rep, patterns in rep2pattern.items():
        for pattern in patterns:
            data.loc[
                (data["Регион_ПП"].notna()) & (data["Регион_ПП"].str.contains(pattern)),
                "Регион_ПП",
            ] = rep

    """
    Где_Находится_УЗ
    """
    data["Где_Находится_УЗ"] = data["Где_Находится_УЗ"].str.lower()

    patterns = [
        "(\w+)\sг$",
        "(\w+)\sг[\.\,]$",
        "г[\.\,]?\s(\w+)",
        "г[\.\,](\w+)",
        "(\w+)\sс$",
        "(\w+)\sрп$",
    ]

    for pattern in patterns:
        data.loc[data["Где_Находится_УЗ"].notna(), "Где_Находится_УЗ"] = data.loc[
            data["Где_Находится_УЗ"].notna(), "Где_Находится_УЗ"
        ].apply(
            lambda x: re.findall(pattern, x)[0] if re.findall(pattern, x) != [] else x
        )

    reps = {}
    counts = data["Где_Находится_УЗ"].value_counts()
    candidates = counts[counts < np.inf].index
    for candidate in tqdm(candidates):
        temp = data.loc[
            (data["Где_Находится_УЗ"].notna())
            & (data["Где_Находится_УЗ"] != candidate)
            & (data["Где_Находится_УЗ"].str.contains(candidate + " "))
        ]
        if temp.size == 0:
            continue

        rep_el = temp["Где_Находится_УЗ"].value_counts().index[0]
        if (
            (levenshtein_distance(candidate, rep_el) / len(rep_el) <= 0.3)
            or (rep_el.startswith(candidate))
            or (rep_el.endswith(candidate))
            or (candidate in rep_el.split())
        ):
            reps[candidate] = rep_el

    data["Где_Находится_УЗ"] = data.loc[
        data["Где_Находится_УЗ"].notna(), "Где_Находится_УЗ"
    ].replace(reps)

    for i in tqdm(data["Уч_Заведение"].unique()):
        if i is np.nan:
            continue
        temp = data.loc[
            (data["Уч_Заведение"] == i) & (data["Где_Находится_УЗ"].notna()),
            "Где_Находится_УЗ",
        ]
        if temp.size != 0:
            temp = temp.value_counts().index[0]

        data.loc[data["Уч_Заведение"] == i, "Где_Находится_УЗ"] = temp

    """
    Город_ПП
    """
    data["Город_ПП"] = data["Город_ПП"].str.lower()
    data["Город_ПП"] = data["Город_ПП"].str.replace("\s+", " ")
    data["Город_ПП"] = data["Город_ПП"].str.replace("-", "")
    data["Город_ПП"] = data["Город_ПП"].str.strip()
    data.loc[data["Город_ПП"] == "", "Город_ПП"] = np.nan
    data.loc[data["Город_ПП"] == " ", "Город_ПП"] = np.nan

    patterns = [
        "(\w+)\sг$",
        "(\w+)\sг[\.\,]$",
        "^г[\.\,]?\s(\w+)",
        "г[\.\,](\w+)",
        "(\w+)\sс$",
        "(\w+)\sрп$",
        "^с[\.\s]\.?\s?(\w+)",
        "(\w+)\sп$",
        "(\w+)\sс\.$",
        "(\w+)\sст$",
        "^зато\s(\w+)",
        "^(\w+)\sг",
    ]

    for pattern in patterns:
        data.loc[data["Город_ПП"].notna(), "Город_ПП"] = data.loc[
            data["Город_ПП"].notna(), "Город_ПП"
        ].apply(
            lambda x: re.findall(pattern, x)[0] if re.findall(pattern, x) != [] else x
        )

    reps = {}
    counts = data["Город_ПП"].value_counts()
    candidates = counts[counts < np.inf].index
    for candidate in tqdm(candidates):
        temp = data.loc[
            (data["Город_ПП"].notna())
            & (data["Город_ПП"] != candidate)
            & (data["Город_ПП"].str.contains(candidate + " "))
        ]
        if temp.size == 0:
            continue

        rep_el = temp["Город_ПП"].value_counts().index[0]
        if (
            (levenshtein_distance(candidate, rep_el) / len(rep_el) <= 0.3)
            or (rep_el.startswith(candidate))
            or (rep_el.endswith(candidate))
            or (candidate in rep_el.split())
        ):
            reps[candidate] = rep_el

    data["Город_ПП"] = data.loc[data["Город_ПП"].notna(), "Город_ПП"].replace(reps)

    """
    Unique to NaNs
    """
    unique_cols = ["Уч_Заведение"]
    threhs = [5]
    for idx, col in enumerate(unique_cols):
        temp = data[col].value_counts(dropna=False) <= threhs[idx]
        data.loc[data[col].isin(temp.loc[temp].index), col] = np.nan

    return data
