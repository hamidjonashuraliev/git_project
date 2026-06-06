<!-- fullWidth: false tocVisible: false tableWrap: true -->
# Data

Loyihadagi barcha ma'lumotlar shu papkada saqlanadi.\
Uch bosqichdan iborat: xom, tozalangan va boyitilgan.

---

## Raw_Data/

Yahoo Finance dan to'g'ridan-to'g'ri yuklab olingan narxlar.\
`data_load.py` ishga tushirilganda avtomatik to'ldiriladi.\


---

## Preprocessed_Data/

`preprocessing.py` tomonidan tozalangan ma'lumotlar.\
Raw_Data dan farqi:

- Bo'sh qiymatlar to'ldirilgan
- Matn ustunlar raqamlarga aylangan
- Narxlar 0 dan 1 gacha normallashtirilgan
- Close ustuni asl qiymatida — model targeti

---

## Engineered_Data/

`feature_engineering.py` tomonidan boyitilgan ma'lumotlar.\
Preprocessed_Data ga qo'shimcha yangi ustunlar yaratilgan:

- Harakatlanuvchi o'rtacha (MA)
- Narx o'zgarish foizi
- Kunlik svinglar
- Boshqa texnik ko'rsatkichlar

Model shu papkadagi ma'lumot asosida o'qitiladi.

---

## Ma'lumot oqimi

```
Raw_Data/          <- data_load.py
    ↓
Preprocessed_Data/ <- preprocessing.py
    ↓
Engineered_Data/   <- feature_engineering.py
    ↓
models/            <- train.py
```