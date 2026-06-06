<!-- fullWidth: false tocVisible: false tableWrap: true -->
# src Folder

## Maqsadi

Bu papka loyihaning asosiy kodlarini saqlash uchun ishlatiladi.\
Bu yerda reusable funksiyalar, helperlar, model logikasi va boshqa asosiy modul fayllar bo‘ladi.

## Nima bor

- `train.py` — modelni o‘qitish uchun.
- `evaluate.py` — modelni test setda baholash uchun.
- `predict.py` — yangi data ustida prediction qilish uchun.
- `preprocess.py` — raw data bo‘lsa, uni tozalash uchun.

## Loyihadagi roli

`src` papkasi kodni tartibli saqlashga yordam beradi.\
Bu loyiha kattalashsa ham kodni oson boshqarish imkonini beradi.

## Qanday ishlatiladi

Agar loyiha tuzilmasi to‘g‘ri sozlangan bo‘lsa, bu papkadagi modullar boshqa fayllar tomonidan import qilinadi.\
Masalan:

```python
from scripts.train import load_data
```

\

## Eslatma

Agar `clean_data.csv` allaqachon tayyor bo‘lsa, `preprocess.py` shart emas.\
Bunday holatda bevosita `train.py`, `evaluate.py` va `predict.py` ishlatiladi.