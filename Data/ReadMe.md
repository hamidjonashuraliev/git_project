EDA
🗂️ 1. Raw_Data (Asl ma’lumotlar)

Bu dataset — dastlabki, o‘zgartirilmagan ma’lumotlar to‘plamidir.

Xususiyatlari:

Hech qanday tozalash (cleaning) qilinmagan
Missing value (bo‘sh qiymatlar) mavjud bo‘lishi mumkin
Noto‘g‘ri yoki keraksiz ma’lumotlar bo‘lishi ehtimoli bor
Data analizdan oldingi “real holat”

Maqsad:

Ma’lumotlar bilan dastlabki tanishish
Data strukturasi va muammolarni aniqlash
🧹 2. Preprocessed_Data (Tozalangan ma’lumotlar)

Bu bosqichda dataset tozalangan va analizga tayyor holatga keltirilgan.

Bajarilgan ishlar:

Missing value’lar bilan ishlash (drop yoki fill)
Noto‘g‘ri qiymatlarni tuzatish
Keraksiz ustunlarni olib tashlash
(masalan: player_name kabi irrelevant ustunlar)
Data type’larni to‘g‘rilash
Duplicate (takroriy) qatorlarni o‘chirish

Maqsad:

Toza va ishonchli dataset yaratish
Model uchun mos formatga keltirish
⚙️ 3. Engineered_Data (Feature Engineering qilingan data)

Bu bosqichda yangi feature (ustunlar) yaratiladi va modelni yaxshilash uchun data boyitiladi.

Bajarilgan ishlar:

Yangi feature’lar yaratish
(masalan: ratio, farq, agregatsiya)
Kategorik ma’lumotlarni encoding qilish
(Label Encoding, One-Hot Encoding)
Feature scaling
(StandardScaler, MinMaxScaler)
Feature selection (eng muhim ustunlarni tanlash)

Maqsad:

Model aniqligini oshirish
Data ichidagi yashirin patternlarni ochish
🔍 Xulosa

EDA jarayoni quyidagi 3 bosqich orqali amalga oshiriladi:

Raw_Data → ma’lumotni tushunish
Preprocessed_Data → ma’lumotni tozalash
Engineered_Data → model uchun optimal feature’lar yaratish
