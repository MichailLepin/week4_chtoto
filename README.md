# MovieLens Two‑Tower Recommender (TensorFlow.js)

Простой демонстрационный проект рекомендательной системы на основе двухбашенной архитектуры (two‑tower) с вариантами: базовая модель на ID‑эмбеддингах и расширенная глубокая модель с пользовательскими и жанровыми признаками. Всё работает в браузере на TensorFlow.js, без сервера.

## Запуск

- **Требования**: любой современный браузер. Никаких сборок, Node.js не обязателен.
- **Шаги**:
  1) Клонируйте репозиторий или скачайте архив.
  2) Убедитесь, что файлы данных находятся в `data/`: `u.data`, `u.item` (опционально `u.user`).
  3) Откройте `index.html` локально (двойной клик) или через любой статический сервер.
  4) В UI нажмите «Load Data», затем «Train», затем «Test».

## Структура

- `index.html`: минимальный UI, подключение `@tensorflow/tfjs`, вывод графиков и результатов.
- `app.js`: загрузка и парсинг данных MovieLens 100K, подготовка признаков, обучение и инференс двух моделей, визуализация.
- `two-tower.js`: реализация моделей `TwoTowerModel` (базовая) и `DeepTwoTowerModel` (с признаками), обучение через in‑batch sampled softmax, вычисление скорингов.
- `data/`: файлы датасета (`u.data`, `u.item`, опционально `u.user`).

## Основная логика

- **Базовая модель (`TwoTowerModel`)**: две обучаемые таблицы эмбеддингов: пользователей `[numUsers, D]` и фильмов `[numItems, D]`. Скора вычисляется как скалярное произведение эмбеддингов. Обучение — через in‑batch sampled softmax: для батча пользователей и соответствующих фильмов строится матрица логитов `U @ I^T`, целевые — диагональ.
- **Глубокая модель (`DeepTwoTowerModel`)**: помимо ID‑эмбеддингов, есть проекция признаков пользователей (возраст, пол, one‑hot профессии) и фильмов (19 жанровых флагов) в пространство `D`, конкатенация с ID‑эмбеддингом и MLP‑слой с ReLU. Скора также через dot‑product.
- **Данные**: 
  - `u.data`: строки `userId\titemId\trating\ttimestamp`.
  - `u.item`: метаданные фильма и 19 флагов жанров в конце строки.
  - `u.user` (опционально): `userId|age|gender|occupation|zip`.
- **UI**: кнопки `Load Data`, `Train`, `Test`; график лосса базовой модели; PCA‑проекция эмбеддингов фильмов; таблицы с топ‑10 исторических и рекомендованных фильмов.

## Примечания по реализации

- Индексы для `tf.gather` всегда приводятся к `int32` (`tf.tensor1d([...], 'int32')` или `tf.cast(idx, 'int32')`).
- `tf.range` в местах, где используется как индексы, вызывается с типом `'int32'`.
- Обучение разделено на эпохи и батчи; deep‑модель обучается после baseline на тех же батчах.
- Для простоты инференса исключаются уже оценённые пользователем фильмы.

---

## Большой ПРОМПТ для генерации такой же логики с нуля

Сгенерируй минимальный, но полноценный демонстрационный проект рекомендательной системы MovieLens 100K, работающий полностью в браузере (без сервера) на TensorFlow.js. Соблюдай требования:

1) Файлы и структура:
- Создай файлы: `index.html`, `app.js`, `two-tower.js`, папку `data/` (ожидай `u.data`, `u.item`, опционально `u.user`).
- В `index.html` подключи CDN `@tensorflow/tfjs` стабильной версии и два скрипта: `two-tower.js`, `app.js`.
- Вёрстка: три кнопки (`Load Data`, `Train`, `Test`), блок статуса, два canvas для графиков (loss, PCA), область результатов.

2) Данные и парсинг (в `app.js`):
- Загрузка текстом через `fetch('data/u.data')`, `fetch('data/u.item')`, опционально `fetch('data/u.user')`.
- Парсинг:
  - `u.data`: таб‑разделённые строки: `(userId, itemId, rating, timestamp)` -> числа.
  - `u.item`: `itemId|title|...|<19 жанровых флагов>`; выдели `year` из скобок в конце `title` (если есть), в мапу фильмов сохрани `title` без года и `year`.
  - `u.user` (если есть): `userId|age|gender|occupation|zip` -> сохрани `age`, `gender`, `occupation`.
- Построй:
  - `userMap`, `itemMap` (внешние ID -> 0‑based индексы) и обратные мапы.
  - `userTopRated`: для каждого пользователя отсортируй его взаимодействия по `rating desc`, затем `timestamp desc`.
  - Список `qualifiedUsers`: пользователи с ≥20 оценками.
- Подготовь признаки:
  - Пользовательские: `age` (нормируй на `[0,1]` по максимуму), `gender` как {M=1,F=0}, one‑hot профессия по словарю профессий, итого размер `2 + numOccupations`.
  - Фильмов: 19 жанровых флагов как float.
  - Создай `userFeaturesList` `[numUsers, userFeatureDim]` и `itemFeaturesList` `[numItems, itemFeatureDim]` в порядке индексов.

3) Модели (в `two-tower.js`):
- Реализуй `class TwoTowerModel`:
  - Конструктор: `numUsers`, `numItems`, `embeddingDim`.
  - Параметры: `userEmbeddings`, `itemEmbeddings` как `tf.variable(tf.randomNormal([n, d], 0, 0.05))`.
  - `userForward(indices)`, `itemForward(indices)`: `tf.gather` с индексацией `int32` (обязательно приводи тип).
  - `score(u, i)`: `tf.sum(u * i, -1)`.
  - `trainStep(userIndices:number[], itemIndices:number[])`: внутри `tf.tidy`:
    - Преобразуй батчи в `tf.tensor1d([...], 'int32')`.
    - Вычисли `logits = U @ I^T`.
    - `labels = tf.oneHot(tf.range(0, batchSize, 1, 'int32'), batchSize)`.
    - Лосс `tf.losses.softmaxCrossEntropy(labels, logits)`; оптимизация через Adam.
  - `getUserEmbedding(userIndex:number)`: `userForward(tf.tensor1d([userIndex], 'int32')).squeeze()`.
  - `getScoresForAllItems(userEmbedding)`: `tf.dot(itemEmbeddings, userEmbedding).dataSync()`.
- Реализуй `class DeepTwoTowerModel`:
  - Конструктор: дополнительно принимает `userFeatureDim`, `itemFeatureDim`, `hiddenDim`, `userFeatures2D`, `itemFeatures2D`.
  - Параметры: ID‑эмбеддинги, веса проекций признаков `userFeatWeight`, `itemFeatWeight` и смещения; скрытые веса `userHiddenWeight`, `itemHiddenWeight` на размер выхода `embeddingDim`.
  - Храни `userFeatures` и `itemFeatures` как `tf.tensor2d` float32.
  - `userForward(indices)`, `itemForward(indices)`: `idEmb = gather(ID)`, `feat = gather(features)`, затем `featProj = relu(feat @ W + b)`, конкатенация `[idEmb, featProj]`, `hidden = relu(concat @ W_hidden + b_hidden)`.
  - `trainStep` аналогично базовой, с теми же батчами; `labels` через `tf.range(..., 'int32')`.
  - `getUserEmbedding(userIndex)`: как в базовой, через индексы `int32`.
  - `getScoresForAllItems(userEmbedding)`: вычисли все item‑вектора `itemForward(tf.range(0, numItems, 1, 'int32'))`, затем `scores = (allItemEmb @ userEmbedding.reshape([D,1])).squeeze().dataSync()`.

4) UI/Логика обучения (в `app.js`):
- Кнопки: `Load Data` -> загрузка/парсинг, подготовка признаков; `Train` -> обучение обеих моделей (baseline сначала, затем deep) по одинаковым батчам; `Test` -> выбор случайного пользователя из `qualifiedUsers`, расчёт скорингов, исключение уже оценённых фильмов, топ‑10 по убыванию.
- Графики: во время тренировки базовой модели рисовать кривую лосса на canvas; после обучения визуализировать PCA проекцию подмножества эмбеддингов фильмов.
- В статусной строке отображать прогресс и итог.

5) Ключевые технические детали и инварианты:
- Индексы для `tf.gather` ДОЛЖНЫ быть `int32`. Если вход — массив чисел, превращай в `tf.tensor1d([...], 'int32')`; если вход — тензор, делай `tf.cast(t, 'int32')`.
- В `tf.range` указывай шаг и тип: `tf.range(0, N, 1, 'int32')`, когда будешь использовать как индексы.
- Оборачивай вычисления, создающие промежуточные тензоры, в `tf.tidy`.
- Визуализация PCA допускается упрощённая (power iteration / без сторонних библиотек).
- Не используй серверные зависимости — всё должно работать из статических файлов.

6) Качество и UX:
- Код оформляй читаемо, без избыточных сокращений, с понятными именами переменных и функций.
- UI должен быть простым и понятным; таблицы с результатами выравнивай и подписывай заголовки.
- Обрабатывай ошибки загрузки данных и отображай сообщение в статусе.

Сгенерируй полный исходный код для всех указанных файлов, готовый к запуску в браузере.
