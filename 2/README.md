# Задание 2. Генерация текста с Transformer Decoder

## Задача 1. Greedy Decoding

1. **Если запустить алгоритм несколько раз, то будут ли различаться генерации?**
   Нет, генерации будут одинаковыми, т.к. алгоритм детерминированный и всегда берём токен с макс вероятностью.

2. **Какие есть проблемы с таким подходом?**
   Для сказки: однообразные и скучные тексты без вариативности. Для JSON: зато идеально для структурированных данных, т.к. нам не нужна случайность.

## Задача 2. Sampling

1. **Если запустить алгоритм несколько раз, то будут ли различаться генерации?**
   Да, каждый раз получаем разные тексты из-за случайности сэмплирования из распределения.

2. **Какие есть проблемы с таким подходом?**
   Для сказки: иногда генерирует бред или несвязные куски. Для JSON: может нарушать формат и генерировать невалидный JSON, что критично.

## Задача 3. Sampling meets Temperature

1. **Как отличаются генерации с разными температурами?**
   - `0.001`: почти как greedy - один и тот же связный но скучный текст (как видно по логам)
   - `0.1`: мало разнообразия, стандартный сюжет про ежика находящего сокровище
   - `0.5`: норм баланс разнообразия и связности, появляется больше деталей (золотые камни)
   - `1.0`: креативнее, появляются необычные сюжеты (Сапфировые Камни, королевство Торнвульфа)
   - `10.0`: полный рандом, просто набор несвязных слов на разных языках

   Чем ниже температура, тем более предсказуемый текст. Для структурированных данных типа JSON лучше низкая температура (0.1-0.5), для творческих текстов - средняя (0.7-1.0).

## Задача 4. Nucleus Sampling

1. **Как отличаются генерации с разными параметрами?**
   - `temperature=1, top_p=0.9`: разнообразный, но осмысленный текст с необычными деталями (сюжет с дверью и сокровищем)
   - `temperature=1, top_p=0.15`: консервативнее, меньше креатива (практически как greedy)
   - `temperature=0.5, top_p=0.9`: сбалансированный результат, связный и интересный
   - `temperature=0.5, top_p=0.15`: максимально безопасный, но скучный (идентичен greedy)

2. **Помог ли nucleus sampling исправить проблемы?**
   Да, отсекает маловероятные токены, что уменьшает бред при высоких температурах, но сохраняет разнообразие, в отличие от снижения температуры.

## Задача 5. Early-Stopped Beam Search

1. **Как отличаются результаты с разными параметрами?**
   - `num_beams=1, length_penalty=1.0`: эквивалентно greedy
   - `num_beams=4, length_penalty=1.0`: более связный текст с глобально оптимальным сюжетом
   - `num_beams=4, length_penalty=0.5`: результат такой же как при length_penalty=1.0 для нашего примера
   - `num_beams=4, length_penalty=2.0`: результат не изменился для нашего примера
   - `num_beams=8, length_penalty=1.0`: более сложный сюжет с большим количеством персонажей (гриффины, пикси, мудрая сова)

   Увеличение num_beams даёт более глобально оптимальный результат, length_penalty регулирует длину (но не влияет на наш пример, т.к. EOS токен сработал).

2. **Помог ли текущий способ исправить проблемы Greedy Decoding?**
   Да, даёт более глобально оптимальные последовательности токенов с лучшим сюжетом. Beam search лучше для структурированных данных (для JSON корректно генерирует RUB вместо rubles), тогда как nucleus sampling - для творческих задач. 