# Переобучение

Для обычной отправки переобучение не нужно: выбранные веса входят в репозиторий. Быстрый `replay` воспроизводит именно эти веса и их контрольные предсказания.

Для обучения нужны исходные `train.csv`, `human_boxes_train.csv`, `train_frames.csv`, `train_context.csv`, `train_stream_predictions.csv` и изображения TRAIN/контекста. Дополнительные ручные TRAIN-аннотации включены:

- `annotations/person_boxes.csv`: исходные люди и проверенные дополнительные люди на train-контексте; скрытые границы часто приблизительные, что отмечено в `extent_inferred`.
- `annotations/semantic_labels.csv`: явные метки person/clothes; ambiguous исключаются. Метка error не используется как замена разметке одежды.

```bash
uv sync --extra vision
uv run python training/train.py --data-dir /path/to/DATA_DIR --output outputs/retrained
uv run ded predict --root outputs/retrained --data-dir /path/to/DATA_DIR --output outputs/retrained_submission.csv
```

Можно отдельно перестроить `--kind tabular`, `clothing`, `multi`, `person` или `tiles`; остальные включённые веса тогда копируются из исходного комплекта. Новые SHA-256 записываются в конфигурацию выходного комплекта. Повторное обучение на другом устройстве не обещает битового совпадения нейросетевых весов.

Для локального контроля целый UTC-день исключается из **всех** обучаемых компонентов и их warm start:

```bash
uv run python training/train.py --data-dir /path/to/DATA_DIR --output outputs/holdout25 --holdout 20261025
uv run ded predict --root outputs/holdout25 --data-dir /path/to/DATA_DIR --split train --day 20261025 --output outputs/holdout25.csv
uv run python training/evaluate.py --predictions outputs/holdout25.csv --labels /path/to/DATA_DIR/train.csv
```

Используйте этот комплект только для оценки соответствующего отложенного дня. Публичные Grounding DINO и MediaPipe не дообучаются. В тренировочные и диагностические загрузчики YOLO передаются только разрешённые train-кадры; диагностический `val` — копия train, без выбора эпох по отложенному дню.

Сохранённые параметры исходных финальных детекторов: `original_training_parameters.json`. Точные параметры каждого этапа, используемые скриптом: `detector_recipes.json`. Для full-data person/multi применяется warm start на 26 октября, затем обучение на обоих днях. Предыдущие этапы: person — 20 эпох исходного GT + 12 эпох расширенной разметки; multi — 30 эпох исходного четырёхклассового GT. Финальные этапы: person 8 эпох, multi 12, tiles 18.

Этот скрипт восстанавливает выбранную архитектуру и путь обучения. Сравнение в основном README относится к включённым весам и сохранённым day-held-out предсказаниям, а не к ещё не выполненному переобучению.
