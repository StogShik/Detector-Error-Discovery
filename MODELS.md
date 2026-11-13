# Модели

| Файл / модель | Назначение | Исходные публичные веса |
|---|---|---|
| `models/geometry_0.cbm` | CatBoost depth=3, 100 деревьев, scene, seed=0 | Обучен на TRAIN error |
| `models/geometry_1.cbm` | CatBoost depth=2, 400 деревьев, scene, seed=0 | Обучен на TRAIN error |
| `models/geometry_2.cbm` | CatBoost depth=2, 400 деревьев, compact, seed=0 | Обучен на TRAIN error |
| `models/multi.pt` | YOLO11s: person, nude_hands, cup, tray | [YOLO11s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt) |
| `models/person.pt` | YOLO11s: люди с полными границами | [YOLO11s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt) |
| `models/tiles.pt` | YOLO11s: руки, чашки, подносы на тайлах | [YOLO11s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt) |
| `models/clothing.pt` | YOLO11s-cls: person/clothes | [YOLO11s-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-cls.pt) |
| `models/hand_landmarker.task` | Замороженный MediaPipe Hand Landmarker float16 | [Публичный task-файл](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) |
| `IDEA-Research/grounding-dino-base` | Замороженная независимая проверка границ | [Hugging Face](https://huggingface.co/IDEA-Research/grounding-dino-base) |

Grounding DINO использует один общий набор весов для двух запросов: `person. hand. cup. glass. tray.` и `hand. cup. glass. tray. glove. bottle.`. Ревизия закреплена в `config.json`.

SHA-256 включённых файлов проверяется командой `uv run ded verify`. Их суммарный размер около 73 MiB; большой публичный Grounding DINO хранится в стандартном кэше Hugging Face, а не в Git. Файлы `.pt` являются стандартными Ultralytics checkpoints; исходные training arguments внутри них не определяют путь к вашим данным при запуске этого проекта.

Сторонние библиотеки и веса сохраняют собственные лицензии; оригинальные лицензии и условия использования доступны по указанным ссылкам и в дистрибутивах зависимостей.

## Необязательный детектор полного тела

`--recipe person-truncation` и `correct-people` используют публичный **ByteTrack YOLOX-X MOT17 mixed**, depth=1.33, width=1.25, вход 800×1440. [Официальные веса](https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view), SHA-256 `e3945f3523fde1e107708aacd64dab0670c34e371d136e54587cac7a50d3cfba`. Автоматическая загрузка — `.cache/models/bytetrack_x_mot17.pth.tar`; дообучение отсутствует.

[Официальный репозиторий](https://github.com/FoundationVision/ByteTrack) указывает обучение на CrowdHuman, MOT17, Cityperson и ETHZ. [Конвертер CrowdHuman](https://github.com/FoundationVision/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/tools/convert_crowdhuman_to_coco.py) использует `fbox`, полный бокс тела. Это источник дополнительного сигнала о скрытой нижней части человека, а не гарантия правильности конкретного бокса.

Минимальные модули детектора перенесены из ревизии `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` в `src/detector_error_discovery/_bytetrack/`. Лицензия MIT включена рядом. Изменения: относительный импорт `bboxes_iou`, стандартный logger вместо loguru и перенос сеток декодирования на устройство выходного тензора для MPS. Трекер и его скомпилированные зависимости не нужны. Препроцессинг соответствует старому ByteTrack: RGB/255, ImageNet mean/std, заполнение 114 слева сверху; современный ненормализованный YOLOX-препроцессинг здесь не подходит.
