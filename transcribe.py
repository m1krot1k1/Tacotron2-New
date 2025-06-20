import os
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse
import transformers
from transformers import pipeline

def transcribe_audio_pipeline(data_dir, output_dir, batch_size):
    """
    Транскрибирует аудиофайлы в пакетном режиме, используя Hugging Face pipeline
    для максимальной производительности.

    Args:
        data_dir (str): Путь к папке с аудиосегментами.
        output_dir (str): Путь к папке для сохранения файлов датасета.
        batch_size (int): Размер пакета для обработки.
    """
    print(f"Используется версия transformers: {transformers.__version__}")

    # Проверяем наличие GPU и настраиваем устройство
    if torch.cuda.is_available():
        device = "cuda:0"
        torch_dtype = torch.float16
        print(f"Обнаружен CUDA. Транскрибация будет выполняться на {device} с использованием float16.")
    else:
        device = "cpu"
        torch_dtype = torch.float32
        print("ВНИМАНИЕ: CUDA не найдена. Транскрибация будет выполняться на CPU, что может быть очень медленно.")

    # Создаем выходные директории, если их нет
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Загрузка модели Whisper large-v3 через Hugging Face pipeline... (это может занять время)")
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        torch_dtype=torch_dtype,
        device=device
    )
    print("Модель загружена.")

    all_audio_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]
    
    # Проверяем файлы на корректность (не пустые и доступны для чтения)
    audio_files = []
    print("Проверка аудиофайлов перед транскрибацией...")
    for f in tqdm(all_audio_files, desc="Проверка файлов"):
        try:
            # Проверяем, что размер файла больше минимального порога (например, 1KB)
            if os.path.getsize(f) > 1024:
                audio_files.append(f)
            else:
                print(f"\nПРЕДУПРЕЖДЕНИЕ: Пропускается слишком маленький файл (возможно, поврежден): {f}")
        except Exception as e:
            print(f"\nОШИБКА: Не удалось получить доступ к файлу {f}, он будет пропущен. Ошибка: {e}")

    if not audio_files:
        print(f"В директории {data_dir} не найдено корректных .wav файлов для обработки.")
        return

    print(f"Найдено {len(audio_files)} корректных аудиофайлов для транскрибации. Начинаю пакетную обработку...")
    
    try:
        # Пакетная транскрибация. Pipeline сам покажет прогресс.
        outputs = pipe(
            audio_files, 
            batch_size=batch_size, 
            generate_kwargs={'language': 'russian', 'task': 'transcribe'}
        )
    except torch.cuda.OutOfMemoryError:
        print("\n---------------------------------------------------------")
        print("ОШИБКА: Недостаточно видеопамяти (CUDA Out of Memory).")
        print(f"Текущий размер пачки ({batch_size}) слишком велик для вашей GPU.")
        print("Попробуйте перезапустить скрипт с меньшим значением batch_size (например, 4 или 2).")
        print("---------------------------------------------------------")
        return # Завершаем выполнение, чтобы пользователь мог изменить параметры

    # Собираем результаты
    results = []
    for i, output in enumerate(tqdm(outputs, desc="Обработка результатов")):
        path = audio_files[i]
        text = output['text'].strip()
        # Для Tacotron важно, чтобы в конце не было точки
        if text.endswith('.'):
            text = text[:-1]
        
        # Добавляем в список только если транскрибация не пустая
        if text:
            results.append(f"{os.path.abspath(path)}|{text}")

    if not results:
        print("Не удалось получить ни одного результата транскрибации.")
        return

    # Разделяем данные на обучающую и валидационную выборки (95/5)
    train_data, val_data = train_test_split(results, test_size=0.05, random_state=42)

    # Сохраняем файлы в формате CSV без заголовков
    train_filepath = os.path.join(output_dir, 'train.csv')
    val_filepath = os.path.join(output_dir, 'val.csv')

    with open(train_filepath, 'w', encoding='utf-8') as f:
        for line in train_data:
            f.write(f"{line}\n")

    with open(val_filepath, 'w', encoding='utf-8') as f:
        for line in val_data:
            f.write(f"{line}\n")
            
    print("\nТранскрибация завершена.")
    print(f"Созданы файлы датасета:")
    print(f"  - Обучающая выборка: {train_filepath} ({len(train_data)} записей)")
    print(f"  - Валидационная выборка: {val_filepath} ({len(val_data)} записей)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт для пакетной транскрибации аудио с помощью Whisper.")
    parser.add_argument('--data_dir', type=str, default='data/segment_audio', 
                        help='Директория с сегментированными аудиофайлами (.wav).')
    parser.add_argument('--output_dir', type=str, default='data/dataset', 
                        help='Директория для сохранения файлов датасета (train.csv, val.csv).')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Размер пакета (batch size) для транскрибации. Подбирайте в зависимости от VRAM вашей GPU.')

    args = parser.parse_args()
    
    transcribe_audio_pipeline(args.data_dir, args.output_dir, args.batch_size) 