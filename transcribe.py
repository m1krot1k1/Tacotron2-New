import whisper
import os
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

def transcribe_audio(data_dir, output_dir):
    """
    Транскрибирует аудиофайлы из data_dir, используя OpenAI Whisper,
    и создает файлы для обучения и валидации.

    Args:
        data_dir (str): Путь к папке с аудиосегментами.
        output_dir (str): Путь к папке для сохранения файлов датасета.
    """
    # Проверяем наличие GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ВНИМАНИЕ: CUDA не найдена. Транскрибация будет выполняться на CPU, что может быть очень медленно.")

    # Создаем выходные директории, если их нет
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Загрузка модели Whisper large-v3... (это может занять время)")
    model = whisper.load_model("large-v3", device=device)
    print("Модель загружена.")

    audio_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    if not audio_files:
        print(f"В директории {data_dir} не найдено .wav файлов.")
        return

    print(f"Найдено {len(audio_files)} аудиофайлов для транскрибации.")
    
    results = []
    # Используем tqdm для отображения прогресс-бара
    for audio_file in tqdm(audio_files, desc="Транскрибация аудио"):
        path = os.path.join(data_dir, audio_file)
        try:
            # Загружаем аудио и транскрибируем
            result = model.transcribe(path, language="ru", fp16=torch.cuda.is_available())
            
            # Форматируем строку: путь|транскрипция
            text = result['text'].strip()
            # Для Tacotron важно, чтобы в конце не было точки
            if text.endswith('.'):
                text = text[:-1]
                
            results.append(f"{os.path.abspath(path)}|{text}")
        except Exception as e:
            print(f"Ошибка при обработке файла {path}: {e}")

    if not results:
        print("Не удалось получить ни одного результата транскрибации.")
        return

    # Разделяем данные на обучающую и валидационную выборки (95/5)
    train_data, val_data = train_test_split(results, test_size=0.05, random_state=42)

    # Сохраняем файлы
    train_filepath = os.path.join(output_dir, 'train.txt')
    val_filepath = os.path.join(output_dir, 'val.txt')

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
    parser = argparse.ArgumentParser(description="Скрипт для транскрибации аудио с помощью Whisper.")
    parser.add_argument('--data_dir', type=str, default='data/segment_audio', 
                        help='Директория с сегментированными аудиофайлами (.wav).')
    parser.add_argument('--output_dir', type=str, default='data/dataset', 
                        help='Директория для сохранения файлов датасета (train.txt, val.txt).')

    args = parser.parse_args()
    
    transcribe_audio(args.data_dir, args.output_dir) 