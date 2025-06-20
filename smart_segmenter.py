import subprocess
import os
import re
from tqdm import tqdm
import argparse

# Регулярные выражения для поиска временных меток тишины в выводе ffmpeg
silence_start_re = re.compile(r'silence_start: (\d+\.?\d*)')
silence_end_re = re.compile(r'silence_end: (\d+\.?\d*)')

def get_speech_segments(input_file, silence_db='-30dB', silence_duration='0.5'):
    """
    Анализирует аудиофайл с помощью ffmpeg silencedetect и возвращает
    список сегментов с речью (не тишиной).
    """
    # Сначала получаем общую длительность файла с помощью ffprobe
    try:
        duration_command = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', input_file
        ]
        duration_result = subprocess.run(duration_command, text=True, capture_output=True, check=True)
        total_duration = float(duration_result.stdout)
    except (subprocess.CalledProcessError, ValueError, IndexError):
        tqdm.write(f"Не удалось получить длительность для файла: {input_file}. Пропускаем.")
        return []

    # Команда для обнаружения тишины
    command = [
        'ffmpeg', '-i', input_file,
        '-af', f'silencedetect=noise={silence_db}:d={silence_duration}',
        '-f', 'null', '-'
    ]
    
    # ffmpeg выводит информацию о тишине в stderr
    result = subprocess.run(command, text=True, capture_output=True)
    stderr_output = result.stderr

    starts = silence_start_re.findall(stderr_output)
    ends = silence_end_re.findall(stderr_output)

    # Собираем интервалы тишины
    silences = []
    if len(starts) == len(ends):
        for s, e in zip(starts, ends):
            silences.append((float(s), float(e)))
    else: # Если начало и конец не совпадают, что-то пошло не так
        tqdm.write(f"Предупреждение: Несовпадение меток начала/конца тишины в файле {input_file}")
        # Можно попробовать восстановить, но пока пропустим
        # В данном примере, если что-то не так, лучше не рисковать
        if len(starts) > len(ends):
             starts = starts[:len(ends)]
        else:
             ends = ends[:len(starts)]
        for s, e in zip(starts, ends):
            silences.append((float(s), float(e)))


    # Создаем сегменты речи на основе инверсии сегментов тишины
    speech_segments = []
    last_end = 0.0
    for start, end in silences:
        if start > last_end:
            speech_segments.append((last_end, start))
        last_end = end
    
    if last_end < total_duration:
        speech_segments.append((last_end, total_duration))
        
    return speech_segments

def split_audio_into_chunks(input_file, output_dir, speech_segments, min_len=2.0, max_len=15.0):
    """
    Нарезает аудиофайл на чанки на основе сегментов речи и ограничений по длине.
    """
    basename = os.path.splitext(os.path.basename(input_file))[0]
    segment_count = 0
    
    for start, end in speech_segments:
        duration = end - start
        
        if duration < min_len:
            continue
            
        if duration > max_len:
            # Если сегмент слишком длинный, делим его на части по max_len
            num_chunks = int(duration // max_len) + 1
            chunk_duration = duration / num_chunks
            
            for i in range(num_chunks):
                chunk_start = start + i * chunk_duration
                # Длительность текущего чанка
                current_chunk_duration = min(chunk_duration, end - chunk_start)
                # Пропускаем слишком короткие остатки в конце
                if current_chunk_duration < min_len:
                    continue
                
                output_filename = os.path.join(output_dir, f"{basename}_seg{segment_count:04d}.wav")
                command = [
                    'ffmpeg', '-i', input_file, '-ss', str(chunk_start), '-t', str(current_chunk_duration),
                    '-ar', '22050', '-ac', '1', '-c:a', 'pcm_s16le',
                    output_filename, '-y', '-hide_banner', '-loglevel', 'error'
                ]
                subprocess.run(command)
                segment_count += 1
        else:
            # Сегмент уже имеет подходящую длину
            output_filename = os.path.join(output_dir, f"{basename}_seg{segment_count:04d}.wav")
            command = [
                'ffmpeg', '-i', input_file, '-ss', str(start), '-t', str(duration),
                '-ar', '22050', '-ac', '1', '-c:a', 'pcm_s16le',
                output_filename, '-y', '-hide_banner', '-loglevel', 'error'
            ]
            subprocess.run(command)
            segment_count += 1
    return segment_count

def main():
    parser = argparse.ArgumentParser(description="Умный сегментатор аудио на основе определения тишины.")
    parser.add_argument('--input_dir', type=str, default='data/audio', help='Директория с исходными аудиофайлами.')
    parser.add_argument('--output_dir', type=str, default='data/segment_audio', help='Директория для сохранения сегментов.')
    parser.add_argument('--min_len', type=float, default=2.0, help='Минимальная длина сегмента в секундах.')
    parser.add_argument('--max_len', type=float, default=15.0, help='Максимальная длина сегмента в секундах.')
    parser.add_argument('--silence_db', type=str, default='-30dB', help='Порог тишины в дБ.')
    parser.add_argument('--silence_dur', type=str, default='0.5', help='Длительность тишины для срабатывания детектора.')
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.input_dir, exist_ok=True)

    source_files = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    
    if not source_files:
        print(f"В директории {args.input_dir} не найдены аудиофайлы. Пожалуйста, добавьте файлы и повторите попытку.")
        return

    print(f"Найдено {len(source_files)} файлов для обработки.")
    total_segments = 0

    for filename in tqdm(source_files, desc="Общий прогресс", unit="файл"):
        input_path = os.path.join(args.input_dir, filename)
        
        # 1. Находим участки с речью
        speech_segments = get_speech_segments(input_path, args.silence_db, args.silence_dur)
        
        # 2. Нарезаем аудио на основе найденных сегментов
        if speech_segments:
            count = split_audio_into_chunks(input_path, args.output_dir, speech_segments, args.min_len, args.max_len)
            total_segments += count
        else:
            tqdm.write(f"Не найдено речевых сегментов в файле {filename}.")
            
    print(f"\nУмная сегментация завершена.")
    print(f"Всего создано {total_segments} сегментов.")
    print(f"Готовые сегменты сохранены в: {args.output_dir}")

if __name__ == "__main__":
    main() 