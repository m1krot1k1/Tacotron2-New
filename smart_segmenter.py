import subprocess
import os
import re
from tqdm import tqdm
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Регулярные выражения для поиска временных меток тишины
silence_start_re = re.compile(r'silence_start: (\d+\.?\d*)')
silence_end_re = re.compile(r'silence_end: (\d+\.?\d*)')

def get_speech_segments_for_file(file_info):
    """
    Функция-воркер: анализирует ОДИН файл и возвращает его сегменты.
    """
    input_file, silence_db, silence_duration = file_info
    try:
        duration_command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_file]
        duration_result = subprocess.run(duration_command, text=True, capture_output=True, check=True, encoding='utf-8')
        total_duration = float(duration_result.stdout)
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return (input_file, []) # Возвращаем пустой список для этого файла

    command = ['ffmpeg', '-i', input_file, '-af', f'silencedetect=noise={silence_db}:d={silence_duration}', '-f', 'null', '-']
    process = subprocess.Popen(command, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')

    starts, ends = [], []
    for line in iter(process.stderr.readline, ''):
        if start_match := silence_start_re.search(line): starts.append(start_match.group(1))
        if end_match := silence_end_re.search(line): ends.append(end_match.group(1))
    process.wait()

    if not starts and not ends: return (input_file, [(0.0, total_duration)])
    if len(starts) > len(ends): ends.append(str(total_duration))

    silences = [(float(s), float(e)) for s, e in zip(starts, ends)]
    speech_segments = []
    last_silence_end = 0.0
    for silence_start, silence_end in silences:
        if silence_start > last_silence_end:
            speech_segments.append((last_silence_end, silence_start))
        last_silence_end = silence_end
    if total_duration > last_silence_end:
        speech_segments.append((last_silence_end, total_duration))
    
    return (input_file, speech_segments)

def process_chunk(chunk_info):
    """
    Функция-воркер: нарезает ОДИН маленький чанк.
    """
    input_file, output_dir, chunk_details, segment_index = chunk_info
    start, end = chunk_details
    basename = os.path.splitext(os.path.basename(input_file))[0]
    output_filename = os.path.join(output_dir, f"{basename}_seg{segment_index:05d}.wav")
    
    command = ['ffmpeg', '-nostdin', '-i', input_file, '-ss', str(start), '-t', str(end - start),
               '-ar', '22050', '-ac', '1', '-c:a', 'pcm_s16le', output_filename, '-y', '-hide_banner', '-loglevel', 'error']
    subprocess.run(command)
    return 1

def main():
    parser = argparse.ArgumentParser(description="Высокопроизводительный многопроцессорный сегментатор аудио.")
    parser.add_argument('--input_dir', type=str, default='data/audio', help='Директория с исходными аудиофайлами.')
    parser.add_argument('--output_dir', type=str, default='data/segment_audio', help='Директория для сохранения сегментов.')
    parser.add_argument('--min_len', type=float, default=2.0, help='Минимальная длина сегмента в секундах.')
    parser.add_argument('--max_len', type=float, default=15.0, help='Максимальная длина сегмента в секундах.')
    parser.add_argument('--silence_db', type=str, default='-30dB', help='Порог тишины в дБ.')
    parser.add_argument('--silence_dur', type=str, default='0.5', help='Длительность тишины для срабатывания детектора.')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), help='Количество параллельных процессов для обработки.')
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.isdir(args.input_dir):
        os.makedirs(args.input_dir); print(f"Создана директория {args.input_dir}. Пожалуйста, поместите в нее аудиофайлы."); return

    source_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    if not source_files: print(f"В директории {args.input_dir} не найдены аудиофайлы."); return

    print(f"Найдено {len(source_files)} файлов. Запускаю этап 1: Анализ речевых сегментов...")

    # --- ЭТАП 1: Параллельный анализ всех файлов ---
    tasks_analysis = [(f, args.silence_db, args.silence_dur) for f in source_files]
    all_chunks_to_process = []
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(get_speech_segments_for_file, task): task for task in tasks_analysis}
        for future in tqdm(as_completed(futures), total=len(tasks_analysis), desc="Анализ файлов", unit="файл"):
            input_file, speech_segments = future.result()
            for seg_start, seg_end in speech_segments:
                duration = seg_end - seg_start
                if duration < args.min_len: continue
                if duration > args.max_len:
                    num_sub_chunks = int(duration // args.max_len) + 1
                    sub_chunk_dur = duration / num_sub_chunks
                    for i in range(num_sub_chunks):
                        start = seg_start + i * sub_chunk_dur
                        end = start + sub_chunk_dur
                        if (end - start) >= args.min_len:
                            all_chunks_to_process.append({'file': input_file, 'start': start, 'end': end})
                else:
                    all_chunks_to_process.append({'file': input_file, 'start': seg_start, 'end': seg_end})

    if not all_chunks_to_process:
        print("Анализ завершен. Не найдено ни одного подходящего речевого сегмента для нарезки."); return

    print(f"\nАнализ завершен. Найдено {len(all_chunks_to_process)} сегментов. Запускаю этап 2: Параллельная нарезка...")

    # --- ЭТАП 2: Параллельная нарезка всех найденных сегментов ---
    tasks_slicing = [(chunk['file'], args.output_dir, (chunk['start'], chunk['end']), i) for i, chunk in enumerate(all_chunks_to_process)]
    total_segments_created = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures_slicing = {executor.submit(process_chunk, task): task for task in tasks_slicing}
        for future in tqdm(as_completed(futures_slicing), total=len(tasks_slicing), desc="Нарезка сегментов", unit="сегмент"):
            total_segments_created += future.result()

    print(f"\n==================================================")
    print(f"Умная сегментация завершена.")
    print(f"Всего создано {total_segments_created} сегментов.")
    print(f"Готовые сегменты сохранены в: {args.output_dir}")
    print(f"==================================================")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main() 