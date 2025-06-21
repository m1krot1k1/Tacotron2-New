#!/usr/bin/env python3
"""
Live Log Parser for Smart Tuner V2
Reads TensorBoard event files in real-time and writes them to a human-readable text file.
"""

import os
import sys
import time
import argparse
import logging
import tensorflow as tf
import pandas as pd
from collections import defaultdict

# Настройка логирования для этого скрипта
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - (LiveLogParser) - %(message)s',
)

class LiveLogParser:
    def __init__(self, tfevents_dir, output_txt_file, poll_interval=15):
        self.tfevents_dir = tfevents_dir
        self.output_txt_file = output_txt_file
        self.poll_interval = poll_interval
        self.processed_steps = set()
        self.tfevents_file_path = None

        # Создаем директорию для файла, если ее нет
        os.makedirs(os.path.dirname(self.output_txt_file), exist_ok=True)
        # Очищаем файл при старте
        with open(self.output_txt_file, 'w') as f:
            f.write(f"--- Live Training Log ---\\n")
            f.write(f"Source: {self.tfevents_dir}\\n")
            f.write(f"Last Updated: {time.ctime()}\\n")
            f.write("="*30 + "\\n")
        
        logging.info(f"Инициализирован парсер для директории: {self.tfevents_dir}")
        logging.info(f"Выходной файл: {self.output_txt_file}")

    def find_tfevents_file(self):
        """Находит самый свежий tfevents файл в директории."""
        try:
            files = [os.path.join(self.tfevents_dir, f) for f in os.listdir(self.tfevents_dir) if 'tfevents' in f]
            if not files:
                return None
            return max(files, key=os.path.getmtime)
        except FileNotFoundError:
            return None
            
    def parse_and_write(self):
        """Основная функция парсинга и записи."""
        if not self.tfevents_file_path:
            self.tfevents_file_path = self.find_tfevents_file()
            if not self.tfevents_file_path:
                logging.info("Ожидание файла tfevents...")
                return

        data = defaultdict(dict)
        new_steps_found = False
        try:
            for event in tf.compat.v1.train.summary_iterator(self.tfevents_file_path):
                step = event.step
                if step not in self.processed_steps:
                    new_steps_found = True
                    for value in event.summary.value:
                        # Собираем все метрики для данного шага
                        data[step][value.tag] = value.simple_value
        except Exception as e:
            # Файл может быть в процессе записи, игнорируем ошибки чтения
            logging.debug(f"Ошибка чтения файла {self.tfevents_file_path}: {e}")
            return
            
        if not new_steps_found:
            return

        # Записываем новые данные в файл
        with open(self.output_txt_file, 'a') as f:
            sorted_new_steps = sorted([s for s in data.keys() if s not in self.processed_steps])
            for step in sorted_new_steps:
                metrics_str = ", ".join([f"{tag}: {val:.4f}" for tag, val in data[step].items()])
                f.write(f"Step {step}: {metrics_str}\\n")
                self.processed_steps.add(step)
        
        logging.info(f"Добавлены записи до шага {max(self.processed_steps)}")

    def watch(self):
        """Запускает бесконечный цикл отслеживания."""
        logging.info("Запуск live-parser в режиме отслеживания...")
        while True:
            try:
                self.parse_and_write()
                time.sleep(self.poll_interval)
            except KeyboardInterrupt:
                logging.info("Парсер остановлен вручную.")
                break
            except Exception as e:
                logging.error(f"Критическая ошибка в цикле watch: {e}")
                time.sleep(self.poll_interval * 2) # Ждем дольше при ошибке


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live TensorBoard Event File Parser")
    parser.add_argument("--tfevents_dir", type=str, required=True, help="Директория с файлами tfevents")
    parser.add_argument("--output_txt_file", type=str, required=True, help="Путь к выходному текстовому лог-файлу")
    parser.add_argument("--poll_interval", type=int, default=15, help="Интервал опроса в секундах")
    args = parser.parse_args()

    live_parser = LiveLogParser(args.tfevents_dir, args.output_txt_file, args.poll_interval)
    live_parser.watch() 