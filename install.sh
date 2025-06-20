#!/bin/bash

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Функции ---

# Функция установки системных зависимостей и зависимостей Python
install_dependencies() {
    echo -e "${BLUE}--- Шаг 1: Установка зависимостей ---${NC}"
    
    echo "Обновление списка пакетов..."
    sudo apt-get update
    
    echo "Установка системных утилит (ffmpeg, sox)..."
    sudo apt-get install -y ffmpeg sox libsndfile1
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Ошибка при установке системных пакетов. Пожалуйста, проверьте вывод и попробуйте снова.${NC}"
        exit 1
    fi
    
    echo "Установка зависимостей Python из requirements.txt..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Ошибка при установке Python пакетов. Убедитесь, что у вас активировано виртуальное окружение, если это необходимо.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Установка зависимостей успешно завершена!${NC}"
}

# Функция сегментации аудио
segment_audio() {
    echo -e "${BLUE}--- Сегментация аудио ---${NC}"
    
    SRC_DIR="data/audio"
    DEST_DIR="data/segment_audio"
    
    mkdir -p "$SRC_DIR"
    mkdir -p "$DEST_DIR"
    
    echo -e "Пожалуйста, поместите ваши аудиофайлы (mp3, flac, wav и т.д.) в папку: ${YELLOW}$SRC_DIR${NC}"
    echo "Нажмите Enter, когда будете готовы продолжить..."
    read
    
    if [ -z "$(ls -A $SRC_DIR)" ]; then
       echo -e "${YELLOW}Папка $SRC_DIR пуста. Сегментация не будет выполнена.${NC}"
       return
    fi
    
    echo "Начинаю сегментацию. Файлы будут сохранены в $DEST_DIR"
    echo "Параметры конвертации: WAV, 16-бит PCM, 22050 Гц, моно."
    
    for FILE in "$SRC_DIR"/*; do
        if [ -f "$FILE" ]; then
            FILENAME=$(basename -- "$FILE")
            BASENAME="${FILENAME%.*}"
            echo "Обработка файла: $FILENAME"
            ffmpeg -i "$FILE" -ar 22050 -ac 1 -c:a pcm_s16le "$DEST_DIR/${BASENAME}.wav" -y -hide_banner -loglevel error
        fi
    done
    
    echo -e "${GREEN}Сегментация аудио завершена!${NC}"
}

# Функция транскрибации
transcribe_data() {
    echo -e "${BLUE}--- Транскрибация аудио ---${NC}"
    
    if ! command -v python &> /dev/null
    then
        echo -e "${YELLOW}Python не найден. Пожалуйста, установите Python и попробуйте снова.${NC}"
        return
    fi
    
    echo "Запуск скрипта транскрибации..."
    python transcribe.py --data_dir="data/segment_audio" --output_dir="data/dataset"
    
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Во время транскрибации произошла ошибка. Проверьте вывод скрипта.${NC}"
    else
        echo -e "${GREEN}Транскрибация успешно завершена!${NC}"
    fi
}


# Функция меню для работы с датасетом
dataset_menu() {
    while true; do
        echo -e "\n${YELLOW}--- Меню работы с датасетом ---${NC}"
        echo "1. Сегментация аудио (из /data/audio в /data/segment_audio)"
        echo "2. Транскрибация аудио (из /data/segment_audio)"
        echo "0. Назад в главное меню"
        echo -n "Выберите опцию: "
        read -r choice
        
        case $choice in
            1) segment_audio ;;
            2) transcribe_data ;;
            0) break ;;
            *) echo "Неверный выбор." ;;
        esac
    done
}

# --- Главное меню ---
main_menu() {
    while true; do
        echo -e "\n${YELLOW}--- Главное меню подготовки проекта ---${NC}"
        echo "1. Установка всех зависимостей (ПО и Python)"
        echo "2. Работа с датасетом (Сегментация и Транскрибация)"
        echo "---"
        echo "0. Выполнить все шаги по порядку (1 -> 2.1 -> 2.2)"
        echo "9. Выход"
        echo -n "Выберите опцию: "
        read -r choice
        
        case $choice in
            1) install_dependencies ;;
            2) dataset_menu ;;
            0)
                echo -e "${BLUE}--- Запуск всех шагов ---${NC}"
                install_dependencies
                segment_audio
                transcribe_data
                echo -e "${GREEN}Все шаги выполнены!${NC}"
                ;;
            9) exit 0 ;;
            *) echo "Неверный выбор." ;;
        esac
    done
}

# Запуск главного меню
main_menu 