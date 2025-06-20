

#!/bin/bash

# ==============================================================================
# Скрипт для полной настройки рабочего окружения проекта TTS
# ==============================================================================
#
# Этот скрипт выполняет следующие действия:
# 1. Устанавливает необходимые системные пакеты (Python, build-essential).
# 2. Проверяет наличие драйверов NVIDIA (критически важно).
# 3. Создает изолированное виртуальное окружение Python (venv).
# 4. Автоматически определяет версию CUDA и устанавливает совместимую
#    версию PyTorch с поддержкой GPU.
# 5. Устанавливает все остальные зависимости проекта.
# 6. Предоставляет меню для обработки данных (сегментация, транскрибация).
#
# ==============================================================================

# --- Конфигурация ---
PYTHON_VERSION="3.10"
VENV_DIR="venv"

# --- Цвета для вывода ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# --- Функции ---

# Функция для проверки и создания виртуального окружения
setup_venv() {
    if [ ! -f "$VENV_DIR/bin/python" ]; then
        echo -e "${BLUE}Создание виртуального окружения в '$VENV_DIR' с использованием python${PYTHON_VERSION}...${NC}"
        # Удаляем старую папку, если она не является рабочим venv
        if [ -d "$VENV_DIR" ]; then
            rm -rf "$VENV_DIR"
        fi
        "python${PYTHON_VERSION}" -m venv "$VENV_DIR"
        if [ $? -ne 0 ]; then
            echo -e "${RED}Не удалось создать виртуальное окружение. Убедитесь, что 'python${PYTHON_VERSION}-venv' установлен.${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}Виртуальное окружение '$VENV_DIR' уже существует и готово к использованию.${NC}"
    fi
}

# Главная функция установки зависимостей
install_environment() {
    echo -e "${BLUE}--- Шаг 1: Настройка окружения и установка зависимостей ---${NC}"

    # 1. Проверка и установка системных пакетов
    echo -e "\n${YELLOW}--> 1.1 Проверка и установка системных пакетов...${NC}"
    
    REQUIRED_PKGS=( "build-essential" "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-dev" "python${PYTHON_VERSION}-venv" "ffmpeg" "sox" "libsndfile1" )
    PKGS_TO_INSTALL=()

    for pkg in "${REQUIRED_PKGS[@]}"; do
        if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
            PKGS_TO_INSTALL+=("$pkg")
        fi
    done

    if [ ${#PKGS_TO_INSTALL[@]} -ne 0 ]; then
        echo "Следующие пакеты будут установлены: ${PKGS_TO_INSTALL[*]}"
        sudo apt-get update
        sudo apt-get install -y "${PKGS_TO_INSTALL[@]}"
        if [ $? -ne 0 ]; then
            echo -e "${RED}Ошибка при установке системных пакетов. Пожалуйста, проверьте вывод.${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}Все необходимые системные пакеты уже установлены.${NC}"
    fi

    # 2. Проверка драйверов NVIDIA
    echo -e "\n${YELLOW}--> 1.2 Проверка наличия драйверов NVIDIA...${NC}"
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}ОШИБКА: Драйверы NVIDIA не найдены ('nvidia-smi' не доступна).${NC}"
        echo -e "${YELLOW}Пожалуйста, установите проприетарные драйверы NVIDIA для вашей видеокарты."
        echo -e "После установки ${RED}ОБЯЗАТЕЛЬНО ПЕРЕЗАГРУЗИТЕ${NC} компьютер и запустите скрипт снова."
        exit 1
    fi
    echo -e "${GREEN}Драйверы NVIDIA обнаружены.${NC}"
    nvidia-smi

    # 3. Создание VENV
    echo -e "\n${YELLOW}--> 1.3 Настройка виртуального окружения Python...${NC}"
    setup_venv

    # 4. Проверка и установка PyTorch
    echo -e "\n${YELLOW}--> 1.4 Проверка и установка PyTorch с поддержкой GPU...${NC}"
    
    # Проверяем, установлен ли уже PyTorch с поддержкой CUDA
    if "$VENV_DIR/bin/python" -c "import torch; exit(0) if torch.cuda.is_available() else exit(1)" &>/dev/null; then
        PYTORCH_VERSION=$("$VENV_DIR/bin/python" -c "import torch; print(torch.__version__)")
        echo -e "${GREEN}PyTorch ($PYTORCH_VERSION) с поддержкой CUDA уже установлен. Пропускаем.${NC}"
    else
        echo -e "${BLUE}PyTorch с CUDA не найден или неисправен. Начинаю установку...${NC}"
        
        CUDA_VERSION_STRING=$(nvidia-smi | grep "CUDA Version:" | awk '{print $9}')
        if [ -z "$CUDA_VERSION_STRING" ]; then
            echo -e "${RED}Не удалось автоматически определить версию CUDA из вывода 'nvidia-smi'.${NC}"
            exit 1
        fi
        
        CUDA_MAJOR=$(echo "$CUDA_VERSION_STRING" | cut -d'.' -f1)
        PYTORCH_URL=""

        if [[ "$CUDA_MAJOR" -ge "12" ]]; then
            echo "Обнаружена CUDA ${CUDA_VERSION_STRING}. Выбираем PyTorch для CUDA 12.1."
            PYTORCH_URL="https://download.pytorch.org/whl/cu121"
        elif [[ "$CUDA_MAJOR" -ge "11" ]]; then
            echo "Обнаружена CUDA ${CUDA_VERSION_STRING}. Выбираем PyTorch для CUDA 11.8."
            PYTORCH_URL="https://download.pytorch.org/whl/cu118"
        else
            echo -e "${RED}Не удалось определить совместимую версию PyTorch для вашей CUDA (${CUDA_VERSION_STRING}).${NC}"
            exit 1
        fi

        "$VENV_DIR/bin/pip" install torch torchvision torchaudio --index-url "$PYTORCH_URL"
        if [ $? -ne 0 ]; then
            echo -e "${RED}Ошибка при установке PyTorch. Проверьте вывод.${NC}"
            exit 1
        fi
        echo -e "${GREEN}PyTorch успешно установлен.${NC}"
    fi

    # 5. Установка остальных зависимостей
    echo -e "\n${YELLOW}--> 1.5 Установка зависимостей проекта из requirements.txt...${NC}"
    "$VENV_DIR/bin/pip" install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}Ошибка при установке Python пакетов. Проверьте 'requirements.txt' и вывод ошибок.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Зависимости проекта успешно синхронизированы.${NC}"
    
    echo -e "\n${GREEN}=======================================================${NC}"
    echo -e "${GREEN}Настройка окружения и установка зависимостей успешно завершена!${NC}"
    echo -e "${GREEN}=======================================================${NC}"
}

# Функция умной сегментации аудио
segment_audio() {
    echo -e "${BLUE}--- Шаг 2.1: Умная сегментация аудио ---${NC}"
    
    SRC_DIR="data/audio"
    DEST_DIR="data/segment_audio"
    mkdir -p "$SRC_DIR" "$DEST_DIR"
    
    echo -e "Убедитесь, что ваши аудиофайлы находятся в: ${YELLOW}$SRC_DIR${NC}"
    echo "Нажмите Enter, чтобы продолжить..."
    read
    
    if [ -z "$(ls -A $SRC_DIR 2>/dev/null)" ]; then
       echo -e "${YELLOW}Папка $SRC_DIR пуста. Сегментация не выполняется.${NC}"
       return
    fi
    
    if [ ! -f "$VENV_DIR/bin/python" ]; then
        echo -e "${RED}Python в виртуальном окружении не найден. Запустите сначала установку (пункт 1).${NC}"
        return
    fi

    echo "Запуск скрипта умной сегментации..."
    "$VENV_DIR/bin/python" smart_segmenter.py --input_dir "$SRC_DIR" --output_dir "$DEST_DIR"
    echo -e "${GREEN}Умная сегментация завершена.${NC}"
}

# Функция транскрибации
transcribe_data() {
    echo -e "${BLUE}--- Шаг 2.2: Транскрибация аудио ---${NC}"
    
    if [ ! -f "$VENV_DIR/bin/python" ]; then
        echo -e "${RED}Python в виртуальном окружении не найден. Запустите сначала установку (пункт 1).${NC}"
        return
    fi
    
    echo "Запуск скрипта транскрибации..."
    "$VENV_DIR/bin/python" transcribe.py --data_dir="data/segment_audio" --output_dir="data/dataset"
    echo -e "${GREEN}Транскрибация завершена.${NC}"
}

# Меню работы с датасетом
dataset_menu() {
    while true; do
        echo -e "\n${YELLOW}--- Меню обработки данных ---${NC}"
        echo "1. Умная сегментация аудио (из /data/audio в /data/segment_audio)"
        echo "2. Транскрибация аудио (из /data/segment_audio в /data/dataset)"
        echo "0. Назад в главное меню"
        echo -n "Выберите опцию: "
        read -r choice
        
        case $choice in
            1) segment_audio ;;
            2) transcribe_data ;;
            0) break ;;
            *) echo -e "${RED}Неверный выбор.${NC}" ;;
        esac
    done
}

# Главное меню
main_menu() {
    while true; do
        echo -e "\n${YELLOW}--- Главное меню подготовки проекта ---${NC}"
        echo "0. Выполнить все шаги по порядку (1 -> 2.1 -> 2.2)"
        echo "1. Настройка окружения и установка всех зависимостей"
        echo "2. Обработка данных (Сегментация и Транскрибация)"
        echo "---"
        echo "9. Выход"
        echo -n "Выберите опцию: "
        read -r choice
        
        case $choice in
            0)
                echo -e "${BLUE}--- Запуск всех шагов ---${NC}"
                install_environment
                segment_audio
                transcribe_data
                echo -e "${GREEN}Все шаги выполнены!${NC}"
                ;;
            1) install_environment ;;
            2) dataset_menu ;;
            9) exit 0 ;;
            *) echo -e "${RED}Неверный выбор.${NC}" ;;
        esac
    done
}

# --- Запуск ---
main_menu 