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

    # 5. Очистка от конфликтующих пакетов и установка зависимостей
    echo -e "\n${YELLOW}--> 1.5 Очистка от конфликтующих пакетов и установка зависимостей...${NC}"

    # Удаляем torchvision и torchaudio, т.к. они не используются и могут конфликтовать с версией torch
    echo "Удаление потенциально конфликтующих пакетов (torchvision, torchaudio)..."
    "$VENV_DIR/bin/pip" uninstall -y torchvision torchaudio &>/dev/null

    echo "Установка и обновление зависимостей из requirements.txt..."
    "$VENV_DIR/bin/pip" install --upgrade -r requirements.txt
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
    
    # Запрашиваем у пользователя размер пачки
    echo -e "${YELLOW}Введите размер пакета (batch size) для транскрибации.${NC}"
    echo "Рекомендации: 16 (для 24ГБ VRAM), 8 (для 16ГБ VRAM), 4 (для 8-12ГБ VRAM)."
    read -p "Ваш выбор (по умолчанию: 16): " BATCH_SIZE
    
    # Если пользователь ничего не ввел, используем значение по умолчанию
    : "${BATCH_SIZE:=8}"
    
    echo "Запуск скрипта транскрибации с размером пачки: $BATCH_SIZE..."
    "$VENV_DIR/bin/python" transcribe.py --data_dir="data/segment_audio" --output_dir="data/dataset" --batch_size="$BATCH_SIZE"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Во время транскрибации произошла ошибка. Если это 'Out of Memory', попробуйте уменьшить размер пачки.${NC}"
    else
        echo -e "${GREEN}Транскрибация завершена.${NC}"
    fi
}

# Функция для запуска процесса обучения
train_model() {
    echo -e "${BLUE}--- Шаг 3: Запуск умного обучения ---${NC}"
    
    # Проверка, существует ли виртуальное окружение
    if [ ! -f "$VENV_DIR/bin/python" ]; then
        echo -e "${RED}Python в виртуальном окружении не найден. Запустите сначала установку (пункт 1).${NC}"
        return
    fi
    
    # Проверка, существует ли Smart Tuner
    if [ ! -f "smart_tuner_main.py" ]; then
        echo -e "${RED}Smart Tuner (smart_tuner_main.py) не найден. Убедитесь, что все компоненты на месте.${NC}"
        return
    fi

    TRAIN_FILE="data/dataset/train.csv"
    if [ ! -f "$TRAIN_FILE" ] || [ ! -s "$TRAIN_FILE" ]; then
        echo -e "${RED}Файл с данными для обучения ($TRAIN_FILE) не найден или пуст.${NC}"
        echo -e "${YELLOW}Пожалуйста, сначала выполните шаги 2.1 (Сегментация) и 2.2 (Транскрибация).${NC}"
        return
    fi

    echo -e "${GREEN}🤖 Запуск Smart Tuner V2 в режиме полноценного обучения...${NC}"
    echo "Система автоматически:"
    echo "  ✅ Продолжит обучение с последнего чекпоинта"
    echo "  ✅ Будет использовать лучшие известные параметры"
    echo "  ✅ Остановит обучение при переобучении или стагнации"
    echo "  ✅ Сохранит лучшую модель и все логи в MLflow"
    echo "  ✅ Отправит уведомления в Telegram (если настроено)"
    echo -e "${YELLOW}Для остановки нажмите Ctrl+C в этом терминале.${NC}"
    echo

    # Настройка Telegram уведомлений (опционально)
    # Проверяем, не настроен ли Telegram в config.yaml
    if ! grep -q 'enabled: true' smart_tuner/config.yaml; then
        echo -e "${YELLOW}🔔 Настройка Telegram уведомлений (опционально):${NC}"
        echo "Для получения уведомлений о прогрессе обучения введите данные бота:"
        echo -n "Telegram Bot Token (Enter для пропуска): "
        read -r BOT_TOKEN
        
        if [ -n "$BOT_TOKEN" ]; then
            echo -n "Telegram Chat ID: "
            read -r CHAT_ID
            
            if [ -n "$CHAT_ID" ];
            then
                echo -e "${GREEN}✅ Настройка Telegram уведомлений...${NC}"
                # Обновляем конфигурацию
                sed -i 's/enabled: false/enabled: true/' smart_tuner/config.yaml
                sed -i "s/bot_token: .*/bot_token: \"$BOT_TOKEN\"/" smart_tuner/config.yaml
                sed -i "s/chat_id: .*/chat_id: \"$CHAT_ID\"/" smart_tuner/config.yaml
                echo "✓ Telegram уведомления включены"
            fi
        else
            echo "⏭️ Telegram уведомления пропущены"
        fi
    else
        echo "✅ Telegram уведомления уже настроены."
    fi

    # --- Подготовка и запуск ---
    echo -e "\n${YELLOW}🗑️  Полная очистка и подготовка к новому запуску...${NC}"
    
    # 1. Останавливаем все старые процессы
    pkill -f "tensorboard" &>/dev/null
    pkill -f "mlflow" &>/dev/null
    pkill -f "smart_tuner/web_interfaces.py" &>/dev/null
    sleep 1
    echo "✓ Старые процессы мониторинга остановлены"

    # 2. Удаляем старые логи и артефакты
    rm -rf output/ mlruns/ smart_tuner/models/ tensorboard.log mlflow.log smart_tuner_main.log smart_tuner/optuna_studies.db
    mkdir -p output/ mlruns/ smart_tuner/models/
    echo "✓ Старые логи и артефакты удалены, директории пересозданы"

    # 3. Запускаем систему мониторинга
    echo -e "\n${GREEN}📊 Запуск системы мониторинга...${NC}"
    IP_ADDR=$(hostname -I | awk '{print $1}')
    if [ -z "$IP_ADDR" ]; then
        IP_ADDR="localhost"
    fi

    nohup "$VENV_DIR/bin/python" -m tensorboard.main --logdir "output/" --host 0.0.0.0 --port 5001 --reload_interval 5 > tensorboard.log 2>&1 &
    echo "✓ TensorBoard запущен на порту 5001"

    nohup "$VENV_DIR/bin/mlflow" ui --host 0.0.0.0 --port 5000 --backend-store-uri "file://$(pwd)/mlruns" > mlflow.log 2>&1 &
    echo "✓ MLflow UI запущен на порту 5000"
    sleep 3

    echo -e "\n${BLUE}📈 Мониторинг будет доступен по адресам (через ~1-2 минуты):${NC}"
    echo -e "  MLflow:      ${GREEN}http://${IP_ADDR}:5000${NC}"
    echo -e "  TensorBoard: ${GREEN}http://${IP_ADDR}:5001${NC}"
    echo

    # 4. Запускаем основной процесс Smart Tuner
    echo -e "${GREEN}🚀 Запуск Smart Tuner...${NC}"
    "$VENV_DIR/bin/python" smart_tuner_main.py --mode train

    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}🎉 Обучение успешно завершено!${NC}"
        echo -e "${YELLOW}Результаты сохранены в:${NC}"
        echo "  📁 Модели: output/ и smart_tuner/models/"
        echo "  📊 Логи: mlruns/"
        echo "  📋 Подробные логи: smart_tuner_main.log"
    else
        echo -e "\n${RED}❌ Во время обучения произошла ошибка.${NC}"
        echo -e "${YELLOW}Проверьте логи для диагностики:${NC}"
        echo "  tail -f smart_tuner_main.log"
    fi
}

# Функция запуска веб-демо TTS
run_tts_demo() {
    echo -e "${BLUE}--- Запуск веб-демо TTS ---${NC}"
    
    if [ ! -f "$VENV_DIR/bin/python" ]; then
        echo -e "${RED}Python в виртуальном окружении не найден. Запустите сначала установку (пункт 1).${NC}"
        return
    fi
    
    # Проверка наличия чекпоинтов
    if [ -z "$(find output/ -name 'checkpoint_*' 2>/dev/null)" ]; then
        echo -e "${YELLOW}⚠️ Чекпоинты не найдены в папке output/.${NC}"
        echo -e "${YELLOW}Убедитесь, что обучение было запущено хотя бы один раз.${NC}"
        echo -e "${YELLOW}Демо все равно запустится, но для генерации потребуются модели.${NC}"
        echo
    fi
    
    # Проверка установки Streamlit
    if ! "$VENV_DIR/bin/python" -c "import streamlit" &>/dev/null; then
        echo -e "${YELLOW}Streamlit не установлен. Устанавливаем...${NC}"
        "$VENV_DIR/bin/pip" install streamlit
    fi
    
    IP_ADDR=$(hostname -I | awk '{print $1}')
    if [ -z "$IP_ADDR" ]; then
        IP_ADDR="localhost"
    fi
    
    echo -e "${GREEN}🎤 Запуск TTS Demo на порту 5005...${NC}"
    echo -e "${BLUE}Откройте в браузере: ${GREEN}http://${IP_ADDR}:5005${NC}"
    echo -e "${YELLOW}Для остановки нажмите Ctrl+C${NC}"
    echo
    
    # Запуск Streamlit
    "$VENV_DIR/bin/streamlit" run demo.py \
        --server.port 5005 \
        --server.address 0.0.0.0 \
        --browser.gatherUsageStats false
}

# Функция отладки обучения
debug_training() {
    echo -e "${BLUE}--- Запуск отладчика обучения ---${NC}"
    if [ ! -f "debug_train.sh" ]; then
        echo -e "${RED}Скрипт отладки 'debug_train.sh' не найден.${NC}"
        return
    fi
    bash ./debug_train.sh
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
        echo "0. Выполнить все шаги по порядку (1 -> 2.1 -> 2.2 -> 3)"
        echo "1. Настройка окружения и установка всех зависимостей"
        echo "2. Обработка данных (Сегментация и Транскрибация)"
        echo "3. Обучение модели"
        echo "4. Запуск веб-демо TTS (Streamlit на порту 5005)"
        echo "5. Отладить запуск обучения"
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
                train_model
                echo -e "${GREEN}Все шаги выполнены!${NC}"
                ;;
            1) install_environment ;;
            2) dataset_menu ;;
            3) train_model ;;
            4) run_tts_demo ;;
            5) debug_training ;;
            9) exit 0 ;;
            *) echo -e "${RED}Неверный выбор.${NC}" ;;
        esac
    done
}

# --- Запуск ---
main_menu 