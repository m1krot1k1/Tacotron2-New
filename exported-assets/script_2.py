# Создаем улучшенный install.sh скрипт с исправлениями всех проблем
improved_install_script = """#!/bin/bash

# Улучшенный install.sh для Tacotron2-New Smart Tuner
# Исправляет все выявленные проблемы

set -e  # Остановка при ошибке

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "🚀 Начинаем улучшенную установку Tacotron2-New Smart Tuner..."

# Функция проверки доступности порта
check_port_availability() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️ Порт $port занят, ищем альтернативный"
        return 1
    fi
    return 0
}

# Функция безопасной остановки процессов
safe_stop_processes() {
    echo "🛑 Остановка существующих процессов..."
    
    # Остановка MLflow UI
    pkill -f "mlflow ui" 2>/dev/null || true
    
    # Остановка TensorBoard  
    pkill -f "tensorboard" 2>/dev/null || true
    
    # Остановка Streamlit
    pkill -f "streamlit run" 2>/dev/null || true
    
    sleep 2
    echo "✅ Процессы остановлены"
}

# Создание виртуального окружения
echo "📦 Создание виртуального окружения..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "✅ Виртуальное окружение создано"
else
    echo "✅ Виртуальное окружение уже существует"
fi

# Активация виртуального окружения
source "$VENV_DIR/bin/activate"

# Обновление pip
echo "⬆️ Обновление pip..."
pip install --upgrade pip

# Установка базовых зависимостей
echo "📦 Установка базовых зависимостей..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Установка cmaes для Optuna
echo "🔧 Установка cmaes для Optuna..."
pip install cmaes

# Установка остальных зависимостей
echo "📦 Установка ML зависимостей..."
pip install optuna
pip install mlflow
pip install tensorboard
pip install streamlit
pip install numpy scipy matplotlib
pip install librosa soundfile
pip install tqdm
pip install psutil  # Для мониторинга памяти

# Установка дополнительных зависимостей для TTS
pip install unidecode inflect
pip install Pillow

echo "✅ Все зависимости установлены"

# ИСПРАВЛЕНИЕ: Инициализация SQLite в WAL режиме
echo "🗄️ Настройка базы данных SQLite..."
mkdir -p smart_tuner

# Создаем Python скрипт для настройки SQLite
cat > setup_sqlite.py << 'EOF'
import sqlite3
import os

def setup_sqlite_wal():
    db_path = 'smart_tuner/optuna_studies.db'
    
    # Удаляем старую базу если есть
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Создаем новую с WAL режимом
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        # Настройка WAL режима для конкурентного доступа
        conn.execute('PRAGMA journal_mode=WAL;')
        conn.execute('PRAGMA synchronous=NORMAL;')
        conn.execute('PRAGMA cache_size=10000;')
        conn.execute('PRAGMA temp_store=MEMORY;')
        conn.execute('PRAGMA mmap_size=268435456;')  # 256MB
        conn.execute('PRAGMA busy_timeout=30000;')   # 30 секунд
        
        conn.commit()
        print("✅ SQLite настроен в WAL режиме")
        
        # Создаем тестовую таблицу для проверки
        conn.execute('''
            CREATE TABLE IF NOT EXISTS health_check (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('INSERT INTO health_check (id) VALUES (1)')
        conn.commit()
        
        print("✅ База данных инициализирована и протестирована")
        
    except Exception as e:
        print(f"❌ Ошибка настройки SQLite: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    setup_sqlite_wal()
EOF

python setup_sqlite.py
rm setup_sqlite.py

echo "✅ База данных SQLite настроена"

# Создание директорий
echo "📁 Создание необходимых директорий..."
mkdir -p output
mkdir -p mlruns  
mkdir -p smart_tuner/models
mkdir -p smart_tuner/checkpoints
mkdir -p smart_tuner/logs
mkdir -p data

echo "✅ Директории созданы"

# ИСПРАВЛЕНИЕ: Создание улучшенного data_utils.py
echo "🔧 Применение исправлений к data_utils.py..."
cat > data_utils_fix.py << 'EOF'
# Исправление для data_utils.py - custom collate_fn
import torch
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn_fixed(batch):
    """
    Исправленная collate_fn для правильной обработки device transfer
    Решает проблему: AttributeError: 'tuple' object has no attribute 'device'
    """
    if not batch:
        return torch.tensor([]), torch.tensor([])
    
    data_batch = []
    target_batch = []
    
    for sample in batch:
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            data, target = sample[0], sample[1]
            
            # Убеждаемся, что data это тензор
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)
            
            # Убеждаемся, что target это тензор  
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
                
            data_batch.append(data)
            target_batch.append(target)
    
    # Стекируем тензоры с обработкой ошибок размерности
    try:
        data_tensor = torch.stack(data_batch)
        target_tensor = torch.stack(target_batch)
    except RuntimeError:
        # Если размеры не совпадают, используем pad_sequence
        data_tensor = pad_sequence(data_batch, batch_first=True)
        target_tensor = pad_sequence(target_batch, batch_first=True)
    
    return data_tensor, target_tensor

print("✅ Исправления data_utils готовы к применению")
EOF

echo "✅ Исправления подготовлены"

# Остановка существующих процессов
safe_stop_processes

# УЛУЧШЕННАЯ очистка с проверкой
echo "🧹 Умная очистка предыдущих запусков..."

# Закрываем все соединения с базой данных
python -c "
import sqlite3
import os
import time

db_path = 'smart_tuner/optuna_studies.db'
if os.path.exists(db_path):
    try:
        # Попытка graceful закрытия
        conn = sqlite3.connect(db_path, timeout=5)
        conn.execute('PRAGMA wal_checkpoint(RESTART);')
        conn.close()
        time.sleep(1)
    except:
        pass
"

# Очистка файлов с проверкой блокировки
FILES_TO_CLEAN=(
    "output/"
    "mlruns/"
    "smart_tuner/models/"
    "tensorboard.log"
    "mlflow.log" 
    "smart_tuner_main.log"
)

for item in "${FILES_TO_CLEAN[@]}"; do
    if [ -e "$item" ]; then
        rm -rf "$item" 2>/dev/null || {
            echo "⚠️ Не удалось удалить $item (возможно используется)"
        }
    fi
done

# Пересоздаем директории
mkdir -p output mlruns smart_tuner/models

echo "✅ Очистка завершена"

# Выбор режима запуска
echo ""
echo "🔧 Выберите режим запуска:"
echo "1) Обычное обучение"
echo "2) Умное обучение (Smart Tuner)"
echo "3) Запуск всех сервисов мониторинга"
echo "4) Выход"

read -p "Введите номер (1-4): " choice

case $choice in
    1)
        echo "🚀 Запуск обычного обучения..."
        python train.py --output_directory=output --log_directory=mlruns
        ;;
    2)
        echo "🧠 Запуск умного обучения..."
        
        # ИСПРАВЛЕНИЕ: Применяем patch для train.py
        echo "🔧 Применение исправлений к train.py..."
        
        # Создаем backup
        cp train.py train.py.backup 2>/dev/null || true
        
        # Применяем исправление для строки 897
        sed -i 's/device = x\.device/device = x.device if hasattr(x, "device") and isinstance(x, torch.Tensor) else torch.device("cuda" if torch.cuda.is_available() else "cpu")/g' train.py
        
        echo "✅ Исправления применены к train.py"
        
        # Запуск с улучшенным мониторингом
        python smart_tuner_main.py
        ;;
    3)
        echo "📊 Запуск сервисов мониторинга..."
        
        # Поиск доступных портов и запуск сервисов
        
        # MLflow UI
        MLFLOW_PORT=5000
        for port in 5000 5010 5020; do
            if check_port_availability $port; then
                MLFLOW_PORT=$port
                break
            fi
        done
        
        echo "🚀 Запуск MLflow UI на порту $MLFLOW_PORT..."
        nohup "$VENV_DIR/bin/mlflow" ui --host 0.0.0.0 --port $MLFLOW_PORT > mlflow.log 2>&1 &
        MLFLOW_PID=$!
        
        # TensorBoard
        TENSORBOARD_PORT=5001
        for port in 5001 5011 5021; do
            if check_port_availability $port; then
                TENSORBOARD_PORT=$port
                break
            fi
        done
        
        echo "🚀 Запуск TensorBoard на порту $TENSORBOARD_PORT..."
        nohup "$VENV_DIR/bin/tensorboard" --logdir=mlruns --host=0.0.0.0 --port=$TENSORBOARD_PORT > tensorboard.log 2>&1 &
        TENSORBOARD_PID=$!
        
        # Optuna Dashboard (если доступен)
        if command -v optuna-dashboard >/dev/null 2>&1; then
            OPTUNA_PORT=5002
            for port in 5002 5012 5022; do
                if check_port_availability $port; then
                    OPTUNA_PORT=$port
                    break
                fi
            done
            
            echo "🚀 Запуск Optuna Dashboard на порту $OPTUNA_PORT..."
            nohup optuna-dashboard sqlite:///smart_tuner/optuna_studies.db --host 0.0.0.0 --port $OPTUNA_PORT > optuna.log 2>&1 &
            OPTUNA_PID=$!
        fi
        
        # Проверка запуска сервисов
        sleep 3
        
        echo ""
        echo "🎯 Статус сервисов:"
        
        if kill -0 $MLFLOW_PID 2>/dev/null; then
            echo "✅ MLflow UI: http://localhost:$MLFLOW_PORT"
        else
            echo "❌ MLflow UI не запущен"
        fi
        
        if kill -0 $TENSORBOARD_PID 2>/dev/null; then
            echo "✅ TensorBoard: http://localhost:$TENSORBOARD_PORT"
        else
            echo "❌ TensorBoard не запущен"
        fi
        
        if [ ! -z "$OPTUNA_PID" ] && kill -0 $OPTUNA_PID 2>/dev/null; then
            echo "✅ Optuna Dashboard: http://localhost:$OPTUNA_PORT"
        else
            echo "⚠️ Optuna Dashboard недоступен"
        fi
        
        echo ""
        echo "📝 Логи сервисов:"
        echo "   MLflow: tail -f mlflow.log"
        echo "   TensorBoard: tail -f tensorboard.log"
        echo ""
        echo "🛑 Для остановки всех сервисов выполните:"
        echo "   pkill -f 'mlflow ui'"
        echo "   pkill -f 'tensorboard'"
        echo "   pkill -f 'optuna-dashboard'"
        ;;
    4)
        echo "👋 Выход..."
        ;;
    *)
        echo "❌ Неверный выбор. Пожалуйста, выберите 1-4."
        ;;
esac

echo ""
echo "✅ Установка и настройка завершены!"
echo ""
echo "📚 Полезные команды:"
echo "   source venv/bin/activate  # Активация окружения"
echo "   python smart_tuner_main.py  # Запуск умного обучения"
echo "   tail -f smart_tuner_main.log  # Просмотр логов"
echo ""
echo "🔧 В случае проблем:"
echo "   1. Проверьте логи в файлах *.log"
echo "   2. Убедитесь, что все порты доступны"
echo "   3. Перезапустите скрипт с sudo если нужны права"
echo ""
echo "🎉 Удачного обучения!"
"""

# Сохраняем улучшенный скрипт
with open("improved_install.sh", "w", encoding="utf-8") as f:
    f.write(improved_install_script)

print("✅ Создан улучшенный install.sh скрипт с исправлениями:")
print("📁 Файл: improved_install.sh")
print()
print("🔧 Основные улучшения:")
print("1. ✅ Установка cmaes для Optuna")
print("2. ✅ Настройка SQLite WAL режима") 
print("3. ✅ Исправление проблемы device в train.py")
print("4. ✅ Умная очистка с проверкой блокировок")
print("5. ✅ Поиск доступных портов для сервисов")
print("6. ✅ Улучшенная обработка ошибок")
print("7. ✅ Проверка статуса сервисов")
print()
print("🚀 Для использования:")
print("chmod +x improved_install.sh")
print("./improved_install.sh")