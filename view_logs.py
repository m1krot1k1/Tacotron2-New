#!/usr/bin/env python3
# Просмотр последних логов
from pathlib import Path
import os

smart_logs = Path("smart_logs")
if smart_logs.exists():
    sessions = list(smart_logs.glob("training_sessions/*.md"))
    if sessions:
        latest = max(sessions, key=os.path.getctime)
        print(f"Последний лог: {latest}")
        
        if input("Открыть? (y/n): ").lower() in ['y', 'yes', '']:
            os.system(f"cat '{latest}'")
    else:
        print("Логи не найдены")
else:
    print("Папка smart_logs не найдена")
