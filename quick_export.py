#!/usr/bin/env python3
# Быстрый экспорт для AI Assistant
from training_export_system import export_training_for_ai
import sys

if len(sys.argv) > 1:
    run_id = sys.argv[1]
    print(f"Экспорт run: {run_id}")
    export_training_for_ai(run_id)
else:
    print("Экспорт последнего обучения...")
    export_training_for_ai()
