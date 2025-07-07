# Анализируем прикрепленный файл с логами обучения
import re
import pandas as pd
from datetime import datetime

# Читаем файл с логами
with open("paste.txt", "r", encoding="utf-8") as f:
    log_content = f.read()

# Извлекаем ключевые метрики из логов
loss_pattern = r"Шаг (\d+): Loss: ([\d.]+), Grad: ([\d.]+), Attn: ([\d.]+), Gate: ([\d.]+)"
guided_attention_pattern = r"guided attention weight: ([\d.]+) → ([\d.]+)"
lr_pattern = r"LR изменен на ([\d.e-]+)"
gradient_pattern = r"Градиенты обрезаны: ([\d.]+) → ([\d.]+)"

# Извлекаем данные по шагам
step_data = []
for match in re.finditer(loss_pattern, log_content):
    step, loss, grad, attn, gate = match.groups()
    step_data.append({
        'step': int(step),
        'loss': float(loss),
        'grad_norm': float(grad), 
        'attention': float(attn),
        'gate_accuracy': float(gate)
    })

# Создаем DataFrame для анализа
if step_data:
    df = pd.DataFrame(step_data)
    print("=== АНАЛИЗ МЕТРИК ОБУЧЕНИЯ ===")
    print(f"Количество записанных шагов: {len(df)}")
    print(f"Диапазон шагов: {df['step'].min()} - {df['step'].max()}")
    print("\n📊 СТАТИСТИКА МЕТРИК:")
    print(df.describe())
    
    print("\n🔍 АНАЛИЗ ПРОБЛЕМ:")
    
    # Анализ loss
    initial_loss = df['loss'].iloc[0]
    final_loss = df['loss'].iloc[-1]
    print(f"• Loss: {initial_loss:.2f} → {final_loss:.2f} (изменение: {((final_loss-initial_loss)/initial_loss*100):+.1f}%)")
    
    # Анализ attention
    avg_attention = df['attention'].mean()
    print(f"• Средняя attention diagonality: {avg_attention:.3f} (норма: >0.7)")
    if avg_attention < 0.1:
        print("  ❌ КРИТИЧЕСКИ НИЗКАЯ attention diagonality")
    elif avg_attention < 0.3:
        print("  ⚠️ Низкая attention diagonality")
        
    # Анализ gate accuracy
    avg_gate = df['gate_accuracy'].mean()
    print(f"• Средняя gate accuracy: {avg_gate:.3f} (норма: >0.8)")
    if avg_gate < 0.5:
        print("  ⚠️ Низкая gate accuracy")
        
    # Анализ градиентов
    avg_grad = df['grad_norm'].mean()
    print(f"• Средняя норма градиентов: {avg_grad:.2f}")
    
    # Подсчет автоматических исправлений
    guided_attention_changes = len(re.findall(guided_attention_pattern, log_content))
    lr_changes = len(re.findall(lr_pattern, log_content))
    gradient_clips = len(re.findall(gradient_pattern, log_content))
    
    print(f"\n🔧 АВТОМАТИЧЕСКИЕ ИСПРАВЛЕНИЯ:")
    print(f"• Изменения guided attention weight: {guided_attention_changes}")
    print(f"• Изменения learning rate: {lr_changes}")
    print(f"• Обрезания градиентов: {gradient_clips}")
    
    # Проверяем стабильность обучения
    loss_std = df['loss'].std()
    print(f"\n📈 СТАБИЛЬНОСТЬ:")
    print(f"• Стандартное отклонение loss: {loss_std:.2f}")
    if loss_std > 5:
        print("  ⚠️ Высокая нестабильность loss")
        
else:
    print("❌ Не удалось извлечь данные из логов")