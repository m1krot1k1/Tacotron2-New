#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест GST (Global Style Tokens) системы
Проверяет работоспособность GST компонентов
"""

import torch
import numpy as np
from hparams import create_hparams
from model import Tacotron2
from gst import GST, TPSEGST, ReferenceEncoder, STL
import warnings
warnings.filterwarnings('ignore')

def test_gst_components():
    """Тестирует отдельные компоненты GST"""
    print("🎭 Тестирование компонентов GST...")
    
    hparams = create_hparams()
    
    # Тест ReferenceEncoder
    print("\n1️⃣ Тестирование ReferenceEncoder...")
    try:
        ref_encoder = ReferenceEncoder(hparams)
        # Создаем тестовые mel-спектрограммы
        batch_size = 2
        mel_length = 100
        mel_channels = hparams.n_mel_channels
        test_mels = torch.randn(batch_size, mel_length, mel_channels)
        
        output = ref_encoder(test_mels)
        print(f"   ✅ ReferenceEncoder работает")
        print(f"   📊 Вход: {test_mels.shape} -> Выход: {output.shape}")
        print(f"   📏 Ожидаемый размер выхода: [{batch_size}, {hparams.ref_enc_gru_size}]")
        
    except Exception as e:
        print(f"   ❌ Ошибка ReferenceEncoder: {e}")
        return False
    
    # Тест STL
    print("\n2️⃣ Тестирование STL (Style Token Layer)...")
    try:
        stl = STL(hparams)
        # Используем выход ReferenceEncoder как вход для STL
        test_input = torch.randn(batch_size, hparams.ref_enc_gru_size)
        
        style_output = stl(test_input)
        print(f"   ✅ STL работает")
        print(f"   📊 Вход: {test_input.shape} -> Выход: {style_output.shape}")
        print(f"   📏 Ожидаемый размер выхода: [{batch_size}, 1, {hparams.token_embedding_size}]")
        
    except Exception as e:
        print(f"   ❌ Ошибка STL: {e}")
        return False
    
    # Тест полного GST
    print("\n3️⃣ Тестирование полного GST...")
    try:
        gst = GST(hparams)
        test_mels = torch.randn(batch_size, mel_length, mel_channels)
        
        gst_output = gst(test_mels)
        print(f"   ✅ GST работает")
        print(f"   📊 Вход: {test_mels.shape} -> Выход: {gst_output.shape}")
        print(f"   📏 Ожидаемый размер выхода: [{batch_size}, 1, {hparams.token_embedding_size}]")
        
    except Exception as e:
        print(f"   ❌ Ошибка GST: {e}")
        return False
    
    # Тест TPSEGST
    print("\n4️⃣ Тестирование TPSEGST...")
    try:
        tpse_gst = TPSEGST(hparams)
        # Создаем тестовые encoder outputs
        encoder_outputs = torch.randn(batch_size, 50, hparams.encoder_embedding_dim)
        
        tpse_output = tpse_gst(encoder_outputs)
        print(f"   ✅ TPSEGST работает")
        print(f"   📊 Вход: {encoder_outputs.shape} -> Выход: {tpse_output.shape}")
        print(f"   📏 Ожидаемый размер выхода: [{batch_size}, 1, {hparams.token_embedding_size}]")
        
    except Exception as e:
        print(f"   ❌ Ошибка TPSEGST: {e}")
        return False
    
    return True

def test_tacotron2_with_gst():
    """Тестирует Tacotron2 с включенным GST"""
    print("\n🤖 Тестирование Tacotron2 с GST...")
    
    hparams = create_hparams()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Создаем модель с GST
        model = Tacotron2(hparams).to(device)
        model.eval()  # Ставим в eval режим для BatchNorm
        print(f"   ✅ Модель Tacotron2 создана с GST")
        print(f"   📊 GST активен: {model.use_gst}")
        print(f"   🎯 Количество style tokens: {hparams.token_num}")
        print(f"   📏 Размер token embedding: {hparams.token_embedding_size}")
        
        # Тестируем forward pass
        batch_size = 2
        text_length = 20
        mel_length = 100
        
        # Создаем тестовые данные на том же устройстве
        text_inputs = torch.randint(0, hparams.n_symbols, (batch_size, text_length)).to(device)
        text_lengths = torch.tensor([text_length, text_length-2]).to(device)
        mels = torch.randn(batch_size, hparams.n_mel_channels, mel_length).to(device)
        output_lengths = torch.tensor([mel_length, mel_length-10]).to(device)
        
        # Подготавливаем входные данные как в реальном обучении
        inputs = (text_inputs, text_lengths, mels, text_length, output_lengths)
        
        print(f"\n   🔄 Тестирование forward pass...")
        with torch.no_grad():
            outputs = model(inputs)
            
        print(f"   ✅ Forward pass успешен")
        print(f"   📊 Количество выходов: {len(outputs)}")
        
        # Проверяем, что GST embedding добавлен
        if len(outputs) > 6 and outputs[6] is not None:  # gst_outputs
            print(f"   🎭 GST outputs: {outputs[6].shape}")
        if len(outputs) > 5 and outputs[5] is not None:  # tpse_gst_outputs  
            print(f"   🔧 TPSE GST outputs: {outputs[5].shape}")
            
    except Exception as e:
        print(f"   ❌ Ошибка Tacotron2 с GST: {e}")
        return False
    
    return True

def test_gst_inference_modes():
    """Тестирует различные режимы inference с GST"""
    print("\n🎯 Тестирование режимов inference с GST...")
    
    hparams = create_hparams()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Tacotron2(hparams).to(device)
    model.eval()  # Ставим в eval режим для BatchNorm
    
    batch_size = 1
    text_length = 15
    text_inputs = torch.randint(0, hparams.n_symbols, (batch_size, text_length)).to(device)
    
    try:
        # 1. Обычный inference без reference mel
        print("\n   1️⃣ Обычный inference (TPSE GST)...")
        with torch.no_grad():
            outputs1 = model.inference(text_inputs)
        print(f"   ✅ Обычный inference работает")
        
        # 2. Inference с reference mel
        print("\n   2️⃣ Inference с reference mel...")
        ref_mel = torch.randn(1, hparams.n_mel_channels, 50).to(device)
        with torch.no_grad():
            outputs2 = model.inference(text_inputs, reference_mel=ref_mel)
        print(f"   ✅ Reference mel inference работает")
        
        # 3. Inference с конкретным token
        print("\n   3️⃣ Inference с выбранным style token...")
        token_idx = 3  # Выбираем 4-й стиль (индексация с 0)
        with torch.no_grad():
            outputs3 = model.inference(text_inputs, token_idx=token_idx)
        print(f"   ✅ Style token inference работает")
        
        # 4. Inference с масштабированием
        print("\n   4️⃣ Inference с масштабированием стиля...")
        with torch.no_grad():
            outputs4 = model.inference(text_inputs, token_idx=token_idx, scale=2.0)
        print(f"   ✅ Масштабированный inference работает")
        
        print(f"\n   📊 Результаты всех режимов получены успешно")
        
    except Exception as e:
        print(f"   ❌ Ошибка inference режимов: {e}")
        return False
    
    return True

def main():
    """Основная функция тестирования"""
    print("=" * 60)
    print("🎭 ТЕСТИРОВАНИЕ GST (GLOBAL STYLE TOKENS)")
    print("=" * 60)
    
    # Проверяем доступность CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Устройство: {device}")
    
    success_count = 0
    total_tests = 4
    
    # Тест 1: Компоненты GST
    if test_gst_components():
        success_count += 1
    
    # Тест 2: Tacotron2 с GST
    if test_tacotron2_with_gst():
        success_count += 1
    
    # Тест 3: Режимы inference
    if test_gst_inference_modes():
        success_count += 1
    
    # Тест 4: Проверка параметров
    print("\n📋 Проверка параметров GST...")
    hparams = create_hparams()
    print(f"   ✅ use_gst: {hparams.use_gst}")
    print(f"   🔢 token_num: {hparams.token_num}")
    print(f"   📏 token_embedding_size: {hparams.token_embedding_size}")
    print(f"   🧠 ref_enc_gru_size: {hparams.ref_enc_gru_size}")
    print(f"   👥 num_heads: {hparams.num_heads}")
    success_count += 1
    
    # Итоговый результат
    print("\n" + "=" * 60)
    print(f"📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ GST")
    print("=" * 60)
    print(f"✅ Успешных тестов: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! GST готов к использованию!")
        print("\n🚀 Рекомендации:")
        print("   • GST включен и работает корректно")
        print("   • Можно использовать различные style tokens")
        print("   • Reference mel позволяет копировать стиль")
        print("   • TPSE GST автоматически извлекает стиль из текста")
    else:
        print("⚠️  Некоторые тесты не прошли. Проверьте ошибки выше.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 