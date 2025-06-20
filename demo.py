import matplotlib
matplotlib.use('Agg')  # Использовать неинтерактивный бэкэнд для Streamlit
import matplotlib.pylab as plt

import sys
sys.path.append('hifigan/')
import numpy as np
import torch
import librosa
import librosa.display
import math
import json
import os
import glob
import soundfile as sf
import librosa

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence, symbol_to_id
from text.cleaners import transliteration_ua_cleaners, english_cleaners, transliteration_cleaners,transliteration_cleaners_with_stress
from text.rudict import RuDict
from PIL import Image 
import time

from torch.nn import functional as F

from sklearn.metrics.pairwise import cosine_similarity as cs


from hifigan.meldataset import MAX_WAV_VALUE
from hifigan.models import Generator
from hifigan.env import AttrDict

from audio_processing import get_mel
import streamlit as st

def plot_data(st, data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower', 
                       interpolation='none')
    plt.savefig('out.png')
    image = Image.open('out.png')
    st.image(image, use_column_width=True)

def load_tts_model(path, hparams):
    model = load_model(hparams)
    model.load_state_dict(torch.load(path)['state_dict'])
    _ = model.cuda().eval().half()
    return model

@st.cache_resource
def load_vocoder_model(config_path, model_path):
    """Загрузить модель HiFiGAN с указанным конфигом и весами"""
    def load_checkpoint(filepath, device):
        assert os.path.isfile(filepath)
        checkpoint_dict = torch.load(filepath, map_location=device, weights_only=False)
        return checkpoint_dict

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Загрузка конфигурации
    with open(config_path, 'r') as fp:
        json_config = json.load(fp)
        h = AttrDict(json_config)
    
    generator = Generator(h).to(device)

    # Загрузка весов модели
    if os.path.exists(model_path):
        state_dict_g = load_checkpoint(model_path, device)
        generator.load_state_dict(state_dict_g['generator'])
    else:
        st.error(f"Модель HiFiGAN не найдена: {model_path}")
        return None
        
    generator.eval()
    generator.remove_weight_norm()
    return generator

def get_style_presets():
    """Получить предустановленные стили речи"""
    return {
        "Нейтральный": {
            "emotion": 0.0,
            "intonation": 0.0, 
            "timbre": 0.0,
            "style": 0.0,
            "expressiveness": 0.0,
            "description": "Спокойная, нейтральная речь"
        },
        "Радостный": {
            "emotion": 0.8,
            "intonation": 0.3,
            "timbre": 0.2,
            "style": -0.3,
            "expressiveness": 0.6,
            "description": "Веселая, позитивная речь"
        },
        "Грустный": {
            "emotion": -0.7,
            "intonation": -0.4,
            "timbre": -0.3,
            "style": 0.1,
            "expressiveness": -0.5,
            "description": "Печальная, меланхоличная речь"
        },
        "Официальный": {
            "emotion": -0.1,
            "intonation": -0.6,
            "timbre": -0.2,
            "style": 0.8,
            "expressiveness": -0.4,
            "description": "Формальная, деловая речь"
        },
        "Дружелюбный": {
            "emotion": 0.5,
            "intonation": 0.2,
            "timbre": 0.1,
            "style": -0.6,
            "expressiveness": 0.4,
            "description": "Теплая, дружественная речь"
        },
        "Энергичный": {
            "emotion": 0.6,
            "intonation": 0.5,
            "timbre": 0.4,
            "style": -0.2,
            "expressiveness": 0.8,
            "description": "Бодрая, энергичная речь"
        },
        "Спокойный": {
            "emotion": 0.1,
            "intonation": -0.2,
            "timbre": -0.1,
            "style": 0.2,
            "expressiveness": -0.6,
            "description": "Размеренная, умиротворенная речь"
        },
        "Удивленный": {
            "emotion": 0.3,
            "intonation": 0.7,
            "timbre": 0.3,
            "style": -0.1,
            "expressiveness": 0.5,
            "description": "Удивленная, вопросительная речь"
        },
        "Строгий": {
            "emotion": -0.3,
            "intonation": -0.5,
            "timbre": -0.4,
            "style": 0.7,
            "expressiveness": -0.2,
            "description": "Серьезная, авторитетная речь"
        },
        "Детский": {
            "emotion": 0.4,
            "intonation": 0.4,
            "timbre": 0.6,
            "style": -0.7,
            "expressiveness": 0.7,
            "description": "Детская, игривая речь"
        }
    }

def get_dominant_gst_token(emotion, intonation, timbre, style, expressiveness):
    """Определить доминирующий GST токен на основе пользовательских настроек"""
    # Создаем веса для каждого токена (предполагаем 10 токенов)
    token_weights = torch.zeros(10)
    
    # Распределяем веса по токенам в зависимости от настроек
    # Токены 0-1: Эмоциональность (радость/грусть)
    token_weights[0] = max(0, emotion)      # Позитивные эмоции
    token_weights[1] = max(0, -emotion)     # Негативные эмоции
    
    # Токены 2-3: Интонация (вопросительная/утвердительная)
    token_weights[2] = max(0, intonation)   # Вопросительная
    token_weights[3] = max(0, -intonation)  # Утвердительная
    
    # Токены 4-5: Тембр (высокий/низкий)
    token_weights[4] = max(0, timbre)       # Высокий тембр
    token_weights[5] = max(0, -timbre)      # Низкий тембр
    
    # Токены 6-7: Стиль (формальный/неформальный)
    token_weights[6] = max(0, style)        # Формальный
    token_weights[7] = max(0, -style)       # Неформальный
    
    # Токены 8-9: Выразительность (экспрессивная/монотонная)
    token_weights[8] = max(0, expressiveness)  # Экспрессивная
    token_weights[9] = max(0, -expressiveness) # Монотонная
    
    # Находим токен с максимальным весом
    dominant_token = torch.argmax(token_weights).item()
    max_weight = token_weights[dominant_token].item()
    
    # Возвращаем токен и силу влияния (scale)
    return dominant_token, max_weight

def inference(mel, generator, loundess=20):
    mel = mel.type(torch.float32)
    with torch.no_grad():
        y_g_hat = generator(mel)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = F.normalize(audio.detach(), dim=0).cpu().numpy()#.astype('int16')
        return audio * loundess 

def get_available_checkpoints():
    """Получить список доступных чекпоинтов из папки output/"""
    checkpoints = []
    
    # Поиск всех чекпоинтов в output/
    import glob
    checkpoint_patterns = [
        "output/*/checkpoint_*",
        "output/checkpoint_*",
        "weights/*/checkpoint_*",
        "weights/checkpoint_*"
    ]
    
    for pattern in checkpoint_patterns:
        found_checkpoints = glob.glob(pattern)
        checkpoints.extend(found_checkpoints)
    
    # Сортировка по времени создания (новые сначала)
    checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return checkpoints

def get_available_hifigan_configs():
    """Получить список доступных конфигов HiFiGAN"""
    configs = {}
    config_files = glob.glob("hifigan/config*.json")
    
    for config_file in config_files:
        config_name = os.path.basename(config_file).replace('.json', '')
        configs[config_name] = config_file
    
    return configs

def get_available_hifigan_models():
    """Получить список доступных моделей HiFiGAN с их конфигами"""
    models = {}
    
    # Поиск в UNIVERSAL_V1 с соответствующим конфигом
    universal_path = "hifigan/UNIVERSAL_V1/g_02500000"
    universal_config = "hifigan/UNIVERSAL_V1/config.json"
    if os.path.exists(universal_path) and os.path.exists(universal_config):
        models["UNIVERSAL_V1 (рекомендуется)"] = {
            "model_path": universal_path,
            "config_path": universal_config,
            "description": "Предобученная универсальная модель, лучшее качество"
        }
    
    # Поиск других моделей (используем совместимые конфиги)
    model_files = glob.glob("hifigan/g_*")
    for model_file in model_files:
        if os.path.isfile(model_file):
            model_name = os.path.basename(model_file)
            models[f"Local: {model_name}"] = {
                "model_path": model_file,
                "config_path": "hifigan/config.json",  # теперь совместимый конфиг
                "description": "Локальная модель"
            }
    
    return models

def main():
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    hparams.end_symbols_ids = [symbol_to_id[s] for s in '?!.']
    hparams.use_gst = True  # GST включен по умолчанию

    st.title("🎤 TTS Demo: Tacotron2 + HiFiGAN")
    st.markdown("---")

    st.sidebar.title("⚙️ Настройки модели")
    
    # Выбор чекпоинта Tacotron2
    st.sidebar.subheader("Модель Tacotron2")
    available_checkpoints = get_available_checkpoints()
    
    if not available_checkpoints:
        st.sidebar.error("Не найдено чекпоинтов!")
        st.error("Не найдено чекпоинтов для загрузки. Убедитесь, что модели обучены.")
        return
    
    # Форматирование имен чекпоинтов для отображения
    checkpoint_display_names = []
    for cp in available_checkpoints:
        # Показываем путь относительно корня проекта
        display_name = cp
        if cp.startswith("output/"):
            # Извлекаем информацию о эксперименте и итерации
            parts = cp.split('/')
            if len(parts) >= 3:
                exp_name = parts[1]
                checkpoint_name = parts[2]
                display_name = f"{exp_name} - {checkpoint_name}"
        checkpoint_display_names.append(display_name)
    
    selected_checkpoint_idx = st.sidebar.selectbox(
        "Выберите чекпоинт:",
        range(len(available_checkpoints)),
        format_func=lambda x: checkpoint_display_names[x]
    )
    checkpoint_path = available_checkpoints[selected_checkpoint_idx]
    
    # Кастомный путь к чекпоинту
    custom_path = st.sidebar.text_input('Или введите путь к чекпоинту:', value="")
    if custom_path.strip():
        checkpoint_path = custom_path.strip()
    
    # Выбор модели HiFiGAN (с автоматическим выбором конфига)
    st.sidebar.subheader("Модель HiFiGAN")
    available_models = get_available_hifigan_models()
    
    if not available_models:
        st.sidebar.error("Не найдено моделей HiFiGAN!")
        return
    
    selected_model_name = st.sidebar.selectbox(
        "Выберите модель:",
        list(available_models.keys()),
        help="Конфиг подбирается автоматически для каждой модели"
    )
    
    selected_model_info = available_models[selected_model_name]
    hifigan_model_path = selected_model_info["model_path"]
    hifigan_config_path = selected_model_info["config_path"]
    
    # Показать информацию о выбранной модели
    st.sidebar.info(f"**Описание:** {selected_model_info['description']}")
    st.sidebar.info(f"**Конфиг:** {os.path.basename(hifigan_config_path)}")
    
    # Дополнительный выбор конфига
    st.sidebar.subheader("Дополнительные настройки")
    
    override_config = st.sidebar.checkbox("Выбрать другой конфиг HiFiGAN", value=False)
    if override_config:
        available_configs = get_available_hifigan_configs()
        if available_configs:
            # Добавляем описания к конфигам
            config_descriptions = {
                "config": "Базовый конфиг (512 каналов)",
                "config_v1": "Вариант 1 (512 каналов)", 
                "config_v2": "Вариант 2 (512 каналов)",
                "config_v3": "Вариант 3 (512 каналов, resblock v2)"
            }
            
            config_options = []
            for config_name in available_configs.keys():
                desc = config_descriptions.get(config_name, "Конфиг HiFiGAN")
                config_options.append(f"{config_name} - {desc}")
            
            selected_config_display = st.sidebar.selectbox(
                "Выберите конфиг HiFiGAN:",
                config_options
            )
            
            # Извлекаем имя конфига из выбранного варианта
            selected_config_name = selected_config_display.split(" - ")[0]
            hifigan_config_path = available_configs[selected_config_name]
            st.sidebar.info(f"Выбран конфиг: {selected_config_name}")
    
    # Дополнительные настройки
    st.sidebar.subheader("Параметры генерации")
    seed = st.sidebar.text_input('Seed (для воспроизводимости)', value='42')
    seed = int(seed) if seed.strip() else None
    
    # Настройка GST
    use_gst_manual = st.sidebar.checkbox("Использовать GST (Global Style Tokens)", value=True)
    hparams.use_gst = use_gst_manual
    
    if 'gst' in checkpoint_path.lower():
        st.sidebar.info("ℹ️ Обнаружена GST модель в пути к чекпоинту")
    
    # Управление стилем речи (только если GST включен)
    emotion = 0.0
    intonation = 0.0
    timbre = 0.0
    style = 0.0
    expressiveness = 0.0
    
    if use_gst_manual:
        st.sidebar.subheader("🎭 Управление стилем речи")
        
        # Пресеты стилей
        style_presets = get_style_presets()
        preset_names = list(style_presets.keys())
        
        selected_preset = st.sidebar.selectbox(
            "Выберите готовый стиль:",
            preset_names,
            index=0,  # По умолчанию "Нейтральный"
            help="Готовые настройки стиля, которые автоматически настроят все параметры"
        )
        
        # Показать описание выбранного стиля
        if selected_preset in style_presets:
            st.sidebar.info(f"📝 {style_presets[selected_preset]['description']}")
        
        # Кнопка для применения пресета
        apply_preset = st.sidebar.button("🎯 Применить выбранный стиль")
        
        # Тонкая настройка параметров
        st.sidebar.markdown("### 🎛️ Тонкая настройка")
        
        # Получаем значения из пресета для инициализации слайдеров
        preset_values = style_presets.get(selected_preset, style_presets["Нейтральный"])
        
        # Если нажата кнопка применения пресета, используем значения пресета
        # Иначе используем значения по умолчанию (нейтральные)
        if apply_preset:
            default_emotion = preset_values["emotion"]
            default_intonation = preset_values["intonation"]
            default_timbre = preset_values["timbre"]
            default_style = preset_values["style"]
            default_expressiveness = preset_values["expressiveness"]
        else:
            default_emotion = 0.0
            default_intonation = 0.0
            default_timbre = 0.0
            default_style = 0.0
            default_expressiveness = 0.0
        
        # Слайдеры для точной настройки
        emotion = st.sidebar.slider(
            "😊 Эмоциональность",
            min_value=-1.0, max_value=1.0, value=default_emotion, step=0.1,
            help="От грустного (-1) до радостного (+1)"
        )
        
        intonation = st.sidebar.slider(
            "❓ Интонация", 
            min_value=-1.0, max_value=1.0, value=default_intonation, step=0.1,
            help="От утвердительной (-1) до вопросительной (+1)"
        )
        
        timbre = st.sidebar.slider(
            "🎵 Тембр",
            min_value=-1.0, max_value=1.0, value=default_timbre, step=0.1,
            help="От низкого (-1) до высокого (+1)"
        )
        
        style = st.sidebar.slider(
            "👔 Стиль",
            min_value=-1.0, max_value=1.0, value=default_style, step=0.1,
            help="От неформального (-1) до формального (+1)"
        )
        
        expressiveness = st.sidebar.slider(
            "🎭 Выразительность",
            min_value=-1.0, max_value=1.0, value=default_expressiveness, step=0.1,
            help="От монотонной (-1) до экспрессивной (+1)"
        )
        
        # Показать текущие значения
        if any([emotion != 0, intonation != 0, timbre != 0, style != 0, expressiveness != 0]):
            st.sidebar.markdown("### 📊 Текущие настройки:")
            st.sidebar.text(f"Эмоции: {emotion:+.1f}")
            st.sidebar.text(f"Интонация: {intonation:+.1f}")
            st.sidebar.text(f"Тембр: {timbre:+.1f}")
            st.sidebar.text(f"Стиль: {style:+.1f}")
            st.sidebar.text(f"Выразительность: {expressiveness:+.1f}")
        
        # Кнопка сброса
        if st.sidebar.button("🔄 Сбросить в нейтральный"):
            # Принудительно сбрасываем значения через session state
            for key in list(st.session_state.keys()):
                if any(param in key for param in ['emotion', 'intonation', 'timbre', 'style', 'expressiveness']):
                    del st.session_state[key]
            st.rerun()
    
    # Выбор типа очистки текста
    cleaner_type = st.sidebar.selectbox(
        "Тип очистки текста:",
        ["transliteration_cleaners_with_stress", "transliteration_cleaners", "english_cleaners"],
        index=0
    )
    
    # Громкость
    loudness = st.sidebar.slider("Громкость:", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
    
    # Основная панель
    st.header("📝 Ввод текста")
    
    # Предустановленные тексты
    predefined_texts = [
        "Привет! Это демо русского синтеза речи.",
        "Н+очь, +улица, фон+арь, апт+ека. Бессм+ысленный, и т+усклый св+ет.",
        "мн+е хот+елось б+ы сказ+ать к+ак я призн+ателен вс+ем прис+утсвующим сд+есь.",
        "Тв+орог или твор+ог, к+озлы или козл+ы, з+амок или зам+ок.", 
        "Вс+е смеш+алось в д+оме Обл+онских. Жен+а узн+ала, что муж был в св+язи с б+ывшею в их д+оме франц+уженкою-гуверн+анткой.",
        "Я открыл зам+ок и вошел в з+амок, сь+ев жарк+ое я п+онял как+ое сейч+ас ж+аркое лето в укра+ине.",
        "Молод+ой парн+ишка Т+анг С+ан одн+ажды оступ+ился и сл+едуя сво+им жел+аниям и пр+ихотям вор+ует секр+етные уч+ения в своей школе боев+ых искусств.",
        "Тетрагидропиранилциклопентилтетрагидропиридопиридиновые вещества",
        "Я открыл замок и вошел в замок, сьев жаркое я понял какое сейчас жаркое лето в украине.",
    ]
    
    selected_text = st.selectbox(
        "Выберите готовый текст:",
        ["(Введите собственный текст)"] + predefined_texts
    )
    
    if selected_text != "(Введите собственный текст)":
        input_text = selected_text
    else:
        input_text = ""
    
    # Поле ввода текста
    text_input = st.text_area(
        "Введите текст для синтеза:",
        value=input_text,
        height=150,
        help="Используйте знак '+' для обозначения ударения (например: прив+ет)"
    )
    
    # Кнопка генерации
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_button = st.button("🎵 Генерировать речь", type="primary", use_container_width=True)
    
    # Генерация речи
    if generate_button and text_input.strip():
        with st.spinner("Загрузка моделей..."):
            # Загрузка моделей
            model = load_tts_model(checkpoint_path, hparams)
            vocoder = load_vocoder_model(hifigan_config_path, hifigan_model_path)
            
            if model is None or vocoder is None:
                st.error("Ошибка загрузки моделей!")
                return
        
        with st.spinner("Генерация речи..."):
            try:
                start_time = time.perf_counter()
                
                # Подготовка текста
                sequence = np.array(text_to_sequence(text_input.strip(), [cleaner_type]))[None, :]
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                sequence = torch.from_numpy(sequence).to(device).long()
                
                # Подготовка GST параметров (если включен)
                token_idx = None
                scale = 1.0
                gst_info = None
                
                if hparams.use_gst and any([emotion != 0, intonation != 0, timbre != 0, style != 0, expressiveness != 0]):
                    token_idx, scale = get_dominant_gst_token(emotion, intonation, timbre, style, expressiveness)
                    # Усиливаем эффект на основе интенсивности настройки
                    scale = 1.0 + scale * 2.0  # Масштабируем от 1.0 до 3.0
                    
                    # Информация для отображения
                    token_names = [
                        "Радость", "Грусть", "Вопрос", "Утверждение", 
                        "Высокий тембр", "Низкий тембр", "Формальность", "Неформальность",
                        "Экспрессивность", "Монотонность"
                    ]
                    gst_info = f"{token_names[token_idx]} (сила: {scale:.1f})"
                
                # Генерация мел-спектрограммы
                with torch.no_grad():
                    if token_idx is not None:
                        # Используем конкретный GST токен
                        _, mel_outputs, mel_outputs_postnet, _, alignments, _ = model.inference(
                            sequence, seed=seed, token_idx=token_idx, scale=scale
                        )
                    else:
                        # Обычная генерация (автоматический GST)
                        _, mel_outputs, mel_outputs_postnet, _, alignments, _ = model.inference(
                            sequence, seed=seed
                        )
                
                # Генерация аудио
                audio = inference(mel_outputs_postnet, vocoder, loudness)
                
                generation_time = time.perf_counter() - start_time
                
                # Сохранение аудио
                output_path = 'generated_audio.wav'
                sf.write(output_path, audio, hparams.sampling_rate)
                
                # Отображение результатов
                st.success(f"✅ Речь сгенерирована за {generation_time:.2f} секунд")
                
                # Проигрывание аудио
                st.audio(output_path, format='audio/wav')
                
                # Визуализация
                st.header("📊 Визуализация")
                
                # Основные графики
                plot_data(st, (
                    mel_outputs.float().data.cpu().numpy()[0],
                    mel_outputs_postnet.float().data.cpu().numpy()[0],
                    alignments.float().data.cpu().numpy()[0].T
                ))
                
                # Визуализация GST токенов (если используется кастомный стиль)
                if hparams.use_gst and any([emotion != 0, intonation != 0, timbre != 0, style != 0, expressiveness != 0]):
                    st.subheader("🎭 Анализ стиля GST")
                    
                    # Создаем веса для визуализации
                    token_weights = torch.zeros(10)
                    token_weights[0] = max(0, emotion)
                    token_weights[1] = max(0, -emotion)
                    token_weights[2] = max(0, intonation)
                    token_weights[3] = max(0, -intonation)
                    token_weights[4] = max(0, timbre)
                    token_weights[5] = max(0, -timbre)
                    token_weights[6] = max(0, style)
                    token_weights[7] = max(0, -style)
                    token_weights[8] = max(0, expressiveness)
                    token_weights[9] = max(0, -expressiveness)
                    
                    # Названия токенов
                    token_names = [
                        "Радость", "Грусть", "Вопрос", "Утверждение",
                        "Высокий тембр", "Низкий тембр", "Формальность", "Неформальность",
                        "Экспрессивность", "Монотонность"
                    ]
                    
                    # Создаем DataFrame для графика
                    import pandas as pd
                    df = pd.DataFrame({
                        'Токен': token_names,
                        'Вес': token_weights.numpy()
                    })
                    
                    # Показываем только токены с ненулевым весом
                    df_filtered = df[df['Вес'] > 0]
                    
                    if len(df_filtered) > 0:
                        st.bar_chart(df_filtered.set_index('Токен')['Вес'])
                        st.caption(f"Доминирующий токен: **{token_names[token_idx]}** (вес: {token_weights[token_idx]:.2f})")
                    else:
                        st.info("Все параметры стиля установлены в нейтральное положение")
                
                # Информация о генерации
                st.header("ℹ️ Информация о генерации")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Параметры модели:**")
                    st.write(f"- Чекпоинт: `{os.path.basename(checkpoint_path)}`")
                    st.write(f"- HiFiGAN конфиг: `{os.path.basename(hifigan_config_path)}`")
                    st.write(f"- HiFiGAN модель: `{selected_model_name}`")
                    st.write(f"- GST: {'Включен' if hparams.use_gst else 'Выключен'}")
                
                with col2:
                    st.write("**Параметры генерации:**")
                    st.write(f"- Время генерации: {generation_time:.2f}s")
                    st.write(f"- Частота дискретизации: {hparams.sampling_rate} Hz")
                    st.write(f"- Длина аудио: {len(audio)/hparams.sampling_rate:.2f}s")
                    st.write(f"- Тип очистки: `{cleaner_type}`")
                    
                    # Информация о стиле (если GST включен)
                    if hparams.use_gst:
                        if gst_info is not None:
                            st.write("**Стиль речи:**")
                            st.write(f"- Доминирующий: {gst_info}")
                            st.write(f"- Эмоции: {emotion:+.1f}")
                            st.write(f"- Интонация: {intonation:+.1f}")
                            st.write(f"- Тембр: {timbre:+.1f}")
                            st.write(f"- Стиль: {style:+.1f}")
                            st.write(f"- Выразительность: {expressiveness:+.1f}")
                        else:
                            st.write("**Стиль речи:** Нейтральный (авто)")
                    
            except Exception as e:
                st.error(f"Ошибка при генерации: {str(e)}")
                import traceback
                st.text(traceback.format_exc())
    
    elif generate_button and not text_input.strip():
        st.warning("⚠️ Пожалуйста, введите текст для синтеза!")
    
    # Информация о приложении
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 О приложении")
    st.sidebar.markdown("""
    Этот интерфейс предоставляет возможность тестирования 
    русскоязычного TTS на основе Tacotron2 + HiFiGAN.
    
    **Возможности:**
    - Автоматический поиск чекпоинтов в output/
    - Выбор различных конфигураций HiFiGAN
    - Визуализация процесса синтеза
    - Поддержка ударений в тексте
    """)

if __name__ == "__main__":
    main()

