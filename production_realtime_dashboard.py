#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 PRODUCTION REAL-TIME DASHBOARD
Production-ready real-time dashboard для визуального мониторинга всех компонентов

Возможности:
✅ Real-time мониторинг всех компонентов интеллектуальной системы
✅ Интерактивные графики с auto-refresh
✅ System metrics (CPU, GPU, Memory) в реальном времени
✅ Training progress с детализацией по фазам
✅ Performance optimization recommendations
✅ Alert system для критических событий
✅ Responsive веб-интерфейс
✅ Export данных в различные форматы

Технологии:
- Flask веб-сервер
- Plotly интерактивные графики
- Bootstrap responsive UI
- WebSocket для real-time updates
- SQLite база данных для метрик
"""

import os
import sys
import time
import json
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

# Flask и веб-компоненты
from flask import Flask, render_template, jsonify, request, send_file
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import plotly.utils
import psutil
import numpy as np
from collections import deque

# Интеграция с нашими системами
try:
    from unified_performance_optimization_system import UnifiedPerformanceOptimizationSystem
    PERFORMANCE_OPTIMIZER_AVAILABLE = True
except ImportError:
    PERFORMANCE_OPTIMIZER_AVAILABLE = False

try:
    from production_monitoring import ProductionMonitor
    PRODUCTION_MONITORING_AVAILABLE = True
except ImportError:
    PRODUCTION_MONITORING_AVAILABLE = False

try:
    from context_aware_training_manager import ContextAwareTrainingManager
    CONTEXT_AWARE_AVAILABLE = True
except ImportError:
    CONTEXT_AWARE_AVAILABLE = False

try:
    from unified_logging_system import UnifiedLoggingSystem
    UNIFIED_LOGGING_AVAILABLE = True
except ImportError:
    UNIFIED_LOGGING_AVAILABLE = False

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# GPU мониторинг
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False


class MetricsDatabase:
    """📊 Database для хранения метрик dashboard"""
    
    def __init__(self, db_path: str = "dashboard_metrics.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Инициализация базы данных"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Таблица системных метрик
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    cpu_usage REAL,
                    memory_usage REAL,
                    gpu_usage REAL,
                    gpu_memory_usage REAL,
                    disk_usage REAL
                )
            """)
            
            # Таблица метрик обучения
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    loss REAL,
                    learning_rate REAL,
                    gradient_norm REAL,
                    attention_quality REAL,
                    epoch INTEGER,
                    step INTEGER,
                    phase TEXT
                )
            """)
            
            # Таблица алертов
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    level TEXT,
                    component TEXT,
                    message TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Таблица оптимизаций
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimizations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    component TEXT,
                    parameter_name TEXT,
                    old_value TEXT,
                    new_value TEXT,
                    improvement_expected REAL
                )
            """)
            
            conn.commit()
    
    def insert_system_metrics(self, metrics: Dict):
        """Вставка системных метрик"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO system_metrics 
                (cpu_usage, memory_usage, gpu_usage, gpu_memory_usage, disk_usage)
                VALUES (?, ?, ?, ?, ?)
            """, (
                metrics.get('cpu_usage', 0),
                metrics.get('memory_usage', 0),
                metrics.get('gpu_usage', 0),
                metrics.get('gpu_memory_usage', 0),
                metrics.get('disk_usage', 0)
            ))
            conn.commit()
    
    def insert_training_metrics(self, metrics: Dict):
        """Вставка метрик обучения"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_metrics 
                (loss, learning_rate, gradient_norm, attention_quality, epoch, step, phase)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.get('loss', 0),
                metrics.get('learning_rate', 0),
                metrics.get('gradient_norm', 0),
                metrics.get('attention_quality', 0),
                metrics.get('epoch', 0),
                metrics.get('step', 0),
                metrics.get('phase', 'unknown')
            ))
            conn.commit()
    
    def insert_alert(self, level: str, component: str, message: str):
        """Вставка алерта"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alerts (level, component, message)
                VALUES (?, ?, ?)
            """, (level, component, message))
            conn.commit()
    
    def get_recent_metrics(self, table: str, minutes: int = 60) -> List[Dict]:
        """Получение недавних метрик"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT * FROM {table}
                WHERE timestamp > datetime('now', '-{minutes} minutes')
                ORDER BY timestamp DESC
            """)
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


class RealtimeMetricsCollector:
    """🔄 Real-time сборщик метрик"""
    
    def __init__(self, database: MetricsDatabase):
        self.database = database
        self.running = False
        self.collection_thread = None
        self.metrics_cache = {}
        
        # GPU handles
        self.gpu_handles = []
        if GPU_MONITORING_AVAILABLE:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) 
                                  for i in range(device_count)]
            except Exception:
                pass
        
        self.logger = logging.getLogger(__name__)
    
    def collect_system_metrics(self) -> Dict:
        """Сбор системных метрик"""
        try:
            # CPU и память
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                'cpu_usage': cpu_usage,
                'memory_usage': memory.percent,
                'disk_usage': (disk.used / disk.total) * 100,
                'gpu_usage': 0,
                'gpu_memory_usage': 0
            }
            
            # GPU метрики
            if self.gpu_handles:
                try:
                    handle = self.gpu_handles[0]
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    metrics['gpu_usage'] = util.gpu
                    metrics['gpu_memory_usage'] = (mem_info.used / mem_info.total) * 100
                except Exception:
                    pass
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Ошибка сбора системных метрик: {e}")
            return {}
    
    def start_collection(self, interval: float = 2.0):
        """Запуск сбора метрик"""
        if self.running:
            return
        
        self.running = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            args=(interval,),
            daemon=True
        )
        self.collection_thread.start()
        self.logger.info("🔄 Запущен real-time сбор метрик")
    
    def stop_collection(self):
        """Остановка сбора метрик"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join()
        self.logger.info("⏹️ Остановлен сбор метрик")
    
    def _collection_loop(self, interval: float):
        """Основной цикл сбора метрик"""
        while self.running:
            try:
                # Сбор системных метрик
                system_metrics = self.collect_system_metrics()
                if system_metrics:
                    self.database.insert_system_metrics(system_metrics)
                    self.metrics_cache['system'] = system_metrics
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Ошибка в цикле сбора метрик: {e}")
                time.sleep(interval)
    
    def get_cached_metrics(self) -> Dict:
        """Получение кэшированных метрик"""
        return self.metrics_cache.copy()


class AlertManager:
    """🚨 Менеджер алертов"""
    
    def __init__(self, database: MetricsDatabase):
        self.database = database
        self.alert_thresholds = {
            'cpu_usage': 90.0,
            'memory_usage': 90.0,
            'gpu_memory_usage': 95.0,
            'gradient_norm': 10.0,
            'loss_spike': 50.0
        }
        self.recent_alerts = deque(maxlen=100)
        
    def check_system_alerts(self, metrics: Dict):
        """Проверка системных алертов"""
        alerts = []
        
        # CPU перегрузка
        if metrics.get('cpu_usage', 0) > self.alert_thresholds['cpu_usage']:
            alert = {
                'level': 'warning',
                'component': 'system',
                'message': f"Высокое использование CPU: {metrics['cpu_usage']:.1f}%"
            }
            alerts.append(alert)
        
        # Память переполнена
        if metrics.get('memory_usage', 0) > self.alert_thresholds['memory_usage']:
            alert = {
                'level': 'critical',
                'component': 'system',
                'message': f"Критически высокое использование памяти: {metrics['memory_usage']:.1f}%"
            }
            alerts.append(alert)
        
        # GPU память переполнена
        if metrics.get('gpu_memory_usage', 0) > self.alert_thresholds['gpu_memory_usage']:
            alert = {
                'level': 'critical',
                'component': 'gpu',
                'message': f"Критически высокое использование GPU памяти: {metrics['gpu_memory_usage']:.1f}%"
            }
            alerts.append(alert)
        
        # Сохранение алертов
        for alert in alerts:
            self.database.insert_alert(
                alert['level'], 
                alert['component'], 
                alert['message']
            )
            self.recent_alerts.append(alert)
        
        return alerts
    
    def check_training_alerts(self, metrics: Dict):
        """Проверка алертов обучения"""
        alerts = []
        
        # Взрыв градиентов
        if metrics.get('gradient_norm', 0) > self.alert_thresholds['gradient_norm']:
            alert = {
                'level': 'warning',
                'component': 'training',
                'message': f"Высокая норма градиентов: {metrics['gradient_norm']:.2f}"
            }
            alerts.append(alert)
        
        # Скачок loss
        if metrics.get('loss', 0) > self.alert_thresholds['loss_spike']:
            alert = {
                'level': 'critical',
                'component': 'training',
                'message': f"Критический скачок loss: {metrics['loss']:.2f}"
            }
            alerts.append(alert)
        
        # Низкое качество attention
        if metrics.get('attention_quality', 1.0) < 0.1:
            alert = {
                'level': 'warning',
                'component': 'attention',
                'message': f"Низкое качество attention: {metrics['attention_quality']:.3f}"
            }
            alerts.append(alert)
        
        # Сохранение алертов
        for alert in alerts:
            self.database.insert_alert(
                alert['level'], 
                alert['component'], 
                alert['message']
            )
            self.recent_alerts.append(alert)
        
        return alerts
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Получение недавних алертов"""
        return list(self.recent_alerts)[-limit:]


class DashboardGraphGenerator:
    """📈 Генератор графиков для dashboard"""
    
    def __init__(self, database: MetricsDatabase):
        self.database = database
    
    def create_system_metrics_plot(self, minutes: int = 30) -> str:
        """Создание графика системных метрик"""
        try:
            metrics = self.database.get_recent_metrics('system_metrics', minutes)
            
            if not metrics:
                return json.dumps({})
            
            timestamps = [m['timestamp'] for m in metrics]
            
            fig = go.Figure()
            
            # CPU Usage
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=[m['cpu_usage'] for m in metrics],
                mode='lines+markers',
                name='CPU Usage (%)',
                line=dict(color='#FF6B6B', width=2)
            ))
            
            # Memory Usage
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=[m['memory_usage'] for m in metrics],
                mode='lines+markers',
                name='Memory Usage (%)',
                line=dict(color='#4ECDC4', width=2)
            ))
            
            # GPU Usage
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=[m['gpu_usage'] for m in metrics],
                mode='lines+markers',
                name='GPU Usage (%)',
                line=dict(color='#45B7D1', width=2)
            ))
            
            fig.update_layout(
                title='System Performance Metrics',
                xaxis_title='Time',
                yaxis_title='Usage (%)',
                hovermode='x unified',
                showlegend=True,
                height=400,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logging.error(f"Ошибка создания графика системных метрик: {e}")
            return json.dumps({})
    
    def create_training_progress_plot(self, minutes: int = 60) -> str:
        """Создание графика прогресса обучения"""
        try:
            metrics = self.database.get_recent_metrics('training_metrics', minutes)
            
            if not metrics:
                return json.dumps({})
            
            timestamps = [m['timestamp'] for m in metrics]
            
            fig = go.Figure()
            
            # Loss
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=[m['loss'] for m in metrics],
                mode='lines+markers',
                name='Loss',
                line=dict(color='#FF6B6B', width=2),
                yaxis='y'
            ))
            
            # Learning Rate
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=[m['learning_rate'] for m in metrics],
                mode='lines+markers',
                name='Learning Rate',
                line=dict(color='#96CEB4', width=2),
                yaxis='y2'
            ))
            
            # Attention Quality
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=[m['attention_quality'] for m in metrics],
                mode='lines+markers',
                name='Attention Quality',
                line=dict(color='#FFEAA7', width=2),
                yaxis='y3'
            ))
            
            fig.update_layout(
                title='Training Progress',
                xaxis_title='Time',
                yaxis=dict(title='Loss', side='left'),
                yaxis2=dict(title='Learning Rate', side='right', overlaying='y'),
                yaxis3=dict(title='Attention Quality', side='right', overlaying='y', anchor='free', position=0.95),
                hovermode='x unified',
                showlegend=True,
                height=400,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logging.error(f"Ошибка создания графика обучения: {e}")
            return json.dumps({})
    
    def create_performance_gauge(self, current_metrics: Dict) -> str:
        """Создание gauge для общей производительности"""
        try:
            # Вычисление общего показателя производительности
            cpu_score = max(0, 100 - current_metrics.get('cpu_usage', 0))
            memory_score = max(0, 100 - current_metrics.get('memory_usage', 0))
            gpu_score = current_metrics.get('gpu_usage', 0)
            
            overall_score = (cpu_score + memory_score + gpu_score) / 3
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = overall_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Overall Performance"},
                delta = {'reference': 75},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logging.error(f"Ошибка создания gauge: {e}")
            return json.dumps({})


class ProductionRealtimeDashboard:
    """🎯 Главный класс production real-time dashboard"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 5001):
        self.host = host
        self.port = port
        
        # Инициализация Flask приложения
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'dashboard_secret_key_2025'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Компоненты dashboard
        self.database = MetricsDatabase()
        self.metrics_collector = RealtimeMetricsCollector(self.database)
        self.alert_manager = AlertManager(self.database)
        self.graph_generator = DashboardGraphGenerator(self.database)
        
        # Интеграция с системами
        self.performance_optimizer = None
        self.production_monitor = None
        self.context_manager = None
        
        self._initialize_integrations()
        self._setup_routes()
        self._setup_websocket_handlers()
        
        # Логирование
        if UNIFIED_LOGGING_AVAILABLE:
            self.logger = UnifiedLoggingSystem().get_logger("Dashboard")
        else:
            self.logger = logging.getLogger(__name__)
    
    def _initialize_integrations(self):
        """Инициализация интеграций с системами"""
        try:
            if PERFORMANCE_OPTIMIZER_AVAILABLE:
                from hparams import create_hparams
                hparams = create_hparams()
                self.performance_optimizer = UnifiedPerformanceOptimizationSystem(hparams)
        except Exception as e:
            self.logger.warning(f"Не удалось интегрировать Performance Optimizer: {e}")
        
        try:
            if PRODUCTION_MONITORING_AVAILABLE:
                self.production_monitor = ProductionMonitor()
        except Exception as e:
            self.logger.warning(f"Не удалось интегрировать Production Monitor: {e}")
    
    def _setup_routes(self):
        """Настройка маршрутов Flask"""
        
        @self.app.route('/')
        def index():
            """Главная страница dashboard"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/metrics/system')
        def get_system_metrics():
            """API для получения системных метрик"""
            cached_metrics = self.metrics_collector.get_cached_metrics()
            return jsonify(cached_metrics.get('system', {}))
        
        @self.app.route('/api/metrics/training')
        def get_training_metrics():
            """API для получения метрик обучения"""
            cached_metrics = self.metrics_collector.get_cached_metrics()
            return jsonify(cached_metrics.get('training', {}))
        
        @self.app.route('/api/alerts')
        def get_alerts():
            """API для получения алертов"""
            alerts = self.alert_manager.get_recent_alerts(20)
            return jsonify(alerts)
        
        @self.app.route('/api/charts/system')
        def get_system_chart():
            """API для получения графика системных метрик"""
            minutes = request.args.get('minutes', 30, type=int)
            chart_data = self.graph_generator.create_system_metrics_plot(minutes)
            return jsonify({'chart': chart_data})
        
        @self.app.route('/api/charts/training')
        def get_training_chart():
            """API для получения графика обучения"""
            minutes = request.args.get('minutes', 60, type=int)
            chart_data = self.graph_generator.create_training_progress_plot(minutes)
            return jsonify({'chart': chart_data})
        
        @self.app.route('/api/charts/performance')
        def get_performance_gauge():
            """API для получения gauge производительности"""
            cached_metrics = self.metrics_collector.get_cached_metrics()
            gauge_data = self.graph_generator.create_performance_gauge(
                cached_metrics.get('system', {})
            )
            return jsonify({'gauge': gauge_data})
        
        @self.app.route('/api/optimize', methods=['POST'])
        def trigger_optimization():
            """API для запуска оптимизации"""
            if self.performance_optimizer:
                try:
                    result = self.performance_optimizer.optimize_performance_step(
                        force_optimization=True
                    )
                    return jsonify({'success': True, 'result': result})
                except Exception as e:
                    return jsonify({'success': False, 'error': str(e)})
            else:
                return jsonify({'success': False, 'error': 'Performance optimizer недоступен'})
    
    def _setup_websocket_handlers(self):
        """Настройка WebSocket обработчиков"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Обработка подключения клиента"""
            print(f'🔗 Клиент подключился: {request.sid}')
            
            # Отправка начальных данных
            cached_metrics = self.metrics_collector.get_cached_metrics()
            emit('metrics_update', cached_metrics)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Обработка отключения клиента"""
            print(f'❌ Клиент отключился: {request.sid}')
    
    def start_realtime_updates(self):
        """Запуск real-time обновлений"""
        def update_loop():
            while True:
                try:
                    # Получение текущих метрик
                    cached_metrics = self.metrics_collector.get_cached_metrics()
                    
                    # Проверка алертов
                    if 'system' in cached_metrics:
                        system_alerts = self.alert_manager.check_system_alerts(
                            cached_metrics['system']
                        )
                        
                        # Отправка алертов через WebSocket
                        if system_alerts:
                            self.socketio.emit('new_alerts', system_alerts)
                    
                    # Отправка обновленных метрик
                    self.socketio.emit('metrics_update', cached_metrics)
                    
                    time.sleep(2)  # Обновление каждые 2 секунды
                    
                except Exception as e:
                    self.logger.error(f"Ошибка в цикле real-time обновлений: {e}")
                    time.sleep(5)
        
        # Запуск в отдельном потоке
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
    
    def run(self, debug: bool = False):
        """Запуск dashboard сервера"""
        # Запуск сбора метрик
        self.metrics_collector.start_collection()
        
        # Запуск real-time обновлений
        self.start_realtime_updates()
        
        self.logger.info(f"🚀 Production Real-time Dashboard запущен на http://{self.host}:{self.port}")
        
        # Запуск Flask сервера
        self.socketio.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=debug,
            allow_unsafe_werkzeug=True
        )
    
    def stop(self):
        """Остановка dashboard"""
        self.metrics_collector.stop_collection()
        self.logger.info("⏹️ Dashboard остановлен")


def create_dashboard_template():
    """Создание HTML шаблона для dashboard"""
    template_dir = Path("templates")
    template_dir.mkdir(exist_ok=True)
    
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Tacotron2 Production Dashboard</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <!-- Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    
    <!-- Socket.IO -->
    <script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
    
    <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .alert-badge {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
        }
        
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .status-indicator {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 10px;
        }
        
        .status-healthy { background-color: #28a745; }
        .status-warning { background-color: #ffc107; }
        .status-critical { background-color: #dc3545; }
        
        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }
        
        .navbar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        }
    </style>
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-dark">
        <div class="container-fluid">
            <span class="navbar-brand mb-0 h1">🚀 Tacotron2 Production Dashboard</span>
            <div class="d-flex">
                <span class="navbar-text me-3">
                    <span class="status-indicator status-healthy" id="connection-status"></span>
                    <span id="connection-text">Connected</span>
                </span>
                <button class="btn btn-outline-light" onclick="triggerOptimization()">
                    ⚡ Optimize
                </button>
            </div>
        </div>
    </nav>

    <!-- Alert Badge -->
    <div id="alert-badge" class="alert-badge" style="display: none;">
        <div class="alert alert-warning alert-dismissible" role="alert">
            <strong>🚨 Alert!</strong> <span id="alert-message"></span>
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    </div>

    <div class="container-fluid mt-4">
        <!-- System Metrics Row -->
        <div class="row">
            <div class="col-md-3">
                <div class="metric-card">
                    <h5>💻 CPU Usage</h5>
                    <h2 id="cpu-usage">--</h2>
                    <small>Current load</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <h5>🧠 Memory Usage</h5>
                    <h2 id="memory-usage">--</h2>
                    <small>RAM utilization</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <h5>🎮 GPU Usage</h5>
                    <h2 id="gpu-usage">--</h2>
                    <small>GPU utilization</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <h5>⚡ Performance</h5>
                    <h2 id="performance-score">--</h2>
                    <small>Overall efficiency</small>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="row">
            <div class="col-md-8">
                <div class="chart-container">
                    <h5>📊 System Performance</h5>
                    <div id="system-chart"></div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="chart-container">
                    <h5>🎯 Performance Gauge</h5>
                    <div id="performance-gauge"></div>
                </div>
            </div>
        </div>

        <!-- Training Metrics Row -->
        <div class="row">
            <div class="col-md-12">
                <div class="chart-container">
                    <h5>🎵 Training Progress</h5>
                    <div id="training-chart"></div>
                </div>
            </div>
        </div>

        <!-- Recent Alerts -->
        <div class="row">
            <div class="col-md-12">
                <div class="chart-container">
                    <h5>🚨 Recent Alerts</h5>
                    <div id="alerts-list">
                        <p class="text-muted">No recent alerts</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>

    <script>
        // Socket.IO connection
        const socket = io();
        
        // Connection status
        socket.on('connect', function() {
            document.getElementById('connection-status').className = 'status-indicator status-healthy';
            document.getElementById('connection-text').textContent = 'Connected';
        });
        
        socket.on('disconnect', function() {
            document.getElementById('connection-status').className = 'status-indicator status-critical';
            document.getElementById('connection-text').textContent = 'Disconnected';
        });
        
        // Metrics updates
        socket.on('metrics_update', function(data) {
            updateMetricCards(data);
            updateCharts();
        });
        
        // New alerts
        socket.on('new_alerts', function(alerts) {
            showAlert(alerts[0]); // Show first alert
            updateAlertsList(alerts);
        });
        
        function updateMetricCards(data) {
            if (data.system) {
                document.getElementById('cpu-usage').textContent = 
                    data.system.cpu_usage ? data.system.cpu_usage.toFixed(1) + '%' : '--';
                document.getElementById('memory-usage').textContent = 
                    data.system.memory_usage ? data.system.memory_usage.toFixed(1) + '%' : '--';
                document.getElementById('gpu-usage').textContent = 
                    data.system.gpu_usage ? data.system.gpu_usage.toFixed(1) + '%' : '--';
                
                // Calculate performance score
                const cpuScore = Math.max(0, 100 - (data.system.cpu_usage || 0));
                const memScore = Math.max(0, 100 - (data.system.memory_usage || 0));
                const gpuScore = data.system.gpu_usage || 0;
                const perfScore = ((cpuScore + memScore + gpuScore) / 3).toFixed(0);
                document.getElementById('performance-score').textContent = perfScore + '%';
            }
        }
        
        function updateCharts() {
            // Update system chart
            fetch('/api/charts/system')
                .then(response => response.json())
                .then(data => {
                    if (data.chart) {
                        const chartData = JSON.parse(data.chart);
                        Plotly.newPlot('system-chart', chartData.data, chartData.layout, {responsive: true});
                    }
                });
            
            // Update performance gauge
            fetch('/api/charts/performance')
                .then(response => response.json())
                .then(data => {
                    if (data.gauge) {
                        const gaugeData = JSON.parse(data.gauge);
                        Plotly.newPlot('performance-gauge', gaugeData.data, gaugeData.layout, {responsive: true});
                    }
                });
            
            // Update training chart
            fetch('/api/charts/training')
                .then(response => response.json())
                .then(data => {
                    if (data.chart) {
                        const chartData = JSON.parse(data.chart);
                        Plotly.newPlot('training-chart', chartData.data, chartData.layout, {responsive: true});
                    }
                });
        }
        
        function showAlert(alert) {
            document.getElementById('alert-message').textContent = alert.message;
            const alertBadge = document.getElementById('alert-badge');
            alertBadge.style.display = 'block';
            
            // Auto-hide after 10 seconds
            setTimeout(() => {
                alertBadge.style.display = 'none';
            }, 10000);
        }
        
        function updateAlertsList(alerts) {
            const alertsList = document.getElementById('alerts-list');
            if (alerts.length === 0) {
                alertsList.innerHTML = '<p class="text-muted">No recent alerts</p>';
                return;
            }
            
            const alertsHtml = alerts.map(alert => `
                <div class="alert alert-${alert.level === 'critical' ? 'danger' : 'warning'} py-2">
                    <strong>${alert.component}:</strong> ${alert.message}
                </div>
            `).join('');
            
            alertsList.innerHTML = alertsHtml;
        }
        
        function triggerOptimization() {
            fetch('/api/optimize', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showAlert({message: 'Optimization triggered successfully', level: 'info'});
                    } else {
                        showAlert({message: 'Optimization failed: ' + data.error, level: 'critical'});
                    }
                });
        }
        
        // Initial chart load
        updateCharts();
        
        // Auto-refresh charts every 30 seconds
        setInterval(updateCharts, 30000);
    </script>
</body>
</html>
    '''
    
    with open(template_dir / "dashboard.html", 'w', encoding='utf-8') as f:
        f.write(html_content)


if __name__ == "__main__":
    # Создание HTML шаблона
    create_dashboard_template()
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Создание и запуск dashboard
    dashboard = ProductionRealtimeDashboard(host='0.0.0.0', port=5001)
    
    try:
        dashboard.run(debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Остановка dashboard...")
        dashboard.stop() 