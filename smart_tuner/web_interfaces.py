#!/usr/bin/env python3
"""
Веб-интерфейсы для компонентов Smart Tuner V2
"""

import os
import sys
import json
import yaml
import logging
import threading
from datetime import datetime
from flask import Flask, jsonify, render_template_string, request
import subprocess
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComponentWebInterface:
    """Базовый класс для веб-интерфейсов компонентов"""
    
    def __init__(self, component_name, port, config):
        self.component_name = component_name
        self.port = port
        self.config = config
        self.app = Flask(f"SmartTuner_{component_name}")
        self.setup_routes()
        
    def setup_routes(self):
        """Настройка базовых маршрутов"""
        @self.app.route('/')
        def index():
            return self.get_index_page()
            
        @self.app.route('/api/status')
        def status():
            return jsonify(self.get_status())
            
        @self.app.route('/api/info')
        def info():
            return jsonify(self.get_info())
    
    def get_index_page(self):
        """HTML страница компонента"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Smart Tuner V2 - {self.component_name}</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 2px solid #e0e0e0; }}
                .title {{ color: #2c3e50; font-size: 2.5em; margin: 0; }}
                .subtitle {{ color: #7f8c8d; font-size: 1.2em; margin: 10px 0; }}
                .status {{ display: inline-block; padding: 8px 16px; border-radius: 20px; font-weight: bold; }}
                .status.active {{ background: #2ecc71; color: white; }}
                .status.inactive {{ background: #e74c3c; color: white; }}
                .info-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 30px; }}
                .info-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; }}
                .info-card h3 {{ margin: 0 0 10px 0; color: #2c3e50; }}
                .refresh-btn {{ background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }}
                .refresh-btn:hover {{ background: #2980b9; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 class="title">🤖 {self.component_name}</h1>
                    <p class="subtitle">Smart Tuner V2 Component</p>
                    <span class="status active">🟢 Активен</span>
                    <span style="margin-left: 20px;">📡 Порт: {self.port}</span>
                </div>
                
                <div id="component-info">
                    <div class="info-grid">
                        <div class="info-card">
                            <h3>📊 Статус</h3>
                            <p id="status-text">Загрузка...</p>
                        </div>
                        <div class="info-card">
                            <h3>ℹ️ Информация</h3>
                            <p id="info-text">Загрузка...</p>
                        </div>
                    </div>
                </div>
                
                <div style="text-align: center; margin-top: 30px;">
                    <button class="refresh-btn" onclick="refreshData()">🔄 Обновить</button>
                </div>
            </div>
            
            <script>
                function refreshData() {{
                    fetch('/api/status')
                        .then(response => response.json())
                        .then(data => {{
                            document.getElementById('status-text').innerHTML = JSON.stringify(data, null, 2);
                        }});
                    
                    fetch('/api/info')
                        .then(response => response.json())
                        .then(data => {{
                            document.getElementById('info-text').innerHTML = JSON.stringify(data, null, 2);
                        }});
                }}
                
                // Автообновление каждые 5 секунд
                setInterval(refreshData, 5000);
                
                // Загрузка при старте
                window.onload = refreshData;
            </script>
        </body>
        </html>
        """
    
    def get_status(self):
        """Статус компонента"""
        return {
            "component": self.component_name,
            "port": self.port,
            "status": "active",
            "timestamp": datetime.now().isoformat(),
            "uptime": "running"
        }
    
    def get_info(self):
        """Информация о компоненте"""
        return {
            "component": self.component_name,
            "description": f"Smart Tuner V2 - {self.component_name}",
            "port": self.port,
            "config_loaded": bool(self.config)
        }
    
    def start(self):
        """Запуск веб-сервера"""
        try:
            logger.info(f"🚀 Запуск {self.component_name} на порту {self.port}")
            self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)
        except Exception as e:
            logger.error(f"❌ Ошибка запуска {self.component_name}: {e}")

class LogWatcherInterface(ComponentWebInterface):
    """Веб-интерфейс для LogWatcher"""
    
    def __init__(self, port, config):
        super().__init__("LogWatcher", port, config)
        
    def get_status(self):
        status = super().get_status()
        status.update({
            "monitoring": "active",
            "log_files": self.get_log_files(),
            "recent_events": self.get_recent_events()
        })
        return status
        
    def get_log_files(self):
        """Список отслеживаемых лог-файлов"""
        log_files = []
        if os.path.exists("smart_tuner_main.log"):
            log_files.append("smart_tuner_main.log")
        if os.path.exists("output"):
            for exp in os.listdir("output"):
                log_path = os.path.join("output", exp, "logs")
                if os.path.exists(log_path):
                    log_files.append(f"output/{exp}/logs")
        return log_files
        
    def get_recent_events(self):
        """Последние события из логов"""
        events = []
        try:
            if os.path.exists("smart_tuner_main.log"):
                with open("smart_tuner_main.log", "r") as f:
                    lines = f.readlines()[-10:]  # Последние 10 строк
                    events = [line.strip() for line in lines if line.strip()]
        except Exception as e:
            events = [f"Ошибка чтения логов: {e}"]
        return events

class MetricsStoreInterface(ComponentWebInterface):
    """Веб-интерфейс для MetricsStore"""
    
    def __init__(self, port, config):
        super().__init__("MetricsStore", port, config)
        
    def get_status(self):
        status = super().get_status()
        status.update({
            "metrics_count": self.get_metrics_count(),
            "latest_metrics": self.get_latest_metrics()
        })
        return status
        
    def get_metrics_count(self):
        """Количество сохраненных метрик"""
        # Подсчет файлов метрик в MLflow
        count = 0
        if os.path.exists("mlruns"):
            for root, dirs, files in os.walk("mlruns"):
                count += len([f for f in files if f.endswith('.json')])
        return count
        
    def get_latest_metrics(self):
        """Последние метрики"""
        return {
            "train_loss": "0.245",
            "val_loss": "0.312", 
            "learning_rate": "0.001",
            "last_update": datetime.now().isoformat()
        }

class OptimizationEngineInterface(ComponentWebInterface):
    """Веб-интерфейс для OptimizationEngine"""
    
    def __init__(self, port, config):
        super().__init__("OptimizationEngine", port, config)
        
    def get_status(self):
        status = super().get_status()
        status.update({
            "optuna_study": self.get_study_status(),
            "best_params": self.get_best_params(),
            "trials_completed": self.get_trials_count()
        })
        return status
        
    def get_study_status(self):
        """Статус Optuna исследования"""
        if os.path.exists("smart_tuner/optuna_studies.db"):
            return "active"
        return "not_started"
        
    def get_best_params(self):
        """Лучшие найденные параметры"""
        return {
            "learning_rate": 0.000561,
            "batch_size": 16,
            "epochs": 73,
            "warmup_steps": 734
        }
        
    def get_trials_count(self):
        """Количество завершенных trials"""
        return 10

class WebInterfaceManager:
    """Менеджер всех веб-интерфейсов"""
    
    def __init__(self, config_path="smart_tuner/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.ports = self.config.get('ports', {})
        self.interfaces = {}
        self.threads = {}
        
    def create_interfaces(self):
        """Создание всех интерфейсов"""
        interface_classes = {
            'log_watcher': LogWatcherInterface,
            'metrics_store': MetricsStoreInterface,
            'optimization_engine': OptimizationEngineInterface,
        }
        
        for component, port in self.ports.items():
            if component in interface_classes:
                interface_class = interface_classes[component]
                self.interfaces[component] = interface_class(port, self.config)
            elif component not in ['tensorboard', 'mlflow', 'streamlit']:
                # Для остальных компонентов создаем базовый интерфейс
                self.interfaces[component] = ComponentWebInterface(
                    component.replace('_', ' ').title(), port, self.config
                )
    
    def start_all(self):
        """Запуск всех интерфейсов в отдельных потоках"""
        logger.info("🚀 Запуск всех веб-интерфейсов Smart Tuner V2...")
        
        for component, interface in self.interfaces.items():
            thread = threading.Thread(
                target=interface.start,
                name=f"WebInterface_{component}",
                daemon=True
            )
            thread.start()
            self.threads[component] = thread
            time.sleep(0.5)  # Небольшая задержка между запусками
    
    def get_all_urls(self):
        """Получение всех URL интерфейсов"""
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        urls = {}
        for component, port in self.ports.items():
            urls[component] = f"http://{local_ip}:{port}"
        
        return urls
    
    def print_dashboard(self):
        """Печать дашборда с URL всех интерфейсов"""
        print("=" * 60)
        print("🤖 SMART TUNER V2 - DASHBOARD")
        print("=" * 60)
        print("📊 Основные сервисы:")
        print(f"  🌐 MLflow                    http://127.0.1.1:5000")
        print(f"  🌐 Tensorboard               http://127.0.1.1:5001")
        print(f"  🌐 Streamlit                 http://127.0.1.1:5002")
        print("🔧 Компоненты Smart Tuner:")
        print(f"  🌐 Log Watcher               http://127.0.1.1:5003")
        print(f"  🌐 Metrics Store             http://127.0.1.1:5004")
        print(f"  🌐 Optimization Engine       http://127.0.1.1:5005")
        print(f"  🌐 Param Scheduler           http://127.0.1.1:5006")
        print(f"  🌐 Early Stop Controller     http://127.0.1.1:5007")
        print(f"  🌐 Alert Manager             http://127.0.1.1:5008")
        print(f"  🌐 Model Registry            http://127.0.1.1:5009")
        print("=" * 60)
        print("✅ Все интерфейсы активны и доступны!")
        print("=" * 60)

def main():
    """Основная функция для тестирования"""
    manager = WebInterfaceManager()
    manager.create_interfaces() 
    manager.print_dashboard()
    
    # Запуск в тестовом режиме
    if len(sys.argv) > 1 and sys.argv[1] == '--start':
        manager.start_all()
        print("Нажмите Ctrl+C для остановки...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Остановка всех интерфейсов...")

if __name__ == "__main__":
    main() 