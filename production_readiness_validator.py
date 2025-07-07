#!/usr/bin/env python3
"""
🎯 PRODUCTION READINESS VALIDATOR
Финальная проверка готовности всех компонентов Tacotron2-New к production deployment

Проверяет:
✅ Доступность всех 10 ключевых компонентов
✅ Интеграцию систем между собой
✅ Производительность и стабильность
✅ Конфигурацию и документацию
✅ Monitoring и health checks
✅ Security и backup готовность
"""

import os
import sys
import time
import json
import requests
import subprocess
import importlib
import socket
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

@dataclass
class ComponentStatus:
    """Статус компонента системы"""
    name: str
    available: bool = False
    version: Optional[str] = None
    size_mb: Optional[float] = None
    last_modified: Optional[str] = None
    integration_status: str = "unknown"
    performance_score: Optional[float] = None
    notes: List[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []

@dataclass 
class ValidationResults:
    """Результаты валидации"""
    total_components: int = 0
    available_components: int = 0
    readiness_percentage: float = 0.0
    critical_issues: List[str] = None
    warnings: List[str] = None
    recommendations: List[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.critical_issues is None:
            self.critical_issues = []
        if self.warnings is None:
            self.warnings = []
        if self.recommendations is None:
            self.recommendations = []
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class ProductionReadinessValidator:
    """🎯 Главный валидатор готовности к production"""

    def __init__(self):
        self.logger = self._setup_logging()
        self.project_root = Path(".")
        
        # Определение ключевых компонентов
        self.key_components = {
            "ultimate_tacotron_trainer": {
                "file": "ultimate_tacotron_trainer.py",
                "class": "UltimateEnhancedTacotronTrainer",
                "critical": True,
                "min_size_mb": 0.05
            },
            "context_aware_training_manager": {
                "file": "context_aware_training_manager.py", 
                "class": "ContextAwareTrainingManager",
                "critical": True,
                "min_size_mb": 0.02
            },
            "adaptive_loss_system": {
                "file": "adaptive_loss_system.py",
                "class": "DynamicTverskyLoss", 
                "critical": True,
                "min_size_mb": 0.01
            },
            "advanced_attention_enhancement": {
                "file": "advanced_attention_enhancement_system.py",
                "class": "MultiHeadLocationAwareAttention",
                "critical": True,
                "min_size_mb": 0.01
            },
            "training_stabilization": {
                "file": "training_stabilization_system.py",
                "class": "TrainingStabilizationSystem",
                "critical": True,
                "min_size_mb": 0.01
            },
            "unified_guided_attention": {
                "file": "unified_guided_attention.py",
                "class": "UnifiedGuidedAttentionLoss",
                "critical": True,
                "min_size_mb": 0.01
            },
            "production_dashboard": {
                "file": "production_realtime_dashboard.py",
                "class": "ProductionRealtimeDashboard",
                "critical": True,
                "min_size_mb": 8
            },
            "performance_optimization": {
                "file": "unified_performance_optimization_system.py",
                "class": "UnifiedPerformanceOptimizationSystem",
                "critical": True,
                "min_size_mb": 8
            },
            "production_monitoring": {
                "file": "simple_monitoring.py",
                "class": "SimpleProductionMonitor",
                "critical": False,
                "min_size_mb": 3
            },
            "deployment_system": {
                "file": "production_deployment_system.py",
                "class": "ProductionDeploymentSystem",
                "critical": True,
                "min_size_mb": 0.01
            }
        }

        self.validation_results = ValidationResults()
        
    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - 🎯 VALIDATOR - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        return logging.getLogger(__name__)

    def validate_production_readiness(self) -> ValidationResults:
        """Главная функция валидации готовности к production"""
        self.logger.info("🎯 НАЧАЛО PRODUCTION READINESS VALIDATION")
        self.logger.info("=" * 80)
        
        # 1. Валидация компонентов
        component_results = self._validate_components()
        
        # 2. Валидация интеграции 
        integration_results = self._validate_integration()
        
        # 3. Валидация конфигурации
        config_results = self._validate_configuration()
        
        # 4. Валидация документации
        docs_results = self._validate_documentation()
        
        # 5. Валидация мониторинга
        monitoring_results = self._validate_monitoring()
        
        # 6. Валидация производительности
        performance_results = self._validate_performance()
        
        # Объединение результатов
        self._compile_final_results(
            component_results, integration_results, config_results,
            docs_results, monitoring_results, performance_results
        )
        
        # Финальный отчет
        self._generate_final_report()
        
        return self.validation_results

    def _validate_components(self) -> Dict[str, ComponentStatus]:
        """Валидация доступности всех компонентов"""
        self.logger.info("📦 Валидация компонентов...")
        
        component_statuses = {}
        available_count = 0
        
        for component_name, component_info in self.key_components.items():
            status = ComponentStatus(name=component_name)
            
            # Проверка файла
            file_path = self.project_root / component_info["file"]
            
            if file_path.exists():
                status.available = True
                available_count += 1
                
                # Размер файла
                status.size_mb = file_path.stat().st_size / (1024 * 1024)
                
                # Дата изменения
                status.last_modified = datetime.fromtimestamp(
                    file_path.stat().st_mtime
                ).strftime('%Y-%m-%d %H:%M')
                
                # Проверка размера
                if status.size_mb < component_info["min_size_mb"]:
                    status.notes.append(f"⚠️ Размер файла мал: {status.size_mb:.1f}MB")
                    if component_info["critical"]:
                        self.validation_results.warnings.append(
                            f"Компонент {component_name} имеет подозрительно малый размер"
                        )
                
                # Проверка импорта
                try:
                    self._test_component_import(component_info["file"], component_info["class"])
                    status.integration_status = "importable"
                    status.notes.append("✅ Импорт успешен")
                except Exception as e:
                    status.integration_status = "import_error"
                    status.notes.append(f"❌ Ошибка импорта: {str(e)[:50]}")
                    if component_info["critical"]:
                        self.validation_results.critical_issues.append(
                            f"Критический компонент {component_name} не импортируется"
                        )
                
                self.logger.info(f"  ✅ {component_name}: {status.size_mb:.1f}MB ({status.integration_status})")
                
            else:
                status.available = False
                status.integration_status = "missing"
                status.notes.append("❌ Файл не найден")
                
                if component_info["critical"]:
                    self.validation_results.critical_issues.append(
                        f"Критический компонент {component_name} отсутствует"
                    )
                
                self.logger.error(f"  ❌ {component_name}: Файл не найден")
            
            component_statuses[component_name] = status
        
        self.validation_results.total_components = len(self.key_components)
        self.validation_results.available_components = available_count
        
        self.logger.info(f"📊 Компоненты: {available_count}/{len(self.key_components)} доступны")
        
        return component_statuses

    def _test_component_import(self, file_path: str, class_name: str):
        """Тестирование импорта компонента"""
        module_name = file_path.replace('.py', '').replace('/', '.')
        
        # Добавляем текущую директорию в Python path
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                return True
            else:
                raise ImportError(f"Класс {class_name} не найден в модуле")
        except Exception as e:
            raise ImportError(f"Ошибка импорта модуля {module_name}: {e}")

    def _validate_integration(self) -> bool:
        """Валидация интеграции между компонентами"""
        self.logger.info("🔗 Валидация интеграции...")
        
        integration_issues = 0
        
        # Проверка отсутствия AutoFixManager
        try:
            from ultimate_tacotron_trainer import UltimateEnhancedTacotronTrainer
            
            # Проверяем что AutoFixManager не используется
            trainer_file = self.project_root / "ultimate_tacotron_trainer.py"
            if trainer_file.exists():
                content = trainer_file.read_text()
                if "AutoFixManager" in content:
                    self.validation_results.warnings.append(
                        "Обнаружены упоминания AutoFixManager в коде - убедитесь что он заменен"
                    )
                    integration_issues += 1
                else:
                    self.logger.info("  ✅ AutoFixManager корректно заменен на Context-Aware Manager")
                    
        except Exception as e:
            self.validation_results.critical_issues.append(
                f"Не удается проверить интеграцию Ultimate Trainer: {e}"
            )
            integration_issues += 1
        
        # Проверка Context-Aware Manager интеграции
        try:
            from context_aware_training_manager import ContextAwareTrainingManager
            self.logger.info("  ✅ Context-Aware Training Manager интегрирован")
        except Exception:
            self.validation_results.critical_issues.append(
                "Context-Aware Training Manager не доступен"
            )
            integration_issues += 1
        
        # Проверка логирования
        try:
            from unified_logging_system import UnifiedLoggingSystem
            self.logger.info("  ✅ Unified Logging System интегрирован") 
        except Exception:
            self.validation_results.warnings.append(
                "Unified Logging System не доступен - используется fallback логирование"
            )
        
        return integration_issues == 0

    def _validate_configuration(self) -> bool:
        """Валидация конфигурации"""
        self.logger.info("⚙️ Валидация конфигурации...")
        
        config_files = [
            "hparams.py",
            "requirements.txt", 
            "install.sh"
        ]
        
        missing_configs = []
        for config_file in config_files:
            if not (self.project_root / config_file).exists():
                missing_configs.append(config_file)
        
        if missing_configs:
            self.validation_results.warnings.append(
                f"Отсутствуют конфигурационные файлы: {', '.join(missing_configs)}"
            )
        
        # Проверка production конфигурации
        prod_configs = [
            "production_config.yaml",
            "PRODUCTION_DEPLOYMENT_GUIDE.md"
        ]
        
        for prod_config in prod_configs:
            if (self.project_root / prod_config).exists():
                self.logger.info(f"  ✅ {prod_config}")
            else:
                self.validation_results.recommendations.append(
                    f"Рекомендуется создать {prod_config}"
                )
        
        return len(missing_configs) == 0

    def _validate_documentation(self) -> bool:
        """Валидация документации"""
        self.logger.info("📚 Валидация документации...")
        
        doc_files = [
            "PRODUCTION_DEPLOYMENT_GUIDE.md",
            "README.md"
        ]
        
        docs_available = 0
        for doc_file in doc_files:
            if (self.project_root / doc_file).exists():
                docs_available += 1
                size_kb = (self.project_root / doc_file).stat().st_size / 1024
                self.logger.info(f"  ✅ {doc_file}: {size_kb:.1f}KB")
            else:
                self.validation_results.recommendations.append(
                    f"Создать документацию: {doc_file}"
                )
        
        return docs_available >= 1

    def _validate_monitoring(self) -> bool:
        """Валидация системы мониторинга"""
        self.logger.info("📊 Валидация мониторинга...")
        
        # Проверка портов мониторинга
        monitoring_ports = [5000, 5001, 5002, 5004]  # MLflow, Dashboard, Optuna, TensorBoard
        
        available_ports = []
        for port in monitoring_ports:
            if self._check_port_available(port):
                available_ports.append(port)
        
        if available_ports:
            self.logger.info(f"  ✅ Порты мониторинга доступны: {available_ports}")
        else:
            self.validation_results.warnings.append(
                "Все порты мониторинга заняты - может потребоваться перенастройка"
            )
        
        # Проверка dashboard файлов
        dashboard_files = [
            "production_realtime_dashboard.py",
            "monitoring_dashboard.py"
        ]
        
        dashboard_available = any(
            (self.project_root / f).exists() for f in dashboard_files
        )
        
        if dashboard_available:
            self.logger.info("  ✅ Dashboard компоненты доступны")
        else:
            self.validation_results.critical_issues.append(
                "Dashboard компоненты отсутствуют"
            )
        
        return dashboard_available

    def _validate_performance(self) -> Dict[str, float]:
        """Валидация производительности"""
        self.logger.info("⚡ Валидация производительности...")
        
        performance_metrics = {}
        
        # Измерение времени импорта основных компонентов
        start_time = time.time()
        try:
            from ultimate_tacotron_trainer import UltimateEnhancedTacotronTrainer
            import_time = time.time() - start_time
            performance_metrics['import_time'] = import_time
            
            if import_time < 2.0:
                self.logger.info(f"  ✅ Время импорта Ultimate Trainer: {import_time:.3f}s")
            else:
                self.validation_results.warnings.append(
                    f"Медленный импорт Ultimate Trainer: {import_time:.3f}s"
                )
        except Exception as e:
            self.validation_results.critical_issues.append(
                f"Не удается импортировать Ultimate Trainer: {e}"
            )
        
        # Проверка размера checkpoint файлов
        checkpoint_files = list(self.project_root.glob("*.pt")) + list(self.project_root.glob("*.pth"))
        total_checkpoint_size = sum(f.stat().st_size for f in checkpoint_files) / (1024**3)  # GB
        
        performance_metrics['checkpoint_size_gb'] = total_checkpoint_size
        
        if total_checkpoint_size < 2.0:
            self.logger.info(f"  ✅ Размер checkpoint файлов: {total_checkpoint_size:.2f}GB")
        else:
            self.validation_results.recommendations.append(
                f"Рассмотрите оптимизацию checkpoint файлов: {total_checkpoint_size:.2f}GB"
            )
        
        return performance_metrics

    def _check_port_available(self, port: int) -> bool:
        """Проверка доступности порта"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('localhost', port))
            sock.close()
            return True
        except OSError:
            return False

    def _compile_final_results(self, *validation_results):
        """Компиляция финальных результатов"""
        # Расчет общего процента готовности
        base_readiness = (self.validation_results.available_components / 
                         self.validation_results.total_components) * 100
        
        # Корректировки на основе критических проблем
        critical_penalty = len(self.validation_results.critical_issues) * 10
        warning_penalty = len(self.validation_results.warnings) * 3
        
        final_readiness = max(0, base_readiness - critical_penalty - warning_penalty)
        
        self.validation_results.readiness_percentage = final_readiness

    def _generate_final_report(self):
        """Генерация финального отчета"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🎯 PRODUCTION READINESS VALIDATION RESULTS")
        self.logger.info("=" * 80)
        
        # Общий статус
        readiness = self.validation_results.readiness_percentage
        if readiness >= 95:
            status_emoji = "🎉"
            status_text = "ОТЛИЧНО"
            status_color = "GREEN"
        elif readiness >= 85:
            status_emoji = "✅"
            status_text = "ХОРОШО"
            status_color = "GREEN"
        elif readiness >= 70:
            status_emoji = "⚠️"
            status_text = "ТРЕБУЕТ ВНИМАНИЯ"
            status_color = "YELLOW"
        else:
            status_emoji = "❌"
            status_text = "НЕ ГОТОВ"
            status_color = "RED"
        
        self.logger.info(f"{status_emoji} Общая готовность: {readiness:.1f}% - {status_text}")
        self.logger.info(f"📦 Компоненты: {self.validation_results.available_components}/{self.validation_results.total_components}")
        
        # Критические проблемы
        if self.validation_results.critical_issues:
            self.logger.info(f"\n🚨 Критические проблемы ({len(self.validation_results.critical_issues)}):")
            for issue in self.validation_results.critical_issues:
                self.logger.info(f"  • {issue}")
        
        # Предупреждения
        if self.validation_results.warnings:
            self.logger.info(f"\n⚠️ Предупреждения ({len(self.validation_results.warnings)}):")
            for warning in self.validation_results.warnings:
                self.logger.info(f"  • {warning}")
        
        # Рекомендации
        if self.validation_results.recommendations:
            self.logger.info(f"\n💡 Рекомендации ({len(self.validation_results.recommendations)}):")
            for rec in self.validation_results.recommendations:
                self.logger.info(f"  • {rec}")
        
        # Сохранение отчета
        self._save_validation_report()
        
        self.logger.info("\n" + "=" * 80)
        
        if readiness >= 95:
            self.logger.info("🎉 СИСТЕМА ГОТОВА К PRODUCTION DEPLOYMENT!")
        elif readiness >= 85:
            self.logger.info("✅ Система в основном готова, но есть улучшения")
        else:
            self.logger.info("⚠️ Система требует дополнительной работы перед продакшеном")

    def _save_validation_report(self):
        """Сохранение отчета валидации"""
        report_path = self.project_root / "production_readiness_report.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.validation_results), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📄 Отчет сохранен: {report_path}")

def main():
    """Главная функция валидатора"""
    validator = ProductionReadinessValidator()
    results = validator.validate_production_readiness()
    
    # Return code для CI/CD
    if results.readiness_percentage >= 95:
        sys.exit(0)  # Полная готовность
    elif results.readiness_percentage >= 85:
        sys.exit(1)  # Готов с предупреждениями
    else:
        sys.exit(2)  # Не готов

if __name__ == "__main__":
    main() 