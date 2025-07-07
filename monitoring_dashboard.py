"""
Monitoring Dashboard для Enhanced Tacotron2 AI System

Веб-интерфейс для визуального мониторинга всех компонентов
интеллектуальной системы с real-time обновлениями.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# Используем простую заглушку вместо недоступного production_monitoring
try:
    from simple_monitoring import (
        SimpleProductionMonitor as ProductionMonitor, 
        MonitoringConfig, ComponentStatus, AlertSeverity
    )
    def create_production_monitor(config=None):
        return ProductionMonitor(config)
except ImportError:
    # Fallback заглушки
    class ProductionMonitor:
        def __init__(self, config=None): pass
        def register_component(self, name, component): pass
        def start_monitoring(self): pass
        def stop_monitoring(self): pass
    
    class MonitoringConfig:
        def __init__(self): pass
    
    class ComponentStatus:
        HEALTHY = "healthy"
        WARNING = "warning"
        CRITICAL = "critical"
        OFFLINE = "offline"
    
    class AlertSeverity:
        INFO = "info"
        WARNING = "warning"
        CRITICAL = "critical"
    
    def create_production_monitor(config=None):
        return ProductionMonitor(config)

# Настройка логирования
logger = logging.getLogger(__name__)

class MonitoringDashboard:
    """Dashboard для мониторинга Enhanced Tacotron2 AI System"""
    
    def __init__(self, monitor: ProductionMonitor, config: Optional[MonitoringConfig] = None):
        self.monitor = monitor
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger(f"{__name__}.MonitoringDashboard")
        
        # Создание Dash приложения
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
            suppress_callback_exceptions=True
        )
        
        # Настройка layout
        self.setup_layout()
        self.setup_callbacks()
        
        self.logger.info("Monitoring Dashboard initialized")
    
    def setup_layout(self):
        """Настройка layout dashboard"""
        
        # Цветовая схема
        colors = {
            'healthy': '#28a745',
            'warning': '#ffc107', 
            'critical': '#dc3545',
            'offline': '#6c757d',
            'background': '#f8f9fa',
            'text': '#495057'
        }
        
        # Header
        header = dbc.Navbar(
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H1("Enhanced Tacotron2 AI System", className="mb-0"),
                        html.Small("Production Monitoring Dashboard", className="text-muted")
                    ], width=8),
                    dbc.Col([
                        html.Div(id="last-update", className="text-end"),
                        dbc.Badge(id="monitoring-status", color="success", className="me-2"),
                    ], width=4, className="d-flex align-items-center justify-content-end")
                ], className="w-100")
            ], fluid=True),
            color="light",
            light=True,
            className="mb-4"
        )
        
        # System Overview Cards
        system_overview = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id="total-components", className="card-title text-primary"),
                        html.P("Total Components", className="card-text")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id="healthy-components", className="card-title text-success"),
                        html.P("Healthy", className="card-text")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id="warning-components", className="card-title text-warning"),
                        html.P("Warning", className="card-text")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id="critical-components", className="card-title text-danger"),
                        html.P("Critical", className="card-text")
                    ])
                ])
            ], width=3)
        ], className="mb-4")
        
        # Main Content Tabs
        main_tabs = dbc.Tabs([
            dbc.Tab(label="System Overview", tab_id="overview"),
            dbc.Tab(label="Component Details", tab_id="components"),
            dbc.Tab(label="Performance Metrics", tab_id="performance"),
            dbc.Tab(label="Alerts", tab_id="alerts"),
            dbc.Tab(label="Historical Data", tab_id="history")
        ], id="main-tabs", active_tab="overview", className="mb-4")
        
        # Tab Content
        tab_content = html.Div(id="tab-content")
        
        # Auto-refresh interval
        interval_component = dcc.Interval(
            id='interval-component',
            interval=self.config.dashboard_update_interval * 1000,  # milliseconds
            n_intervals=0
        )
        
        # Main Layout
        self.app.layout = dbc.Container([
            header,
            system_overview,
            main_tabs,
            tab_content,
            interval_component
        ], fluid=True)
    
    def setup_callbacks(self):
        """Настройка callback функций"""
        
        @self.app.callback(
            [Output('total-components', 'children'),
             Output('healthy-components', 'children'),
             Output('warning-components', 'children'),
             Output('critical-components', 'children'),
             Output('last-update', 'children'),
             Output('monitoring-status', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_overview_cards(n):
            """Обновление карточек обзора системы"""
            try:
                overview = self.monitor.get_system_overview()
                
                last_update = "Never"
                if overview.get('last_update'):
                    update_time = datetime.fromisoformat(overview['last_update'])
                    last_update = f"Last update: {update_time.strftime('%H:%M:%S')}"
                
                status = "Active" if overview['monitoring_active'] else "Inactive"
                
                return (
                    str(overview['total_components']),
                    str(overview['healthy_components']),
                    str(overview['warning_components']),
                    str(overview['critical_components']),
                    last_update,
                    status
                )
            except Exception as e:
                logger.error(f"Failed to update overview cards: {e}")
                return "0", "0", "0", "0", "Error", "Error"
        
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'active_tab'),
             Input('interval-component', 'n_intervals')]
        )
        def update_tab_content(active_tab, n):
            """Обновление содержимого вкладок"""
            if active_tab == "overview":
                return self.create_overview_tab()
            elif active_tab == "components":
                return self.create_components_tab()
            elif active_tab == "performance":
                return self.create_performance_tab()
            elif active_tab == "alerts":
                return self.create_alerts_tab()
            elif active_tab == "history":
                return self.create_history_tab()
            else:
                return html.Div("Tab not found")
    
    def create_overview_tab(self):
        """Создание вкладки обзора системы"""
        try:
            overview = self.monitor.get_system_overview()
            
            # System Status Chart
            status_data = {
                'Status': ['Healthy', 'Warning', 'Critical', 'Offline'],
                'Count': [
                    overview['healthy_components'],
                    overview['warning_components'],
                    overview['critical_components'],
                    overview['offline_components']
                ],
                'Color': ['#28a745', '#ffc107', '#dc3545', '#6c757d']
            }
            
            status_fig = px.pie(
                values=status_data['Count'],
                names=status_data['Status'],
                color_discrete_sequence=status_data['Color'],
                title="System Component Status"
            )
            status_fig.update_layout(height=400)
            
            # Component Status Table
            components_data = []
            for name, data in overview.get('components', {}).items():
                components_data.append({
                    'Component': name.replace('_', ' ').title(),
                    'Status': data['status'].title(),
                    'CPU %': f"{data['cpu_usage']:.1f}",
                    'Memory %': f"{data['memory_usage']:.1f}",
                    'GPU %': f"{data['gpu_usage']:.1f}" if data['gpu_usage'] else "N/A",
                    'Errors': data['error_count'],
                    'Uptime (h)': f"{data['uptime_hours']:.1f}"
                })
            
            components_table = dash_table.DataTable(
                data=components_data,
                columns=[{"name": col, "id": col} for col in ['Component', 'Status', 'CPU %', 'Memory %', 'GPU %', 'Errors', 'Uptime (h)']],
                style_cell={'textAlign': 'left'},
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{Status} = Healthy'},
                        'backgroundColor': '#d4edda',
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '{Status} = Warning'},
                        'backgroundColor': '#fff3cd',
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '{Status} = Critical'},
                        'backgroundColor': '#f8d7da',
                        'color': 'black',
                    }
                ],
                style_header={'backgroundColor': '#e9ecef', 'fontWeight': 'bold'},
                page_size=10
            )
            
            return dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("System Status Distribution"),
                        dbc.CardBody([
                            dcc.Graph(figure=status_fig)
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Resource Usage"),
                        dbc.CardBody([
                            self.create_resource_usage_chart()
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Component Status Details"),
                        dbc.CardBody([
                            components_table
                        ])
                    ])
                ], width=12, className="mt-4")
            ])
            
        except Exception as e:
            logger.error(f"Failed to create overview tab: {e}")
            return html.Div(f"Error creating overview: {str(e)}")
    
    def create_resource_usage_chart(self):
        """Создание графика использования ресурсов"""
        try:
            overview = self.monitor.get_system_overview()
            components = overview.get('components', {})
            
            if not components:
                return html.Div("No component data available")
            
            # Подготовка данных для графика
            component_names = []
            cpu_usage = []
            memory_usage = []
            gpu_usage = []
            
            for name, data in components.items():
                component_names.append(name.replace('_', ' ').title())
                cpu_usage.append(data['cpu_usage'])
                memory_usage.append(data['memory_usage'])
                gpu_usage.append(data['gpu_usage'] if data['gpu_usage'] else 0)
            
            # Создание графика
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='CPU %',
                x=component_names,
                y=cpu_usage,
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                name='Memory %',
                x=component_names,
                y=memory_usage,
                marker_color='lightcoral'
            ))
            
            if any(gpu_usage):
                fig.add_trace(go.Bar(
                    name='GPU %',
                    x=component_names,
                    y=gpu_usage,
                    marker_color='lightgreen'
                ))
            
            fig.update_layout(
                title="Resource Usage by Component",
                xaxis_title="Components",
                yaxis_title="Usage %",
                barmode='group',
                height=350,
                margin=dict(l=50, r=50, t=50, b=100)
            )
            
            # Поворот подписей осей для лучшей читаемости
            fig.update_xaxes(tickangle=45)
            
            return dcc.Graph(figure=fig)
            
        except Exception as e:
            logger.error(f"Failed to create resource usage chart: {e}")
            return html.Div(f"Error creating chart: {str(e)}")
    
    def create_components_tab(self):
        """Создание вкладки деталей компонентов"""
        try:
            overview = self.monitor.get_system_overview()
            components = overview.get('components', {})
            
            component_cards = []
            
            for name, data in components.items():
                # Определение цвета карточки на основе статуса
                color = {
                    'healthy': 'success',
                    'warning': 'warning',
                    'critical': 'danger',
                    'offline': 'secondary'
                }.get(data['status'], 'light')
                
                # Кастомные метрики
                custom_metrics = data.get('custom_metrics', {})
                custom_metrics_list = []
                for key, value in custom_metrics.items():
                    if isinstance(value, (int, float)):
                        custom_metrics_list.append(html.Li(f"{key}: {value:.2f}"))
                    else:
                        custom_metrics_list.append(html.Li(f"{key}: {value}"))
                
                card = dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5(name.replace('_', ' ').title(), className="mb-0"),
                            dbc.Badge(data['status'].title(), color=color, className="ms-2")
                        ]),
                        dbc.CardBody([
                            html.P([
                                html.Strong("CPU Usage: "), f"{data['cpu_usage']:.1f}%"
                            ]),
                            html.P([
                                html.Strong("Memory Usage: "), f"{data['memory_usage']:.1f}%"
                            ]),
                            html.P([
                                html.Strong("GPU Usage: "), 
                                f"{data['gpu_usage']:.1f}%" if data['gpu_usage'] else "N/A"
                            ]),
                            html.P([
                                html.Strong("Errors: "), str(data['error_count'])
                            ]),
                            html.P([
                                html.Strong("Uptime: "), f"{data['uptime_hours']:.1f}h"
                            ]),
                            html.Hr(),
                            html.H6("Custom Metrics:"),
                            html.Ul(custom_metrics_list) if custom_metrics_list else html.P("No custom metrics available")
                        ])
                    ], color=color, outline=True)
                ], width=6, className="mb-3")
                
                component_cards.append(card)
            
            return dbc.Row(component_cards)
            
        except Exception as e:
            logger.error(f"Failed to create components tab: {e}")
            return html.Div(f"Error creating components tab: {str(e)}")
    
    def create_performance_tab(self):
        """Создание вкладки метрик производительности"""
        try:
            # Получение данных производительности за последние 24 часа
            # Это пример - в реальной системе данные будут браться из базы данных
            
            # Создание примера данных производительности
            time_range = pd.date_range(
                start=datetime.now() - timedelta(hours=24),
                end=datetime.now(),
                freq='1H'
            )
            
            # Пример данных (в реальной системе это будет из базы данных)
            performance_data = {
                'timestamp': time_range,
                'training_loss': np.random.uniform(0.5, 2.0, len(time_range)),
                'validation_loss': np.random.uniform(0.6, 2.2, len(time_range)),
                'attention_score': np.random.uniform(0.7, 0.95, len(time_range)),
                'model_quality': np.random.uniform(0.65, 0.9, len(time_range)),
                'throughput': np.random.uniform(50, 200, len(time_range)),
                'memory_efficiency': np.random.uniform(0.6, 0.95, len(time_range))
            }
            
            df = pd.DataFrame(performance_data)
            
            # Создание подграфиков
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    'Training & Validation Loss', 'Attention Score',
                    'Model Quality', 'Throughput (samples/sec)',
                    'Memory Efficiency', 'System Health'
                ],
                vertical_spacing=0.1
            )
            
            # Training & Validation Loss
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['training_loss'], 
                          name='Training Loss', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['validation_loss'], 
                          name='Validation Loss', line=dict(color='red')),
                row=1, col=1
            )
            
            # Attention Score
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['attention_score'], 
                          name='Attention Score', line=dict(color='green')),
                row=1, col=2
            )
            
            # Model Quality
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['model_quality'], 
                          name='Model Quality', line=dict(color='purple')),
                row=2, col=1
            )
            
            # Throughput
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['throughput'], 
                          name='Throughput', line=dict(color='orange')),
                row=2, col=2
            )
            
            # Memory Efficiency
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['memory_efficiency'], 
                          name='Memory Efficiency', line=dict(color='brown')),
                row=3, col=1
            )
            
            # System Health (пример агрегированной метрики)
            system_health = (df['attention_score'] + df['model_quality'] + df['memory_efficiency']) / 3
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=system_health, 
                          name='System Health', line=dict(color='teal')),
                row=3, col=2
            )
            
            fig.update_layout(
                height=800,
                title_text="Performance Metrics (Last 24 Hours)",
                showlegend=False
            )
            
            # Performance Summary Cards
            summary_cards = dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{df['training_loss'].iloc[-1]:.3f}", className="text-primary"),
                            html.P("Current Training Loss")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{df['attention_score'].iloc[-1]:.3f}", className="text-success"),
                            html.P("Attention Score")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{df['model_quality'].iloc[-1]:.3f}", className="text-info"),
                            html.P("Model Quality")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{df['throughput'].iloc[-1]:.0f}", className="text-warning"),
                            html.P("Throughput (samples/sec)")
                        ])
                    ])
                ], width=3)
            ], className="mb-4")
            
            return html.Div([
                summary_cards,
                dcc.Graph(figure=fig)
            ])
            
        except Exception as e:
            logger.error(f"Failed to create performance tab: {e}")
            return html.Div(f"Error creating performance tab: {str(e)}")
    
    def create_alerts_tab(self):
        """Создание вкладки алертов"""
        try:
            alerts = self.monitor.alert_manager.get_active_alerts()
            
            if not alerts:
                return dbc.Alert(
                    "No active alerts. All systems are running normally.",
                    color="success",
                    className="text-center"
                )
            
            alert_cards = []
            
            for alert in alerts:
                # Определение цвета на основе серьезности
                color = {
                    'info': 'info',
                    'warning': 'warning',
                    'critical': 'danger',
                    'emergency': 'dark'
                }.get(alert.severity.value, 'light')
                
                # Иконка на основе серьезности
                icon = {
                    'info': 'fa-info-circle',
                    'warning': 'fa-exclamation-triangle',
                    'critical': 'fa-exclamation-circle',
                    'emergency': 'fa-skull-crossbones'
                }.get(alert.severity.value, 'fa-question')
                
                alert_time = datetime.fromisoformat(alert.timestamp)
                time_ago = datetime.now() - alert_time
                
                card = dbc.Card([
                    dbc.CardHeader([
                        html.I(className=f"fas {icon} me-2"),
                        html.Strong(alert.severity.value.upper()),
                        html.Span(f" - {alert.component}", className="ms-2"),
                        html.Small(f"({time_ago.total_seconds()//60:.0f}m ago)", className="text-muted ms-auto")
                    ]),
                    dbc.CardBody([
                        html.P(alert.message, className="card-text"),
                        html.Hr(),
                        html.H6("Details:"),
                        html.Ul([
                            html.Li(f"{key}: {value}") 
                            for key, value in alert.details.items()
                        ]) if alert.details else html.P("No additional details")
                    ])
                ], color=color, outline=True, className="mb-3")
                
                alert_cards.append(card)
            
            return html.Div(alert_cards)
            
        except Exception as e:
            logger.error(f"Failed to create alerts tab: {e}")
            return html.Div(f"Error creating alerts tab: {str(e)}")
    
    def create_history_tab(self):
        """Создание вкладки исторических данных"""
        try:
            # Селектор компонента
            component_selector = dcc.Dropdown(
                id='component-selector',
                options=[
                    {'label': name.replace('_', ' ').title(), 'value': name}
                    for name in self.config.monitored_components
                ],
                value=self.config.monitored_components[0] if self.config.monitored_components else None,
                placeholder="Select a component"
            )
            
            # Селектор временного диапазона
            time_range_selector = dcc.Dropdown(
                id='time-range-selector',
                options=[
                    {'label': 'Last Hour', 'value': 1},
                    {'label': 'Last 6 Hours', 'value': 6},
                    {'label': 'Last 24 Hours', 'value': 24},
                    {'label': 'Last 7 Days', 'value': 168}
                ],
                value=24,
                placeholder="Select time range"
            )
            
            return html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Label("Component:"),
                        component_selector
                    ], width=6),
                    dbc.Col([
                        html.Label("Time Range:"),
                        time_range_selector
                    ], width=6)
                ], className="mb-4"),
                html.Div(id="history-content")
            ])
            
        except Exception as e:
            logger.error(f"Failed to create history tab: {e}")
            return html.Div(f"Error creating history tab: {str(e)}")
    
    def run_server(self, debug: bool = False, host: str = None, port: int = None):
        """Запуск dashboard сервера"""
        
        host = host or self.config.dashboard_host
        port = port or self.config.dashboard_port
        
        self.logger.info(f"Starting dashboard server on {host}:{port}")
        
        self.app.run_server(
            debug=debug,
            host=host,
            port=port,
            dev_tools_ui=debug,
            dev_tools_props_check=debug
        )

# Удобные функции для создания dashboard
def create_monitoring_dashboard(monitor: ProductionMonitor, 
                              config: Optional[MonitoringConfig] = None) -> MonitoringDashboard:
    """Создание dashboard для мониторинга"""
    return MonitoringDashboard(monitor, config)

def run_dashboard_with_monitoring(components: Dict[str, Any] = None, 
                                config: Optional[MonitoringConfig] = None,
                                debug: bool = False):
    """Запуск полной системы мониторинга с dashboard"""
    
    # Создание монитора
    monitor = create_production_monitor(config)
    
    # Регистрация компонентов
    if components:
        for name, component in components.items():
            monitor.register_component(name, component)
    
    # Запуск мониторинга
    monitor.start_monitoring()
    
    # Создание и запуск dashboard
    dashboard = create_monitoring_dashboard(monitor, config)
    
    try:
        dashboard.run_server(debug=debug)
    finally:
        monitor.stop_monitoring()

# Alias для совместимости с production_realtime_dashboard
class ProductionRealtimeDashboard(MonitoringDashboard):
    """Alias для совместимости"""
    def __init__(self, host='0.0.0.0', port=5001):
        # Создаем mock monitor
        monitor = create_production_monitor()
        super().__init__(monitor)
        self.host = host
        self.port = port

if __name__ == "__main__":
    # Демонстрация использования
    import numpy as np
    
    logging.basicConfig(level=logging.INFO)
    
    # Создание mock компонентов
    class MockComponent:
        def __init__(self, name, base_performance=0.8):
            self.name = name
            self.base_performance = base_performance
            self.healthy = True
            self.error_count = 0
        
        def get_monitoring_metrics(self):
            return {
                'requests_per_second': np.random.uniform(10, 100),
                'average_response_time': np.random.uniform(0.1, 1.0),
                'queue_size': np.random.randint(0, 50),
                'success_rate': np.random.uniform(0.95, 1.0)
            }
        
        def is_healthy(self):
            return self.healthy
        
        def get_performance_metrics(self):
            return {
                'training_loss': np.random.uniform(0.5, 2.0),
                'model_quality': self.base_performance + np.random.uniform(-0.1, 0.1),
                'throughput': np.random.uniform(50, 200)
            }
    
    # Создание mock компонентов
    components = {
        'training_stabilization': MockComponent('training_stabilization', 0.85),
        'attention_enhancement': MockComponent('attention_enhancement', 0.9),
        'checkpointing_system': MockComponent('checkpointing_system', 0.95),
        'meta_learning_engine': MockComponent('meta_learning_engine', 0.8),
        'feedback_loop_manager': MockComponent('feedback_loop_manager', 0.88),
        'risk_assessment_module': MockComponent('risk_assessment_module', 0.92),
        'rollback_controller': MockComponent('rollback_controller', 0.87)
    }
    
    # Конфигурация
    config = MonitoringConfig(
        dashboard_host="0.0.0.0",
        dashboard_port=8050,
        metrics_collection_interval=5,  # Быстрее для демонстрации
        dashboard_update_interval=2
    )
    
    print("Starting Enhanced Tacotron2 AI System Monitoring Dashboard...")
    print("Dashboard will be available at: http://localhost:8050")
    print("Press Ctrl+C to stop")
    
    # Запуск системы мониторинга с dashboard
    run_dashboard_with_monitoring(
        components=components,
        config=config,
        debug=True
    ) 