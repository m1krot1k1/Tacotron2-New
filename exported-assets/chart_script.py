import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# Create figure
fig = go.Figure()

# Define positions for components (x, y coordinates) - spread out more
components = {
    'Context-Aware Training Manager': (0.5, 0.85),
    'Multi-Agent Optimization System': (0.15, 0.65),
    'Adaptive Loss Controller': (0.85, 0.65),
    'Dynamic Attention Supervisor': (0.15, 0.35),
    'Feedback Loop Analyzer': (0.85, 0.35),
    'Meta-Learning Controller': (0.5, 0.15)
}

# Define component colors using the brand colors
component_colors = {
    'Context-Aware Training Manager': '#1FB8CD',
    'Multi-Agent Optimization System': '#FFC185',
    'Adaptive Loss Controller': '#ECEBD5',
    'Dynamic Attention Supervisor': '#5D878F',
    'Feedback Loop Analyzer': '#D2BA4C',
    'Meta-Learning Controller': '#B4413C'
}

# Component abbreviations and functions
component_info = {
    'Context-Aware Training Manager': ('Context Mgr', 'Central Coord'),
    'Multi-Agent Optimization System': ('Multi-Agent', 'Optim Agents'),
    'Adaptive Loss Controller': ('Loss Ctrl', 'Dynamic Loss'),
    'Dynamic Attention Supervisor': ('Attn Supvr', 'Attn Control'),
    'Feedback Loop Analyzer': ('Feedback', 'Perf Analysis'),
    'Meta-Learning Controller': ('Meta-Learn', 'Learning Ctrl')
}

# Add components as scatter points with larger labels and function descriptions
for comp_name, (x, y) in components.items():
    abbrev_name, function = component_info[comp_name]
    
    # Main component node
    fig.add_trace(go.Scatter(
        x=[x], y=[y],
        mode='markers+text',
        marker=dict(
            size=120,
            color=component_colors[comp_name],
            line=dict(color='white', width=3)
        ),
        text=abbrev_name,
        textposition='middle center',
        textfont=dict(size=14, color='black', family='Arial Black'),
        showlegend=False,
        hovertemplate=f"<b>{comp_name}</b><br>{function}<extra></extra>"
    ))
    
    # Add function description below each component
    fig.add_trace(go.Scatter(
        x=[x], y=[y-0.08],
        mode='text',
        text=function,
        textposition='middle center',
        textfont=dict(size=11, color='#333333'),
        showlegend=False,
        hoverinfo='skip'
    ))

# Define connections with different arrow types and labels
connections = [
    # Data flow (blue arrows)
    ('Context-Aware Training Manager', 'Multi-Agent Optimization System', 'blue', 'Training Data'),
    ('Context-Aware Training Manager', 'Adaptive Loss Controller', 'blue', 'Context Info'),
    ('Context-Aware Training Manager', 'Dynamic Attention Supervisor', 'blue', 'State Data'),
    
    # Control signals (red arrows)
    ('Multi-Agent Optimization System', 'Adaptive Loss Controller', 'red', 'Optim Signal'),
    ('Adaptive Loss Controller', 'Dynamic Attention Supervisor', 'red', 'Loss Adjust'),
    ('Dynamic Attention Supervisor', 'Meta-Learning Controller', 'red', 'Param Update'),
    
    # Feedback loops (green arrows)
    ('Feedback Loop Analyzer', 'Context-Aware Training Manager', 'green', 'Perf Metrics'),
    ('Meta-Learning Controller', 'Feedback Loop Analyzer', 'green', 'Learn Stats'),
    ('Adaptive Loss Controller', 'Feedback Loop Analyzer', 'green', 'Loss History'),
    
    # Context information (orange arrows)
    ('Multi-Agent Optimization System', 'Context-Aware Training Manager', 'orange', 'Agent Status'),
    ('Dynamic Attention Supervisor', 'Context-Aware Training Manager', 'orange', 'Attn State')
]

# Color mapping for arrow types
arrow_colors = {
    'blue': '#1FB8CD',
    'red': '#B4413C',
    'green': '#5D878F',
    'orange': '#FFC185'
}

# Add arrows between components with labels
for start_comp, end_comp, color, label in connections:
    start_x, start_y = components[start_comp]
    end_x, end_y = components[end_comp]
    
    # Calculate arrow direction and adjust for node size
    dx = end_x - start_x
    dy = end_y - start_y
    length = np.sqrt(dx**2 + dy**2)
    
    # Adjust start and end points to account for node size
    node_radius = 0.06
    start_x_adj = start_x + (dx/length) * node_radius
    start_y_adj = start_y + (dy/length) * node_radius
    end_x_adj = end_x - (dx/length) * node_radius
    end_y_adj = end_y - (dy/length) * node_radius
    
    # Add arrow line
    fig.add_trace(go.Scatter(
        x=[start_x_adj, end_x_adj],
        y=[start_y_adj, end_y_adj],
        mode='lines',
        line=dict(color=arrow_colors[color], width=4),
        showlegend=False,
        hovertemplate=f"<b>{label}</b><extra></extra>"
    ))
    
    # Add arrowhead
    arrow_length = 0.04
    arrow_angle = np.arctan2(end_y_adj - start_y_adj, end_x_adj - start_x_adj)
    
    # Calculate arrowhead points
    arrow_x = end_x_adj - arrow_length * np.cos(arrow_angle)
    arrow_y = end_y_adj - arrow_length * np.sin(arrow_angle)
    
    left_x = arrow_x - 0.025 * np.cos(arrow_angle + np.pi/2)
    left_y = arrow_y - 0.025 * np.sin(arrow_angle + np.pi/2)
    right_x = arrow_x - 0.025 * np.cos(arrow_angle - np.pi/2)
    right_y = arrow_y - 0.025 * np.sin(arrow_angle - np.pi/2)
    
    fig.add_trace(go.Scatter(
        x=[left_x, end_x_adj, right_x],
        y=[left_y, end_y_adj, right_y],
        mode='lines',
        line=dict(color=arrow_colors[color], width=4),
        fill='toself',
        fillcolor=arrow_colors[color],
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add label on arrow (at midpoint)
    mid_x = (start_x_adj + end_x_adj) / 2
    mid_y = (start_y_adj + end_y_adj) / 2
    
    # Offset label slightly to avoid overlap with arrow
    offset_x = -0.03 * np.sin(arrow_angle)
    offset_y = 0.03 * np.cos(arrow_angle)
    
    fig.add_trace(go.Scatter(
        x=[mid_x + offset_x],
        y=[mid_y + offset_y],
        mode='text',
        text=label,
        textposition='middle center',
        textfont=dict(size=9, color=arrow_colors[color], family='Arial'),
        showlegend=False,
        hoverinfo='skip'
    ))

# Add legend for arrow types
legend_items = [
    ('Data Flow', '#1FB8CD'),
    ('Control Sig', '#B4413C'),
    ('Feedback', '#5D878F'),
    ('Context', '#FFC185')
]

for i, (label, color) in enumerate(legend_items):
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color=color, width=5),
        name=label,
        showlegend=True
    ))

# Update layout
fig.update_layout(
    title="Tacotron2 Training System Architecture",
    xaxis=dict(
        range=[0, 1],
        showgrid=False,
        showticklabels=False,
        zeroline=False
    ),
    yaxis=dict(
        range=[0, 1],
        showgrid=False,
        showticklabels=False,
        zeroline=False
    ),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.05,
        xanchor='center',
        x=0.5,
        font=dict(size=12)
    ),
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# Save the chart
fig.write_image("tacotron2_architecture.png", width=1400, height=900, scale=2)