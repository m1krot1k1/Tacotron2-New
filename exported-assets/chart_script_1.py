import plotly.graph_objects as go
import numpy as np

# Define the flowchart structure with coordinates and connections
nodes = {
    'Training Step Input': {'x': 0, 'y': 10, 'type': 'start', 'color': '#1FB8CD', 'shape': 'circle'},
    'Context Analysis': {'x': 0, 'y': 8.5, 'type': 'process', 'color': '#1FB8CD', 'shape': 'square'},
    'Multi-criteria Eval': {'x': 0, 'y': 7, 'type': 'process', 'color': '#1FB8CD', 'shape': 'square'},
    'Attention Align?': {'x': -3, 'y': 5, 'type': 'decision', 'color': '#D2BA4C', 'shape': 'diamond'},
    'Gradients Stable?': {'x': 0, 'y': 5, 'type': 'decision', 'color': '#D2BA4C', 'shape': 'diamond'},
    'Loss Converging?': {'x': 3, 'y': 5, 'type': 'decision', 'color': '#D2BA4C', 'shape': 'diamond'},
    'Phase Transit?': {'x': 1.5, 'y': 3.5, 'type': 'decision', 'color': '#D2BA4C', 'shape': 'diamond'},
    'Inc Learn Rate': {'x': -4, 'y': 3, 'type': 'action', 'color': '#ECEBD5', 'shape': 'square'},
    'Adj Attention': {'x': -2, 'y': 3, 'type': 'action', 'color': '#ECEBD5', 'shape': 'square'},
    'Dec Learn Rate': {'x': 0, 'y': 3, 'type': 'action', 'color': '#ECEBD5', 'shape': 'square'},
    'Modify Batch': {'x': 2, 'y': 3, 'type': 'action', 'color': '#ECEBD5', 'shape': 'square'},
    'Change Optimizer': {'x': 4, 'y': 3, 'type': 'action', 'color': '#ECEBD5', 'shape': 'square'},
    'Curriculum Learn': {'x': 3, 'y': 1.5, 'type': 'action', 'color': '#ECEBD5', 'shape': 'square'},
    'Success Valid': {'x': -2, 'y': 1, 'type': 'feedback', 'color': '#944454', 'shape': 'circle'},
    'Rollback Proc': {'x': 0, 'y': 1, 'type': 'feedback', 'color': '#944454', 'shape': 'circle'},
    'Learn Decision': {'x': 2, 'y': 1, 'type': 'feedback', 'color': '#944454', 'shape': 'circle'}
}

# Define connections
connections = [
    ('Training Step Input', 'Context Analysis'),
    ('Context Analysis', 'Multi-criteria Eval'),
    ('Multi-criteria Eval', 'Attention Align?'),
    ('Multi-criteria Eval', 'Gradients Stable?'),
    ('Multi-criteria Eval', 'Loss Converging?'),
    ('Loss Converging?', 'Phase Transit?'),
    ('Attention Align?', 'Inc Learn Rate'),
    ('Attention Align?', 'Adj Attention'),
    ('Gradients Stable?', 'Dec Learn Rate'),
    ('Gradients Stable?', 'Modify Batch'),
    ('Loss Converging?', 'Change Optimizer'),
    ('Phase Transit?', 'Curriculum Learn'),
    ('Inc Learn Rate', 'Success Valid'),
    ('Adj Attention', 'Success Valid'),
    ('Dec Learn Rate', 'Rollback Proc'),
    ('Modify Batch', 'Rollback Proc'),
    ('Change Optimizer', 'Learn Decision'),
    ('Curriculum Learn', 'Learn Decision')
]

# Create the figure
fig = go.Figure()

# Add connections as lines
for start, end in connections:
    start_node = nodes[start]
    end_node = nodes[end]
    fig.add_trace(go.Scatter(
        x=[start_node['x'], end_node['x']],
        y=[start_node['y'], end_node['y']],
        mode='lines',
        line=dict(color='#5D878F', width=2),
        showlegend=False,
        hoverinfo='skip',
        cliponaxis=False
    ))

# Add nodes with different shapes
for name, node in nodes.items():
    # Create abbreviated labels for display
    display_name = name if len(name) <= 15 else name[:12] + '...'
    
    # Different marker symbols for different shapes
    if node['shape'] == 'diamond':
        symbol = 'diamond'
        size = 40
    elif node['shape'] == 'square':
        symbol = 'square'
        size = 35
    else:  # circle
        symbol = 'circle'
        size = 30
    
    fig.add_trace(go.Scatter(
        x=[node['x']],
        y=[node['y']],
        mode='markers+text',
        marker=dict(
            size=size,
            color=node['color'],
            symbol=symbol,
            line=dict(width=2, color='white')
        ),
        text=display_name,
        textposition='middle center',
        textfont=dict(size=8, color='black'),
        name=node['type'].title(),
        showlegend=True,
        hovertemplate=f'<b>{name}</b><br>Type: {node["type"]}<extra></extra>',
        cliponaxis=False
    ))

# Update layout
fig.update_layout(
    title="Training Param Adjustment Flow",
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
    plot_bgcolor='white'
)

# Remove duplicate legend entries
seen_types = set()
for trace in fig.data:
    if hasattr(trace, 'name') and trace.name in seen_types:
        trace.showlegend = False
    elif hasattr(trace, 'name'):
        seen_types.add(trace.name)

# Save the chart
fig.write_image("training_decision_flow.png")
print("Chart saved as training_decision_flow.png")