import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Data from the training issues
time_labels = ["20:38:41", "20:38:42", "20:38:57", "20:39:14", "20:39:30", "20:41:21", "20:42:28"]
gradient_explosion = [476894.70, 476894.70, 563978.33, 544413.67, 596312.02, 222373.06, 487421.58]
loss_values = [32.5662, 32.5662, 36.1027, 31.6153, 33.5660, 200.6304, 34.3313]
training_steps = [0, 0, 0, 0, 0, 100, 0]
target_gradient_norm = 5.0

# Create the figure
fig = go.Figure()

# Add gradient explosion line with critical red color
fig.add_trace(go.Scatter(
    x=time_labels,
    y=gradient_explosion,
    mode='lines+markers',
    name='Gradient Norm',
    line=dict(color='#B4413C', width=3),
    marker=dict(color='#B4413C', size=8),
    hovertemplate='Time: %{x}<br>Gradient: %{y:.0f}<extra></extra>',
    cliponaxis=False
))

# Add target gradient norm line
fig.add_trace(go.Scatter(
    x=time_labels,
    y=[target_gradient_norm] * len(time_labels),
    mode='lines',
    name='Target Norm',
    line=dict(color='#13343B', width=2, dash='dash'),
    hovertemplate='Target: %{y:.1f}<extra></extra>',
    cliponaxis=False
))

# Highlight restart point with different marker
restart_idx = training_steps.index(100)
fig.add_trace(go.Scatter(
    x=[time_labels[restart_idx]],
    y=[gradient_explosion[restart_idx]],
    mode='markers',
    name='Restart',
    marker=dict(color='#DB4545', size=12, symbol='diamond'),
    hovertemplate='Restart: %{x}<br>Gradient: %{y:.0f}<extra></extra>',
    cliponaxis=False
))

# Update layout with logarithmic scale to show target norm
fig.update_layout(
    title='Critical Training Instability',
    xaxis_title='Time',
    yaxis_title='Gradient Norm',
    yaxis_type='log',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
    showlegend=True
)

# Save the chart
fig.write_image('training_instability.png')