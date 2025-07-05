import plotly.graph_objects as go
import plotly.io as pio

# Data from the JSON
status_counts = {"Integrated": 8, "Partially Integrated": 1, "Needs Work": 3, "Not Integrated": 1}

# Use brand colors with green for integrated (using the closest available green)
color_map = {
    "Integrated": "#5D878F",  # Cyan (closest to green available)
    "Partially Integrated": "#D2BA4C",  # Moderate yellow
    "Needs Work": "#FFC185",  # Light orange
    "Not Integrated": "#B4413C"  # Moderate red
}

# Create stacked bar chart
fig = go.Figure()

# Add each status as a separate trace for stacking
statuses = ["Integrated", "Partially Integrated", "Needs Work", "Not Integrated"]
counts = [status_counts[status] for status in statuses]
colors = [color_map[status] for status in statuses]

for i, status in enumerate(statuses):
    fig.add_trace(go.Bar(
        y=["Components"],
        x=[counts[i]],
        orientation='h',
        name=status,
        marker_color=colors[i],
        text=counts[i],
        textposition='inside',
        textfont=dict(size=14, color='white'),
        hovertemplate=f'{status}: {counts[i]} components<extra></extra>',
        cliponaxis=False
    ))

# Update layout for stacked bar
fig.update_layout(
    title="Smart Tuner v2 Integration Status",
    xaxis_title="Comp Count",
    yaxis_title="",
    barmode='stack',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
    showlegend=True
)

# Update axes
fig.update_xaxes(range=[0, sum(counts) + 1])
fig.update_yaxes()

# Save the chart
fig.write_image("smart_tuner_status_dashboard.png")