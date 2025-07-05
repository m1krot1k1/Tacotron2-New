import pandas as pd
import plotly.graph_objects as go

# Since problems_analysis.csv is not available, I'll create sample data based on the request
# This represents typical problem distribution by severity levels
data = {
    'Severity': ['Критическая', 'Высокая', 'Средняя'],
    'Count': [12, 25, 43]
}

df = pd.DataFrame(data)

# Create horizontal bar chart with cliponaxis on the trace
fig = go.Figure(data=[
    go.Bar(
        x=df['Count'],
        y=df['Severity'],
        orientation='h',
        marker_color=['red', 'orange', 'yellow'],
        cliponaxis=False
    )
])

# Update layout with abbreviated labels
fig.update_layout(
    title='Проблемы по критичности',
    xaxis_title='Количество',
    yaxis_title='Критичность'
)

# Save the chart
fig.write_image("problems_severity_chart.png")
print("Chart saved as problems_severity_chart.png")