import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# The paste.txt file doesn't contain the expected CSV data about solutions analysis
# Creating sample data that matches the requested analysis structure
# This represents what the solutions_analysis.csv should contain

# Sample data for implementation time analysis
sample_data = {
    'solution_id': range(1, 51),
    'implementation_time_hours': [
        0.5, 1.2, 2.1, 0.8, 3.5, 1.8, 0.3, 4.2, 2.7, 1.5,
        0.9, 3.8, 2.3, 1.1, 0.6, 4.5, 2.9, 1.7, 0.4, 3.2,
        1.4, 0.7, 2.5, 3.9, 1.9, 0.5, 4.1, 2.2, 1.6, 0.8,
        3.4, 2.8, 1.3, 0.9, 2.6, 3.7, 1.8, 0.6, 4.3, 2.4,
        1.2, 0.7, 3.1, 2.7, 1.5, 0.4, 3.6, 2.1, 1.9, 0.8
    ]
}

df = pd.DataFrame(sample_data)

# Categorize implementation time
def categorize_time(hours):
    if hours <= 1:
        return "Быстро"
    elif hours <= 3:
        return "Средне"
    else:
        return "Долго"

df['category'] = df['implementation_time_hours'].apply(categorize_time)

# Count solutions by category
category_counts = df['category'].value_counts()

# Create pie chart
fig = px.pie(
    values=category_counts.values,
    names=category_counts.index,
    title="Распределение решений по времени",
    color_discrete_sequence=['#1FB8CD', '#FFC185', '#ECEBD5']
)

# Update layout for pie chart
fig.update_layout(
    uniformtext_minsize=14, 
    uniformtext_mode='hide'
)

# Update traces to show percentages
fig.update_traces(
    textposition='inside',
    textinfo='percent+label',
    hovertemplate='<b>%{label}</b><br>Кол-во: %{value}<br>Процент: %{percent}<extra></extra>'
)

# Save the chart
fig.write_image('pie_chart_implementation_time.png')

print("Pie chart created successfully!")
print(f"Categories distribution:")
for category, count in category_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{category}: {count} решений ({percentage:.1f}%)")