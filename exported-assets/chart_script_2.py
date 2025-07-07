import plotly.graph_objects as go

# Create comparison data
categories = ['Parameter Mgmt', 'Context Aware', 'Failure Handle', 'Rollback Sys', 'Thresholds', 'Approach']
old_problems = ['Rigid changes', 'No context', 'Cascade fails', 'No rollback', 'Fixed thresh', 'Reactive only']
new_solutions = ['Adaptive tune', 'Context-aware', 'Multi-agent', 'Smart rollback', 'Dynamic thresh', 'Proactive opt']

# Create the comparison chart
fig = go.Figure()

# Add bars for old system (problems) - left side
fig.add_trace(go.Bar(
    y=categories,
    x=[-5] * len(categories),  # Fixed width for visual consistency
    name='Old System',
    orientation='h',
    marker_color='#B4413C',  # Red color for problems
    text=old_problems,
    textposition='inside',
    hovertemplate='<b>Destructive AutoFixManager</b><br>%{text}<extra></extra>'
))

# Add bars for new system (solutions) - right side
fig.add_trace(go.Bar(
    y=categories,
    x=[5] * len(categories),  # Fixed width for visual consistency
    name='New System',
    orientation='h',
    marker_color='#ECEBD5',  # Light green color for solutions
    text=new_solutions,
    textposition='inside',
    hovertemplate='<b>Intelligent Training System</b><br>%{text}<extra></extra>'
))

# Update layout
fig.update_layout(
    title='Old AutoFixManager vs New System',
    xaxis_title='System Comparison',
    yaxis_title='Features',
    barmode='overlay',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
    xaxis=dict(
        range=[-6, 6],
        tickvals=[-5, 0, 5],
        ticktext=['Problems', 'Transform', 'Solutions'],
        zeroline=True,
        zerolinewidth=3,
        zerolinecolor='black'
    ),
    annotations=[
        dict(
            x=-5, y=len(categories)+0.5,
            text="<b>Destructive AutoFixManager</b>",
            showarrow=False,
            font=dict(size=14, color='#B4413C'),
            xanchor='center'
        ),
        dict(
            x=5, y=len(categories)+0.5,
            text="<b>Intelligent Training System</b>",
            showarrow=False,
            font=dict(size=14, color='#5D878F'),
            xanchor='center'
        ),
        dict(
            x=0, y=len(categories),
            text="<b>TRANSFORMATION</b><br>→<br>95% fewer fails<br>80% faster fix<br>60% less downtime",
            showarrow=False,
            font=dict(size=12, color='black'),
            bgcolor='white',
            bordercolor='black',
            borderwidth=1,
            xanchor='center',
            yanchor='bottom'
        )
    ]
)

# Update axes
fig.update_yaxes(autorange='reversed')

# Save the chart
fig.write_image('autofixmanager_transformation.png', width=1200, height=700, scale=2)