import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Load the dataset
df = pd.read_csv('hair_color_purchase_data_kathmandu.csv')

# Convert 'Purchase_Date' to datetime format
df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'])
df['Month'] = df['Purchase_Date'].dt.to_period('M').astype(str)

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Hair Color Purchase Dashboard"

# Refined, presentable style variables
accent_color = '#4a90e2'  # Soft blue accent
background_color = '#f7f9fb'
card_bg = '#ffffff'
text_color = '#222'
header_font = 'Segoe UI, Arial, sans-serif'

refined_style = {
    'backgroundColor': background_color,
    'color': text_color,
    'fontFamily': header_font,
    'padding': '0',
    'margin': '0',
    'minHeight': '100vh',
}

header_style = {
    'textAlign': 'center',
    'color': accent_color,
    'fontWeight': 'bold',
    'fontSize': '2.3rem',
    'marginTop': '2rem',
    'marginBottom': '1.5rem',
    'fontFamily': header_font,
    'letterSpacing': '0.5px',
}

section_style = {
    'backgroundColor': card_bg,
    'borderRadius': '12px',
    'boxShadow': '0 2px 12px rgba(74,144,226,0.07)',
    'padding': '2rem 1.5rem',
    'marginBottom': '2rem',
    'border': f'1px solid {background_color}',
}

insight_list_style = {
    'fontSize': '1.13rem',
    'color': text_color,
    'listStyleType': 'disc',
    'paddingLeft': '1.5rem',
    'fontFamily': header_font,
    'marginBottom': '0',
}

dropdown_style = {
    'width': '100%',
    'fontFamily': header_font,
    'fontSize': '1.05rem',
    'color': text_color,
    'marginBottom': '1.2rem',
    'padding': '0.5rem 0.7rem',
    # No background, border, or boxShadow
}

# App layout
app.layout = html.Div([
    html.H1("Hair Color Product Purchase Analysis", style=header_style),

    html.Label("Select Hair Color Shade:", style={
        'display': 'block',
        'textAlign': 'center',
        'fontWeight': '500',
        'fontSize': '1.15rem',
        'color': accent_color,
        'marginTop': '1.5rem',
        'marginBottom': '0.5rem',
        'fontFamily': header_font
    }),
    dcc.Dropdown(
        id='shade-dropdown',
        options=[{'label': shade, 'value': shade} for shade in df['Shade'].unique()],
        value='Burgundy',
        clearable=False,
        style={
            'maxWidth': '400px',
            'margin': '0 auto 2.5rem auto',
            'display': 'block',
            'textAlign': 'center',
        }
    ),

    html.Div([
        html.H3("Key Insights", style={'textAlign': 'center', 'color': accent_color, 'fontWeight': '600', 'marginBottom': '1.2rem', 'fontFamily': header_font}),
        html.Ul(id='insights-list', style=insight_list_style),
    ], style={**section_style, 'maxWidth': '700px', 'margin': '0 auto 2.5rem auto'}),

    html.Div([
        dcc.Graph(id='monthly-trend', config={'displayModeBar': False}),
        dcc.Graph(id='brand-distribution', config={'displayModeBar': False}),
        dcc.Graph(id='price-age-scatter', config={'displayModeBar': False}),
        dcc.Graph(id='age-distribution', config={'displayModeBar': False}),
        dcc.Graph(id='top-shades', config={'displayModeBar': False}),
        dcc.Graph(id='repeat-purchase', config={'displayModeBar': False}),
        dcc.Graph(id='seasonality', config={'displayModeBar': False}),
    ], style={'backgroundColor': background_color, 'paddingBottom': '2rem'}),
], style=refined_style)

# Helper function for refined plotly template
def refined_fig(fig):
    fig.update_layout(
        template='simple_white',
        font_family=header_font,
        font_color=text_color,
        title_font_size=20,
        title_font_color=accent_color,
        plot_bgcolor=card_bg,
        paper_bgcolor=card_bg,
        margin=dict(l=30, r=30, t=60, b=30),
        legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0, font=dict(size=13)),
        xaxis=dict(showgrid=True, gridcolor='#e6eaf1', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#e6eaf1', zeroline=False),
        colorway=[accent_color, '#222', '#b5b5b5', '#4a90e2', '#d7263d', '#f7b32b', '#2e2e2e', '#888'],
    )
    return fig

# Callback to update monthly trend graph
@app.callback(
    Output('monthly-trend', 'figure'),
    Input('shade-dropdown', 'value')
)
def update_monthly_trend(selected_shade):
    filtered = df[df['Shade'] == selected_shade]
    monthly_counts = filtered.groupby('Month').size().reset_index(name='Count')
    fig = px.bar(monthly_counts, x='Month', y='Count', title=f'Monthly Purchase Trend: {selected_shade}',
                 labels={'Count': 'Number of Purchases'}, color='Count', color_discrete_sequence=[accent_color])
    return refined_fig(fig)

# Callback to update brand distribution pie chart
@app.callback(
    Output('brand-distribution', 'figure'),
    Input('shade-dropdown', 'value')
)
def update_brand_distribution(selected_shade):
    filtered = df[df['Shade'] == selected_shade]
    brand_counts = filtered['Brand'].value_counts().reset_index()
    brand_counts.columns = ['Brand', 'Count']
    fig = px.pie(brand_counts, names='Brand', values='Count', title=f'Brand Distribution for {selected_shade}',
                 hole=0.4, color_discrete_sequence=[accent_color, '#b5b5b5', '#4a90e2', '#222', '#888', '#d7263d', '#f7b32b', '#2e2e2e'])
    fig.update_traces(textinfo='percent+label', pull=[0.03]*len(brand_counts))
    return refined_fig(fig)

# Callback to update price vs age scatter plot
@app.callback(
    Output('price-age-scatter', 'figure'),
    Input('shade-dropdown', 'value')
)
def update_price_age_scatter(selected_shade):
    filtered = df[df['Shade'] == selected_shade]
    fig = px.scatter(filtered, x='Age', y='Price_NPR', color='Brand',
                     title=f'Age vs Price for {selected_shade}',
                     labels={'Price_NPR': 'Price (NPR)'},
                     color_discrete_sequence=[accent_color, '#b5b5b5', '#4a90e2', '#222', '#888', '#d7263d', '#f7b32b', '#2e2e2e'])
    return refined_fig(fig)

# Callback to update key insights
@app.callback(
    Output('insights-list', 'children'),
    Input('shade-dropdown', 'value')
)
def update_insights(selected_shade):
    filtered = df[df['Shade'] == selected_shade]
    total_purchases = len(filtered)
    avg_price = filtered['Price_NPR'].mean()
    top_brand = filtered['Brand'].mode()[0] if not filtered['Brand'].mode().empty else 'N/A'
    repeat_customers = filtered['Customer_ID'].value_counts().gt(1).sum()
    return [
        html.Li(f"Total purchases for {selected_shade}: {total_purchases}"),
        html.Li(f"Average price paid: NPR {avg_price:.2f}"),
        html.Li(f"Most popular brand: {top_brand}"),
        html.Li(f"Number of repeat customers: {repeat_customers}"),
    ]

# Callback to update age distribution
@app.callback(
    Output('age-distribution', 'figure'),
    Input('shade-dropdown', 'value')
)
def update_age_distribution(selected_shade):
    filtered = df[df['Shade'] == selected_shade]
    fig = px.histogram(filtered, x='Age', nbins=13, title=f'Age Distribution for {selected_shade}',
                       labels={'Age': 'Age'}, color_discrete_sequence=[accent_color])
    return refined_fig(fig)

# Callback to update top shades
@app.callback(
    Output('top-shades', 'figure'),
    Input('shade-dropdown', 'value')
)
def update_top_shades(selected_shade):
    shade_counts = df['Shade'].value_counts().reset_index()
    shade_counts.columns = ['Shade', 'Count']
    fig = px.bar(shade_counts, x='Shade', y='Count', title='Top Hair Color Shades', color='Count', color_discrete_sequence=[accent_color])
    return refined_fig(fig)

# Callback to update repeat purchase frequency
@app.callback(
    Output('repeat-purchase', 'figure'),
    Input('shade-dropdown', 'value')
)
def update_repeat_purchase(selected_shade):
    filtered = df[df['Shade'] == selected_shade]
    repeat_counts = filtered['Customer_ID'].value_counts()
    repeat_counts = repeat_counts[repeat_counts > 1]
    fig = px.histogram(repeat_counts, x=repeat_counts.values, nbins=10,
                       title=f'Repeat Purchase Frequency for {selected_shade}',
                       labels={'x': 'Number of Purchases', 'y': 'Number of Customers'}, color_discrete_sequence=[accent_color])
    return refined_fig(fig)

# Callback to update seasonality
@app.callback(
    Output('seasonality', 'figure'),
    Input('shade-dropdown', 'value')
)
def update_seasonality(selected_shade):
    filtered = df[df['Shade'] == selected_shade]
    filtered['MonthNum'] = filtered['Purchase_Date'].dt.month
    month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    filtered['MonthName'] = filtered['MonthNum'].map(month_map)
    month_counts = filtered['MonthName'].value_counts().reindex(list(month_map.values()), fill_value=0)
    fig = px.bar(x=month_counts.index, y=month_counts.values, labels={'x': 'Month', 'y': 'Purchases'},
                 title=f'Seasonality of Purchases for {selected_shade}', color_discrete_sequence=[accent_color])
    return refined_fig(fig)

# Run the Dash app
if __name__ == '__main__':
    app.run(debug=True)
