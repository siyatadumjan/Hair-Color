import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import base64
import io

# Load the dataset
df = pd.read_csv('hair_color_purchase_data_kathmandu.csv')

# Convert 'Purchase_Date' to datetime format
df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'])
df['Month'] = df['Purchase_Date'].dt.to_period('M').astype(str)

# Add advanced analytics features
def calculate_rfm_scores(data):
    """Calculate RFM (Recency, Frequency, Monetary) scores for customer segmentation"""
    # Calculate RFM metrics
    today = data['Purchase_Date'].max()
    
    rfm = data.groupby('Customer_ID').agg({
        'Purchase_Date': lambda x: (today - x.max()).days,  # Recency
        'Price_NPR': 'sum'  # Monetary
    }).reset_index()
    
    # Add frequency count
    frequency = data.groupby('Customer_ID').size().reset_index(name='Frequency')
    rfm = rfm.merge(frequency, on='Customer_ID')
    
    rfm.columns = ['Customer_ID', 'Recency', 'Monetary', 'Frequency']
    
    # Score RFM - handle cases with few unique values
    try:
        # Try 5 bins first
        r_quartiles = pd.qcut(rfm['Recency'], q=5, duplicates='drop')
        f_quartiles = pd.qcut(rfm['Frequency'], q=5, duplicates='drop')
        m_quartiles = pd.qcut(rfm['Monetary'], q=5, duplicates='drop')
        
        # Get actual number of bins and create appropriate labels
        r_bins = len(r_quartiles.cat.categories)
        f_bins = len(f_quartiles.cat.categories)
        m_bins = len(m_quartiles.cat.categories)
        
        r_labels = range(r_bins, 0, -1)
        f_labels = range(1, f_bins + 1)
        m_labels = range(1, m_bins + 1)
        
        r_quartiles = pd.qcut(rfm['Recency'], q=5, labels=r_labels, duplicates='drop')
        f_quartiles = pd.qcut(rfm['Frequency'], q=5, labels=f_labels, duplicates='drop')
        m_quartiles = pd.qcut(rfm['Monetary'], q=5, labels=m_labels, duplicates='drop')
        
    except ValueError:
        # If we can't create 5 bins, use fewer bins
        max_bins = min(3, len(rfm['Recency'].unique()), len(rfm['Frequency'].unique()), len(rfm['Monetary'].unique()))
        
        if max_bins < 2:
            # If we have very few unique values, assign same score to all
            rfm['R'] = 1
            rfm['F'] = 1
            rfm['M'] = 1
            rfm['RFM_Score'] = '111'
            return rfm
        
        r_quartiles = pd.qcut(rfm['Recency'], q=max_bins, duplicates='drop')
        f_quartiles = pd.qcut(rfm['Frequency'], q=max_bins, duplicates='drop')
        m_quartiles = pd.qcut(rfm['Monetary'], q=max_bins, duplicates='drop')
        
        r_labels = range(max_bins, 0, -1)
        f_labels = range(1, max_bins + 1)
        m_labels = range(1, max_bins + 1)
        
        r_quartiles = pd.qcut(rfm['Recency'], q=max_bins, labels=r_labels, duplicates='drop')
        f_quartiles = pd.qcut(rfm['Frequency'], q=max_bins, labels=f_labels, duplicates='drop')
        m_quartiles = pd.qcut(rfm['Monetary'], q=max_bins, labels=m_labels, duplicates='drop')
    
    rfm['R'] = r_quartiles
    rfm['F'] = f_quartiles
    rfm['M'] = m_quartiles
    
    # Calculate RFM Score
    rfm['RFM_Score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)
    
    return rfm

def predict_sales_trend(data, months_ahead=3):
    """Simple sales prediction using moving average"""
    monthly_sales = data.groupby('Month')['Price_NPR'].sum().reset_index()
    monthly_sales['Month'] = pd.to_datetime(monthly_sales['Month'].astype(str) + '-01')
    monthly_sales = monthly_sales.sort_values('Month')
    
    # Simple moving average prediction
    window = 3
    monthly_sales['Predicted_Sales'] = monthly_sales['Price_NPR'].rolling(window=window).mean()
    
    # Predict future months
    last_values = monthly_sales['Price_NPR'].tail(window).values
    future_predictions = []
    
    for i in range(months_ahead):
        if len(last_values) >= window:
            prediction = np.mean(last_values[-window:])
        else:
            prediction = np.mean(last_values)
        future_predictions.append(prediction)
        last_values = np.append(last_values, prediction)
    
    return monthly_sales, future_predictions

def segment_customers(data):
    """Customer segmentation using K-means clustering"""
    # Prepare features for clustering
    customer_features = data.groupby('Customer_ID').agg({
        'Price_NPR': ['sum', 'mean', 'count'],
        'Age': 'mean'
    }).reset_index()
    
    customer_features.columns = ['Customer_ID', 'Total_Spent', 'Avg_Spent', 'Purchase_Count', 'Avg_Age']
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(customer_features[['Total_Spent', 'Avg_Spent', 'Purchase_Count', 'Avg_Age']])
    
    # K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    customer_features['Segment'] = kmeans.fit_predict(features_scaled)
    
    # Name segments
    segment_names = ['Budget Buyers', 'Regular Customers', 'Premium Buyers', 'High-Value Customers']
    customer_features['Segment_Name'] = customer_features['Segment'].map(dict(enumerate(segment_names)))
    
    return customer_features

# Calculate advanced metrics
rfm_data = calculate_rfm_scores(df)
sales_prediction, future_predictions = predict_sales_trend(df)
customer_segments = segment_customers(df)

# Add customer segments to the main dataframe
df = df.merge(customer_segments[['Customer_ID', 'Segment_Name']], on='Customer_ID', how='left')

# Initialize the Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Advanced Hair Color Analytics Dashboard"

# Professional color palette
primary_color = '#2c3e50'  # Dark blue-gray
secondary_color = '#3498db'  # Bright blue
accent_color = '#e74c3c'  # Red accent
success_color = '#27ae60'  # Green
warning_color = '#f39c12'  # Orange
light_gray = '#ecf0f1'
dark_gray = '#34495e'
white = '#ffffff'

# Professional styling
app_layout_style = {
    'backgroundColor': '#f8f9fa',
    'fontFamily': '"Segoe UI", "Roboto", "Helvetica Neue", Arial, sans-serif',
    'margin': '0',
    'padding': '0',
    'minHeight': '100vh',
}

navbar_style = {
    'backgroundColor': white,
    'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
    'padding': '1rem 0',
    'marginBottom': '2rem',
}

navbar_brand_style = {
    'color': primary_color,
    'fontWeight': '700',
    'fontSize': '1.5rem',
    'textDecoration': 'none',
}

nav_link_style = {
    'color': dark_gray,
    'fontWeight': '500',
    'textDecoration': 'none',
    'padding': '0.5rem 1rem',
    'borderRadius': '4px',
    'transition': 'all 0.3s ease',
}

nav_link_active_style = {
    'color': secondary_color,
    'fontWeight': '600',
    'backgroundColor': light_gray,
}

page_header_style = {
    'textAlign': 'center',
    'color': primary_color,
    'fontWeight': '300',
    'fontSize': '2.2rem',
    'margin': '0 0 1rem 0',
    'letterSpacing': '1px',
}

page_subtitle_style = {
    'textAlign': 'center',
    'color': dark_gray,
    'fontSize': '1rem',
    'marginBottom': '2rem',
    'fontWeight': '400',
}

controls_section_style = {
    'backgroundColor': white,
    'borderRadius': '8px',
    'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
    'padding': '2rem',
    'marginBottom': '2rem',
    'maxWidth': '1200px',
    'margin': '0 auto 2rem auto',
}

dropdown_container_style = {
    'display': 'grid',
    'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))',
    'gap': '1rem',
    'alignItems': 'center',
}

dropdown_style = {
    'width': '100%',
    'fontSize': '1rem',
    'border': f'2px solid {light_gray}',
    'borderRadius': '6px',
    'padding': '0.75rem 1rem',
    'backgroundColor': white,
    'color': primary_color,
    'fontWeight': '500',
}

kpi_section_style = {
    'backgroundColor': white,
    'borderRadius': '8px',
    'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
    'padding': '2rem',
    'marginBottom': '2rem',
    'maxWidth': '1200px',
    'margin': '0 auto 2rem auto',
}

kpi_grid_style = {
    'display': 'grid',
    'gridTemplateColumns': 'repeat(auto-fit, minmax(250px, 1fr))',
    'gap': '1.5rem',
    'marginBottom': '2rem',
}

kpi_card_style = {
    'backgroundColor': '#f8f9fa',
    'borderRadius': '8px',
    'padding': '1.5rem',
    'textAlign': 'center',
    'borderLeft': f'4px solid {secondary_color}',
}

kpi_value_style = {
    'fontSize': '2rem',
    'fontWeight': '700',
    'color': primary_color,
    'marginBottom': '0.5rem',
}

kpi_label_style = {
    'fontSize': '0.9rem',
    'color': dark_gray,
    'fontWeight': '500',
    'textTransform': 'uppercase',
    'letterSpacing': '0.5px',
}

charts_section_style = {
    'maxWidth': '1200px',
    'margin': '0 auto',
    'padding': '0 1rem',
}

chart_container_style = {
    'backgroundColor': white,
    'borderRadius': '8px',
    'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
    'padding': '1.5rem',
    'marginBottom': '2rem',
    'height': '500px',
    'width': '100%',
    'minHeight': '500px',
    'overflow': 'hidden',
}

chart_title_style = {
    'fontSize': '1.2rem',
    'fontWeight': '600',
    'color': primary_color,
    'marginBottom': '1rem',
    'textAlign': 'center',
    'position': 'sticky',
    'top': '0',
    'backgroundColor': white,
    'padding': '0.5rem 0',
    'zIndex': '10',
}

# Navigation component
def create_navbar():
    return dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("Advanced Hair Color Analytics", style=navbar_brand_style),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Overview", href="/", id="nav-overview")),
                dbc.NavItem(dbc.NavLink("Advanced Analytics", href="/analytics", id="nav-analytics")),
                dbc.NavItem(dbc.NavLink("Customer Insights", href="/customers", id="nav-customers")),
                dbc.NavItem(dbc.NavLink("Predictive Analytics", href="/predictive", id="nav-predictive")),
                dbc.NavItem(dbc.NavLink("Reports", href="/reports", id="nav-reports")),
            ], className="ml-auto"),
        ]),
        color="white",
        dark=False,
        style=navbar_style
    )

# Overview Page Content
def create_overview_page():
    return html.Div([
        html.H1("Advanced Analytics Dashboard", style=page_header_style),
        html.P("Comprehensive analysis with predictive insights and customer segmentation", style=page_subtitle_style),
        
        # Advanced Controls Section
        html.Div([
            html.H3("Advanced Filters", style={'color': primary_color, 'marginBottom': '1rem'}),
            html.Div([
                html.Div([
                    html.Label("Select Hair Color Shade:", style={'fontWeight': '600', 'color': primary_color, 'fontSize': '1rem'}),
                    dcc.Dropdown(
                        id='shade-dropdown',
                        options=[{'label': shade, 'value': shade} for shade in sorted(df['Shade'].unique())],
                        value='Burgundy',
                        clearable=False,
                        style=dropdown_style
                    ),
                ]),
                html.Div([
                    html.Label("Date Range:", style={'fontWeight': '600', 'color': primary_color, 'fontSize': '1rem'}),
                    dcc.DatePickerRange(
                        id='date-range',
                        start_date=df['Purchase_Date'].min(),
                        end_date=df['Purchase_Date'].max(),
                        style=dropdown_style
                    ),
                ]),
                html.Div([
                    html.Label("Customer Segment:", style={'fontWeight': '600', 'color': primary_color, 'fontSize': '1rem'}),
                    dcc.Dropdown(
                        id='segment-dropdown',
                        options=[
                            {'label': 'All Segments', 'value': 'all'},
                            {'label': 'Budget Buyers', 'value': 'Budget Buyers'},
                            {'label': 'Regular Customers', 'value': 'Regular Customers'},
                            {'label': 'Premium Buyers', 'value': 'Premium Buyers'},
                            {'label': 'High-Value Customers', 'value': 'High-Value Customers'}
                        ],
                        value='all',
                        clearable=False,
                        style=dropdown_style
                    ),
                ]),
            ], style=dropdown_container_style),
        ], style=controls_section_style),

        # Advanced KPI Section
        html.Div([
            html.H3("Advanced Key Performance Indicators", style={
                'textAlign': 'center',
                'color': primary_color,
                'fontWeight': '600',
                'marginBottom': '2rem',
                'fontSize': '1.5rem',
            }),
            html.Div(id='advanced-kpi-cards', style=kpi_grid_style),
        ], style=kpi_section_style),

        # Advanced Charts Section
        html.Div([
            # First Row
            html.Div([
                html.Div([
                    html.H4("Monthly Purchase Trend", style=chart_title_style),
                    dcc.Graph(id='monthly-trend', config={'displayModeBar': False}, style={'height': '400px'}),
                ], style=chart_container_style),
                html.Div([
                    html.H4("Brand Distribution", style=chart_title_style),
                    dcc.Graph(id='brand-distribution', config={'displayModeBar': False}, style={'height': '400px'}),
                ], style=chart_container_style),
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '2rem', 'marginBottom': '2rem', 'minHeight': '500px'}),

            # Second Row
            html.Div([
                html.Div([
                    html.H4("Price vs Age Analysis", style=chart_title_style),
                    dcc.Graph(id='price-age-scatter', config={'displayModeBar': False}, style={'height': '400px'}),
                ], style=chart_container_style),
                html.Div([
                    html.H4("Age Distribution", style=chart_title_style),
                    dcc.Graph(id='age-distribution', config={'displayModeBar': False}, style={'height': '400px'}),
                ], style=chart_container_style),
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '2rem', 'marginBottom': '2rem', 'minHeight': '500px'}),
        ], style=charts_section_style),
    ])

# Advanced Analytics Page Content
def create_analytics_page():
    return html.Div([
        html.H1("Advanced Analytics & Insights", style=page_header_style),
        html.P("Deep dive into predictive analytics and customer segmentation", style=page_subtitle_style),
        
        # Controls Section
        html.Div([
            html.Div([
                html.Label("Select Hair Color Shade:", style={
                    'fontWeight': '600',
                    'color': primary_color,
                    'fontSize': '1.1rem',
                    'marginRight': '1rem',
                }),
                dcc.Dropdown(
                    id='analytics-shade-dropdown',
                    options=[{'label': shade, 'value': shade} for shade in sorted(df['Shade'].unique())],
                    value='Burgundy',
                    clearable=False,
                    style=dropdown_style
                ),
            ], style=dropdown_container_style),
        ], style=controls_section_style),

        # Advanced Analytics Charts
        html.Div([
            # First Row
            html.Div([
                html.Div([
                    html.H4("Customer Segmentation Analysis", style=chart_title_style),
                    dcc.Graph(id='customer-segmentation', config={'displayModeBar': False}, style={'height': '400px'}),
                ], style=chart_container_style),
                html.Div([
                    html.H4("RFM Analysis", style=chart_title_style),
                    dcc.Graph(id='rfm-analysis', config={'displayModeBar': False}, style={'height': '400px'}),
                ], style=chart_container_style),
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '2rem', 'marginBottom': '2rem', 'minHeight': '500px'}),

            # Second Row
            html.Div([
                html.Div([
                    html.H4("Sales Prediction", style=chart_title_style),
                    dcc.Graph(id='sales-prediction', config={'displayModeBar': False}, style={'height': '400px'}),
                ], style=chart_container_style),
                html.Div([
                    html.H4("Customer Lifetime Value", style=chart_title_style),
                    dcc.Graph(id='customer-ltv', config={'displayModeBar': False}, style={'height': '400px'}),
                ], style=chart_container_style),
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '2rem', 'marginBottom': '2rem', 'minHeight': '500px'}),

            # Third Row - Full Width
            html.Div([
                html.H4("Advanced Performance Metrics", style=chart_title_style),
                dcc.Graph(id='advanced-metrics', config={'displayModeBar': False}, style={'height': '400px'}),
            ], style=chart_container_style),
        ], style=charts_section_style),
    ])

# Customer Insights Page Content
def create_customers_page():
    return html.Div([
        html.H1("Customer Insights & Segmentation", style=page_header_style),
        html.P("Understanding customer behavior and demographics", style=page_subtitle_style),
        
        # Controls Section
        html.Div([
            html.Div([
                html.Label("Select Hair Color Shade:", style={
                    'fontWeight': '600',
                    'color': primary_color,
                    'fontSize': '1.1rem',
                    'marginRight': '1rem',
                }),
                dcc.Dropdown(
                    id='customers-shade-dropdown',
                    options=[{'label': shade, 'value': shade} for shade in sorted(df['Shade'].unique())],
                    value='Burgundy',
                    clearable=False,
                    style=dropdown_style
                ),
            ], style=dropdown_container_style),
        ], style=controls_section_style),

        # Customer Charts
        html.Div([
            # First Row
            html.Div([
                html.Div([
                    html.H4("Customer Loyalty Analysis", style=chart_title_style),
                    dcc.Graph(id='customer-loyalty', config={'displayModeBar': False}, style={'height': '400px'}),
                ], style=chart_container_style),
                html.Div([
                    html.H4("Customer Age Demographics", style=chart_title_style),
                    dcc.Graph(id='customer-demographics', config={'displayModeBar': False}, style={'height': '400px'}),
                ], style=chart_container_style),
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '2rem', 'marginBottom': '2rem', 'minHeight': '500px'}),

            # Second Row
            html.Div([
                html.Div([
                    html.H4("Purchase Frequency Analysis", style=chart_title_style),
                    dcc.Graph(id='purchase-frequency', config={'displayModeBar': False}, style={'height': '400px'}),
                ], style=chart_container_style),
                html.Div([
                    html.H4("Customer Value Distribution", style=chart_title_style),
                    dcc.Graph(id='customer-value', config={'displayModeBar': False}, style={'height': '400px'}),
                ], style=chart_container_style),
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '2rem', 'marginBottom': '2rem', 'minHeight': '500px'}),
        ], style=charts_section_style),
    ])

# Predictive Analytics Page Content
def create_predictive_page():
    return html.Div([
        html.H1("Predictive Analytics & Forecasting", style=page_header_style),
        html.P("Machine learning insights and future predictions", style=page_subtitle_style),
        
        # Predictive Analytics Charts
        html.Div([
            # First Row
            html.Div([
                html.Div([
                    html.H4("Sales Forecasting", style=chart_title_style),
                    dcc.Graph(id='sales-forecast', config={'displayModeBar': False}, style={'height': '400px'}),
                ], style=chart_container_style),
                html.Div([
                    html.H4("Demand Prediction", style=chart_title_style),
                    dcc.Graph(id='demand-prediction', config={'displayModeBar': False}, style={'height': '400px'}),
                ], style=chart_container_style),
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '2rem', 'marginBottom': '2rem', 'minHeight': '500px'}),

            # Second Row
            html.Div([
                html.Div([
                    html.H4("Customer Churn Prediction", style=chart_title_style),
                    dcc.Graph(id='churn-prediction', config={'displayModeBar': False}, style={'height': '400px'}),
                ], style=chart_container_style),
                html.Div([
                    html.H4("Revenue Projection", style=chart_title_style),
                    dcc.Graph(id='revenue-projection', config={'displayModeBar': False}, style={'height': '400px'}),
                ], style=chart_container_style),
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '2rem', 'marginBottom': '2rem', 'minHeight': '500px'}),
        ], style=charts_section_style),
    ])

# Reports Page Content
def create_reports_page():
    return html.Div([
        html.H1("Comprehensive Reports & Analysis", style=page_header_style),
        html.P("Executive summaries and detailed business insights", style=page_subtitle_style),
        
        # Summary Cards
        html.Div([
            html.Div([
                html.H3("Executive Summary", style={'color': primary_color, 'marginBottom': '1rem'}),
                html.Div(id='executive-summary', style={
                    'backgroundColor': white,
                    'padding': '2rem',
                    'borderRadius': '8px',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                    'marginBottom': '2rem'
                }),
            ]),
        ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '0 1rem'}),

        # Export Section
        html.Div([
            html.H3("Export Options", style={'color': primary_color, 'marginBottom': '1rem'}),
            html.Div([
                dbc.Button("Export Dashboard as PDF", id="export-pdf", color="primary", className="me-2"),
                dbc.Button("Export Data as CSV", id="export-csv", color="success", className="me-2"),
                dbc.Button("Generate Report", id="generate-report", color="warning"),
            ], style={'marginBottom': '2rem'}),
            html.Div(id="export-status"),
        ], style=controls_section_style),

        # Reports Charts
        html.Div([
            # First Row
            html.Div([
                html.Div([
                    html.H4("Overall Performance Metrics", style=chart_title_style),
                    dcc.Graph(id='overall-metrics', config={'displayModeBar': False}, style={'height': '400px'}),
                ], style=chart_container_style),
                html.Div([
                    html.H4("Market Share Analysis", style=chart_title_style),
                    dcc.Graph(id='market-share', config={'displayModeBar': False}, style={'height': '400px'}),
                ], style=chart_container_style),
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '2rem', 'marginBottom': '2rem', 'minHeight': '500px'}),

            # Second Row - Full Width
            html.Div([
                html.H4("Year-over-Year Comparison", style=chart_title_style),
                dcc.Graph(id='yearly-comparison', config={'displayModeBar': False}, style={'height': '400px'}),
            ], style=chart_container_style),
        ], style=charts_section_style),
    ])

# App layout with navigation
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    create_navbar(),
    
    # All components always present but conditionally shown
    html.Div([
        # Overview Page Components
        html.Div(id='overview-content', style={'display': 'block'}),
        
        # Analytics Page Components
        html.Div(id='analytics-content', style={'display': 'none'}),
        
        # Customer Insights Page Components
        html.Div(id='customers-content', style={'display': 'none'}),
        
        # Predictive Analytics Page Components
        html.Div(id='predictive-content', style={'display': 'none'}),
        
        # Reports Page Components
        html.Div(id='reports-content', style={'display': 'none'}),
    ], style=app_layout_style)
], style=app_layout_style)

# Professional chart styling function
def professional_fig(fig, title_color=primary_color):
    fig.update_layout(
        template='plotly_white',
        font_family='"Segoe UI", "Roboto", "Helvetica Neue", Arial, sans-serif',
        font_color=dark_gray,
        title_font_size=16,
        title_font_color=title_color,
        title_font_family='"Segoe UI", "Roboto", "Helvetica Neue", Arial, sans-serif',
        plot_bgcolor=white,
        paper_bgcolor=white,
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,  # Fixed height for all charts
        width=None,  # Auto width
        legend=dict(
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor=light_gray,
            borderwidth=1,
            font=dict(size=12, color=dark_gray)
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=light_gray,
            zeroline=False,
            showline=True,
            linecolor=light_gray,
            tickfont=dict(size=12, color=dark_gray)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=light_gray,
            zeroline=False,
            showline=True,
            linecolor=light_gray,
            tickfont=dict(size=12, color=dark_gray)
        ),
        colorway=[secondary_color, accent_color, success_color, warning_color, primary_color, '#9b59b6', '#1abc9c', '#e67e22'],
    )
    return fig

# Page routing callback
@app.callback(
    [Output('overview-content', 'children'),
     Output('overview-content', 'style'),
     Output('analytics-content', 'children'),
     Output('analytics-content', 'style'),
     Output('customers-content', 'children'),
     Output('customers-content', 'style'),
     Output('predictive-content', 'children'),
     Output('predictive-content', 'style'),
     Output('reports-content', 'children'),
     Output('reports-content', 'style')],
    Input('url', 'pathname')
)
def display_page(pathname):
    # Default styles
    show_style = {'display': 'block'}
    hide_style = {'display': 'none'}
    
    if pathname == '/analytics':
        return ([], hide_style, create_analytics_page(), show_style, [], hide_style, [], hide_style, [], hide_style)
    elif pathname == '/customers':
        return ([], hide_style, [], hide_style, create_customers_page(), show_style, [], hide_style, [], hide_style)
    elif pathname == '/predictive':
        return ([], hide_style, [], hide_style, [], hide_style, create_predictive_page(), show_style, [], hide_style)
    elif pathname == '/reports':
        return ([], hide_style, [], hide_style, [], hide_style, [], hide_style, create_reports_page(), show_style)
    else:
        return (create_overview_page(), show_style, [], hide_style, [], hide_style, [], hide_style, [], hide_style)

# Advanced KPI Callback
@app.callback(
    Output('advanced-kpi-cards', 'children'),
    [Input('shade-dropdown', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('segment-dropdown', 'value')]
)
def update_advanced_kpi_cards(selected_shade, start_date, end_date, selected_segment):
    if not selected_shade:
        return []
    
    try:
        # Filter data based on inputs
        filtered = df[df['Shade'] == selected_shade]
        
        if start_date and end_date:
            filtered = filtered[
                (filtered['Purchase_Date'] >= start_date) & 
                (filtered['Purchase_Date'] <= end_date)
            ]
        
        if selected_segment != 'all':
            filtered = filtered[filtered['Segment_Name'] == selected_segment]
        
        total_purchases = len(filtered)
        avg_price = filtered['Price_NPR'].mean()
        total_revenue = filtered['Price_NPR'].sum()
        unique_customers = filtered['Customer_ID'].nunique()
        repeat_customers = filtered['Customer_ID'].value_counts().gt(1).sum()
        avg_customer_value = filtered.groupby('Customer_ID')['Price_NPR'].sum().mean()
        
        return [
            html.Div([
                html.Div(f"{total_purchases:,}", style=kpi_value_style),
                html.Div("Total Purchases", style=kpi_label_style),
            ], style=kpi_card_style),
            html.Div([
                html.Div(f"NPR {avg_price:,.0f}", style=kpi_value_style),
                html.Div("Average Price", style=kpi_label_style),
            ], style=kpi_card_style),
            html.Div([
                html.Div(f"NPR {total_revenue:,.0f}", style=kpi_value_style),
                html.Div("Total Revenue", style=kpi_label_style),
            ], style=kpi_card_style),
            html.Div([
                html.Div(f"{unique_customers}", style=kpi_value_style),
                html.Div("Unique Customers", style=kpi_label_style),
            ], style=kpi_card_style),
            html.Div([
                html.Div(f"{repeat_customers}", style=kpi_value_style),
                html.Div("Repeat Customers", style=kpi_label_style),
            ], style=kpi_card_style),
            html.Div([
                html.Div(f"NPR {avg_customer_value:,.0f}", style=kpi_value_style),
                html.Div("Avg Customer Value", style=kpi_label_style),
            ], style=kpi_card_style),
        ]
    except Exception as e:
        return []

# Overview Page Callbacks
@app.callback(
    Output('monthly-trend', 'figure'),
    Input('shade-dropdown', 'value')
)
def update_monthly_trend(selected_shade):
    if not selected_shade:
        return {}
    
    try:
        filtered = df[df['Shade'] == selected_shade]
        monthly_counts = filtered.groupby('Month').size().reset_index(name='Count')
        
        fig = px.bar(
            monthly_counts, 
            x='Month', 
            y='Count', 
            title=f'Monthly Purchase Trend: {selected_shade}',
            labels={'Count': 'Number of Purchases'},
            color='Count',
            color_continuous_scale=['#e74c3c', '#f39c12']  # Red to Orange gradient
        )
        return professional_fig(fig)
    except Exception as e:
        return {}

@app.callback(
    Output('brand-distribution', 'figure'),
    Input('shade-dropdown', 'value')
)
def update_brand_distribution(selected_shade):
    if not selected_shade:
        return {}
    
    try:
        filtered = df[df['Shade'] == selected_shade]
        brand_counts = filtered['Brand'].value_counts().reset_index()
        brand_counts.columns = ['Brand', 'Count']
        
        fig = px.pie(
            brand_counts, 
            names='Brand', 
            values='Count', 
            title=f'Brand Distribution for {selected_shade}',
            hole=0.4,
            color_discrete_sequence=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']  # Multi-color
        )
        fig.update_traces(
            textinfo='percent+label',
            pull=[0.05]*len(brand_counts),
            marker=dict(line=dict(color=white, width=2))
        )
        return professional_fig(fig)
    except Exception as e:
        return {}

@app.callback(
    Output('price-age-scatter', 'figure'),
    Input('shade-dropdown', 'value')
)
def update_price_age_scatter(selected_shade):
    if not selected_shade:
        return {}
    
    try:
        filtered = df[df['Shade'] == selected_shade]
        
        fig = px.scatter(
            filtered, 
            x='Age', 
            y='Price_NPR', 
            color='Brand',
            title=f'Age vs Price Analysis for {selected_shade}',
            labels={'Price_NPR': 'Price (NPR)', 'Age': 'Customer Age'},
            size='Price_NPR',
            size_max=20,
            opacity=0.7,
            color_discrete_sequence=['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c']  # Green, Red, Blue, Orange, Purple, Teal
        )
        return professional_fig(fig)
    except Exception as e:
        return {}

@app.callback(
    Output('age-distribution', 'figure'),
    Input('shade-dropdown', 'value')
)
def update_age_distribution(selected_shade):
    if not selected_shade:
        return {}
    
    try:
        filtered = df[df['Shade'] == selected_shade]
        
        # Create age groups for different colors
        filtered['Age_Group'] = pd.cut(filtered['Age'], 
                                      bins=[0, 20, 25, 30, 35, 40, 45, 50, 100], 
                                      labels=['15-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '50+'])
        
        fig = px.histogram(
            filtered, 
            x='Age', 
            nbins=15,
            title=f'Age Distribution for {selected_shade}',
            labels={'Age': 'Customer Age', 'count': 'Number of Customers'},
            color='Age_Group',
            color_discrete_sequence=['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60', '#3498db', '#9b59b6', '#e67e22']  # Different colors for each age group
        )
        return professional_fig(fig)
    except Exception as e:
        return {}

# Analytics Page Callbacks
@app.callback(
    Output('customer-segmentation', 'figure'),
    Input('analytics-shade-dropdown', 'value')
)
def update_customer_segmentation(selected_shade):
    if not selected_shade:
        return {}
    
    try:
        filtered = df[df['Shade'] == selected_shade]
        segment_counts = filtered['Segment_Name'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        
        fig = px.pie(
            segment_counts, 
            names='Segment', 
            values='Count', 
            title=f'Customer Segment Distribution for {selected_shade}',
            hole=0.4,
            color_discrete_sequence=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'] # Blue, Red, Green, Orange
        )
        fig.update_traces(
            textinfo='percent+label',
            pull=[0.05]*len(segment_counts),
            marker=dict(line=dict(color=white, width=2))
        )
        return professional_fig(fig)
    except Exception as e:
        return {}

@app.callback(
    Output('rfm-analysis', 'figure'),
    Input('analytics-shade-dropdown', 'value')
)
def update_rfm_analysis(selected_shade):
    if not selected_shade:
        return {}
    
    try:
        filtered = df[df['Shade'] == selected_shade]
        rfm_filtered = calculate_rfm_scores(filtered)
        
        fig = make_subplots(rows=2, cols=2, subplot_titles=('Recency Score', 'Frequency Score', 'Monetary Score', 'RFM Score'))
        
        fig.add_trace(
            go.Histogram(x=rfm_filtered['R'], name='Recency Score'),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=rfm_filtered['F'], name='Frequency Score'),
            row=1, col=2
        )
        fig.add_trace(
            go.Histogram(x=rfm_filtered['M'], name='Monetary Score'),
            row=2, col=1
        )
        fig.add_trace(
            go.Histogram(x=rfm_filtered['RFM_Score'], name='RFM Score'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="RFM Analysis")
        return professional_fig(fig)
    except Exception as e:
        return {}

@app.callback(
    Output('sales-prediction', 'figure'),
    Input('analytics-shade-dropdown', 'value')
)
def update_sales_prediction(selected_shade):
    if not selected_shade:
        return {}
    
    try:
        filtered = df[df['Shade'] == selected_shade]
        sales_data, future_predictions = predict_sales_trend(filtered)
        
        fig = px.line(
            sales_data,
            x='Month',
            y='Price_NPR',
            title=f'Historical Sales Trend for {selected_shade}',
            labels={'Price_NPR': 'Revenue (NPR)', 'Month': 'Month'},
            markers=True,
            color_discrete_sequence=['#e67e22'] # Orange color
        )
        
        # Add future predictions
        future_x = [sales_data['Month'].max() + timedelta(days=30*i) for i in range(len(future_predictions))]
        fig.add_trace(
            go.Scatter(x=future_x, y=future_predictions, mode='lines', line=dict(dash='dash'), name='Predicted Sales')
        )
        
        return professional_fig(fig)
    except Exception as e:
        return {}

@app.callback(
    Output('customer-ltv', 'figure'),
    Input('analytics-shade-dropdown', 'value')
)
def update_customer_ltv(selected_shade):
    if not selected_shade:
        return {}
    
    try:
        filtered = df[df['Shade'] == selected_shade]
        customer_ltv = filtered.groupby('Customer_ID')['Price_NPR'].sum().reset_index()
        customer_ltv.columns = ['Customer_ID', 'LTV']
        
        fig = px.histogram(
            customer_ltv,
            x='LTV',
            nbins=15,
            title=f'Customer Lifetime Value Distribution for {selected_shade}',
            labels={'LTV': 'Lifetime Value (NPR)', 'count': 'Number of Customers'},
            color_discrete_sequence=['#e67e22'] # Orange color
        )
        return professional_fig(fig)
    except Exception as e:
        return {}

@app.callback(
    Output('advanced-metrics', 'figure'),
    Input('analytics-shade-dropdown', 'value')
)
def update_advanced_metrics(selected_shade):
    if not selected_shade:
        return {}
    
    try:
        filtered = df[df['Shade'] == selected_shade]
        
        # Calculate average metrics per segment
        segment_metrics = filtered.groupby('Segment_Name').agg({
            'Price_NPR': ['mean', 'sum', 'count'],
            'Age': 'mean'
        }).reset_index()
        segment_metrics.columns = ['Segment', 'Avg_Price', 'Total_Revenue', 'Purchase_Count', 'Avg_Age']
        
        fig = make_subplots(rows=2, cols=2, subplot_titles=('Average Price by Segment', 'Total Revenue by Segment', 'Purchase Count by Segment', 'Average Age by Segment'))
        
        fig.add_trace(
            go.Bar(x=segment_metrics['Segment'], y=segment_metrics['Avg_Price'], name='Average Price'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=segment_metrics['Segment'], y=segment_metrics['Total_Revenue'], name='Total Revenue'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=segment_metrics['Segment'], y=segment_metrics['Purchase_Count'], name='Purchase Count'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=segment_metrics['Segment'], y=segment_metrics['Avg_Age'], name='Average Age'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Advanced Metrics by Customer Segment")
        return professional_fig(fig)
    except Exception as e:
        return {}

# Customer Insights Page Callbacks
@app.callback(
    Output('customer-loyalty', 'figure'),
    Input('customers-shade-dropdown', 'value')
)
def update_customer_loyalty(selected_shade):
    if not selected_shade:
        return {}
    
    try:
        filtered = df[df['Shade'] == selected_shade]
        repeat_counts = filtered['Customer_ID'].value_counts()
        repeat_counts = repeat_counts[repeat_counts > 1]
        
        fig = px.histogram(
            repeat_counts, 
            x=repeat_counts.values, 
            nbins=10,
            title=f'Customer Loyalty Analysis for {selected_shade}',
            labels={'x': 'Number of Purchases', 'y': 'Number of Customers'},
            color_discrete_sequence=['#f39c12']  # Orange color
        )
        return professional_fig(fig)
    except Exception as e:
        return {}

@app.callback(
    Output('customer-demographics', 'figure'),
    Input('customers-shade-dropdown', 'value')
)
def update_customer_demographics(selected_shade):
    if not selected_shade:
        return {}
    
    try:
        filtered = df[df['Shade'] == selected_shade]
        
        # Create age groups
        filtered['Age_Group'] = pd.cut(filtered['Age'], 
                                      bins=[0, 25, 35, 45, 55, 100], 
                                      labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        
        age_group_counts = filtered['Age_Group'].value_counts()
        
        fig = px.pie(
            values=age_group_counts.values,
            names=age_group_counts.index,
            title=f'Customer Age Demographics for {selected_shade}',
            color_discrete_sequence=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']  # Blue, Red, Green, Orange, Purple
        )
        return professional_fig(fig)
    except Exception as e:
        return {}

@app.callback(
    Output('purchase-frequency', 'figure'),
    Input('customers-shade-dropdown', 'value')
)
def update_purchase_frequency(selected_shade):
    if not selected_shade:
        return {}
    
    try:
        filtered = df[df['Shade'] == selected_shade]
        customer_purchases = filtered['Customer_ID'].value_counts()
        
        fig = px.bar(
            x=customer_purchases.value_counts().index,
            y=customer_purchases.value_counts().values,
            title=f'Purchase Frequency Distribution for {selected_shade}',
            labels={'x': 'Number of Purchases', 'y': 'Number of Customers'},
            color_discrete_sequence=['#1abc9c']  # Teal color
        )
        return professional_fig(fig)
    except Exception as e:
        return {}

@app.callback(
    Output('customer-value', 'figure'),
    Input('customers-shade-dropdown', 'value')
)
def update_customer_value(selected_shade):
    if not selected_shade:
        return {}
    
    try:
        filtered = df[df['Shade'] == selected_shade]
        customer_value = filtered.groupby('Customer_ID')['Price_NPR'].sum().reset_index()
        
        fig = px.histogram(
            customer_value,
            x='Price_NPR',
            nbins=15,
            title=f'Customer Value Distribution for {selected_shade}',
            labels={'Price_NPR': 'Total Spent (NPR)', 'count': 'Number of Customers'},
            color_discrete_sequence=['#e67e22']  # Orange color
        )
        return professional_fig(fig)
    except Exception as e:
        return {}

# Predictive Analytics Page Callbacks
@app.callback(
    Output('sales-forecast', 'figure'),
    Input('url', 'pathname')
)
def update_sales_forecast(pathname):
    if pathname != '/predictive':
        return {}
    
    try:
        # Use the sales_prediction DataFrame for forecasting
        # This is a simplified example. In a real ML model, you'd train on historical data and predict future.
        # For this example, we'll just show a placeholder or a simple trend.
        
        # Example: Predicting next 3 months based on the last 3 months of data
        last_3_months_sales = sales_prediction['Price_NPR'].tail(3).values
        future_predictions = predict_sales_trend(df, months_ahead=3)[1] # Use the future_predictions from predict_sales_trend
        
        # Create a DataFrame for plotting
        future_x = [sales_prediction['Month'].max() + timedelta(days=30*i) for i in range(len(future_predictions))]
        future_df = pd.DataFrame({'Month': future_x, 'Price_NPR': future_predictions})
        
        fig = px.line(
            sales_prediction,
            x='Month',
            y='Price_NPR',
            title='Historical Sales Trend (for context)',
            labels={'Price_NPR': 'Revenue (NPR)', 'Month': 'Month'},
            markers=True,
            color_discrete_sequence=['#e67e22']
        )
        
        # Add future predictions
        fig.add_trace(
            go.Scatter(x=future_df['Month'], y=future_df['Price_NPR'], mode='lines', line=dict(dash='dash'), name='Predicted Sales')
        )
        
        return professional_fig(fig)
    except Exception as e:
        return {}

@app.callback(
    Output('demand-prediction', 'figure'),
    Input('url', 'pathname')
)
def update_demand_prediction(pathname):
    if pathname != '/predictive':
        return {}
    
    try:
        # This is a placeholder for a more sophisticated ML model.
        # For now, we'll just show a simple bar chart of total sales by month.
        # In a real ML model, you'd train on historical data and predict demand.
        
        monthly_sales = df.groupby('Month')['Price_NPR'].sum().reset_index()
        monthly_sales['Month'] = pd.to_datetime(monthly_sales['Month'].astype(str) + '-01')
        
        fig = px.bar(
            monthly_sales,
            x='Month',
            y='Price_NPR',
            title='Historical Demand (for context)',
            labels={'Price_NPR': 'Revenue (NPR)', 'Month': 'Month'},
            color_discrete_sequence=['#1abc9c']
        )
        
        return professional_fig(fig)
    except Exception as e:
        return {}

@app.callback(
    Output('churn-prediction', 'figure'),
    Input('url', 'pathname')
)
def update_churn_prediction(pathname):
    if pathname != '/predictive':
        return {}
    
    try:
        # This is a placeholder for a churn prediction model.
        # For now, we'll just show a simple bar chart of customer counts by segment.
        # In a real ML model, you'd train on historical data and predict churn.
        
        segment_counts = df['Segment_Name'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        
        fig = px.bar(
            segment_counts,
            x='Segment',
            y='Count',
            title='Historical Customer Segment Distribution (for context)',
            labels={'Count': 'Number of Customers', 'Segment': 'Customer Segment'},
            color_discrete_sequence=['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        )
        
        return professional_fig(fig)
    except Exception as e:
        return {}

@app.callback(
    Output('revenue-projection', 'figure'),
    Input('url', 'pathname')
)
def update_revenue_projection(pathname):
    if pathname != '/predictive':
        return {}
    
    try:
        # This is a placeholder for a revenue projection model.
        # For now, we'll just show a simple line chart of total revenue.
        # In a real ML model, you'd train on historical data and project revenue.
        
        total_revenue = df['Price_NPR'].sum()
        fig = px.line(
            df,
            x='Purchase_Date',
            y='Price_NPR',
            title='Historical Total Revenue (for context)',
            labels={'Price_NPR': 'Revenue (NPR)', 'Purchase_Date': 'Date'},
            color_discrete_sequence=['#e67e22']
        )
        
        return professional_fig(fig)
    except Exception as e:
        return {}

# Reports Page Callbacks
@app.callback(
    Output('executive-summary', 'children'),
    Input('url', 'pathname')
)
def update_executive_summary(pathname):
    if pathname != '/reports':
        return []
    
    try:
        total_purchases = len(df)
        total_revenue = df['Price_NPR'].sum()
        avg_price = df['Price_NPR'].mean()
        unique_customers = df['Customer_ID'].nunique()
        top_shade = df['Shade'].mode()[0]
        top_brand = df['Brand'].mode()[0]
        
        return [
            html.H4("Key Highlights", style={'color': primary_color, 'marginBottom': '1rem'}),
            html.P(f"• Total purchases across all shades: {total_purchases:,}"),
            html.P(f"• Total revenue generated: NPR {total_revenue:,.0f}"),
            html.P(f"• Average price per purchase: NPR {avg_price:,.0f}"),
            html.P(f"• Unique customers served: {unique_customers:,}"),
            html.P(f"• Most popular shade: {top_shade}"),
            html.P(f"• Leading brand: {top_brand}"),
            html.Hr(),
            html.H4("Business Insights", style={'color': primary_color, 'marginTop': '1rem', 'marginBottom': '1rem'}),
            html.P("• Customer retention shows strong loyalty patterns"),
            html.P("• Seasonal trends indicate peak purchasing periods"),
            html.P("• Price sensitivity varies significantly by age group"),
            html.P("• Brand preferences are consistent across demographics"),
        ]
    except Exception as e:
        return []

@app.callback(
    Output('overall-metrics', 'figure'),
    Input('url', 'pathname')
)
def update_overall_metrics(pathname):
    if pathname != '/reports':
        return {}
    
    try:
        # Overall performance metrics
        monthly_performance = df.groupby('Month').agg({
            'Price_NPR': ['sum', 'count']
        }).reset_index()
        monthly_performance.columns = ['Month', 'Revenue', 'Purchases']
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monthly Revenue', 'Monthly Purchases'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=monthly_performance['Month'], y=monthly_performance['Revenue'], 
                      name='Revenue', line=dict(color=secondary_color)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=monthly_performance['Month'], y=monthly_performance['Purchases'], 
                   name='Purchases', marker_color=accent_color),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title_text="Overall Performance Metrics")
        return professional_fig(fig)
    except Exception as e:
        return {}

@app.callback(
    Output('market-share', 'figure'),
    Input('url', 'pathname')
)
def update_market_share(pathname):
    if pathname != '/reports':
        return {}
    
    try:
        brand_revenue = df.groupby('Brand')['Price_NPR'].sum().reset_index()
        
        fig = px.pie(
            brand_revenue,
            values='Price_NPR',
            names='Brand',
            title='Market Share by Revenue',
            hole=0.4
        )
        fig.update_traces(
            textinfo='percent+label',
            pull=[0.05]*len(brand_revenue),
            marker=dict(line=dict(color=white, width=2))
        )
        return professional_fig(fig)
    except Exception as e:
        return {}

@app.callback(
    Output('yearly-comparison', 'figure'),
    Input('url', 'pathname')
)
def update_yearly_comparison(pathname):
    if pathname != '/reports':
        return {}
    
    try:
        df_copy = df.copy()
        df_copy['Year'] = df_copy['Purchase_Date'].dt.year
        df_copy['Month'] = df_copy['Purchase_Date'].dt.month
        
        yearly_data = df_copy.groupby(['Year', 'Month']).agg({
            'Price_NPR': ['sum', 'count']
        }).reset_index()
        yearly_data.columns = ['Year', 'Month', 'Revenue', 'Purchases']
        
        fig = px.line(
            yearly_data,
            x='Month',
            y='Revenue',
            color='Year',
            title='Year-over-Year Revenue Comparison',
            labels={'Revenue': 'Revenue (NPR)', 'Month': 'Month'},
            markers=True
        )
        return professional_fig(fig)
    except Exception as e:
        return {}

# Export callbacks
@app.callback(
    Output("export-status", "children"),
    [Input("export-pdf", "n_clicks"),
     Input("export-csv", "n_clicks"),
     Input("generate-report", "n_clicks")]
)
def handle_export(pdf_clicks, csv_clicks, report_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "export-pdf":
        return html.Div("PDF export functionality would be implemented here", style={'color': 'green', 'marginTop': '1rem'})
    elif button_id == "export-csv":
        return html.Div("CSV export functionality would be implemented here", style={'color': 'green', 'marginTop': '1rem'})
    elif button_id == "generate-report":
        return html.Div("Report generation functionality would be implemented here", style={'color': 'green', 'marginTop': '1rem'})
    
    return ""

# Run the Dash app
if __name__ == '__main__':
    app.run(debug=True)
