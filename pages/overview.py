import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import io
import requests

# Register page
dash.register_page(__name__, path='/overview', name='Overview')

# =========================================================
# LOAD DATA WITH CACHING
# =========================================================
OVERVIEW_FILE_ID = "1IniF9u4atIRo69vjZAQiB6iXgtF50hJ8"
OVERVIEW_URL = f"https://drive.google.com/uc?export=download&id={OVERVIEW_FILE_ID}"

# Global variable to cache data (loaded once per worker)
_cached_df = None

def load_data():
    """Load data with caching to avoid repeated downloads"""
    global _cached_df
    if _cached_df is not None:
        return _cached_df.copy()
    
    try:
        response = requests.get(OVERVIEW_URL, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        
        # Data preprocessing
        df['Day'] = pd.to_datetime(df['Day'], format='%d-%m-%Y')
        df['Day_str'] = df['Day'].dt.strftime('%d-%m-%Y')
        
        # Rename columns
        COL_MAP = {
            'Campaign Type': 'campaign_type',
            'Advertiser': 'advertiser',
            'Campaign': 'campaign',
            'Ad Impressions': 'impressions',
            'Clicks': 'clicks',
            'Weighted Conversion': 'conversions',
            'Advertiser Cost': 'adv_cost',
            'Max System Cost': 'max_cost',
            'Payout': 'payout',
            'Actual Advertiser Payout': 'actual_payout'
        }
        df = df.rename(columns={c: COL_MAP[c] for c in df.columns if c in COL_MAP})
        
        # Convert numeric columns
        numeric_cols = ['impressions', 'clicks', 'conversions', 'adv_cost', 'max_cost', 
                        'payout', 'actual_payout']
        for c in numeric_cols:
            if c not in df.columns:
                df[c] = 0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        
        # Calculate metrics
        df['ctr'] = np.where(df['impressions'] > 0, 100 * df['clicks'] / df['impressions'], 0)
        df['cvr'] = np.where(df['clicks'] > 0, 100 * df['conversions'] / df['clicks'], 0)
        df['cpa'] = np.where(df['conversions'] > 0, df['adv_cost'] / df['conversions'], 0)
        df['mnet_roas'] = np.where(df['max_cost'] > 0, df['payout'] / df['max_cost'], 0)
        df['adv_roas'] = np.where(df['adv_cost'] > 0, df['payout'] / df['adv_cost'], 0)
        
        _cached_df = df
        return df.copy()
        
    except Exception as e:
        print(f"Error loading Overview data: {e}")
        return pd.DataFrame()

# Load data once at module level
df = load_data()

# Pre-compute options
ALL_ADVERTISERS = sorted(df['advertiser'].dropna().unique()) if len(df) > 0 else []
ALL_CAMPAIGN_TYPES = sorted(df['campaign_type'].dropna().unique()) if len(df) > 0 else []

if len(df) > 0:
    min_date = df['Day'].min()
    max_date = df['Day'].max()
    default_start = max_date - pd.Timedelta(days=14)  # Last 2 weeks
    default_end = max_date
else:
    min_date = datetime(2025, 12, 2)
    max_date = datetime(2025, 12, 16)
    default_start = min_date
    default_end = max_date

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def aggregate_metrics(data, group_cols):
    """Aggregate metrics with weighted averages"""
    if len(data) == 0:
        return pd.DataFrame()
    
    agg_dict = {
        'impressions': 'sum',
        'clicks': 'sum',
        'conversions': 'sum',
        'adv_cost': 'sum',
        'max_cost': 'sum',
        'payout': 'sum',
        'actual_payout': 'sum'
    }
    
    grouped = data.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    
    # Calculate weighted metrics
    grouped['ctr'] = np.where(grouped['impressions'] > 0, 
                              100 * grouped['clicks'] / grouped['impressions'], 0)
    grouped['cvr'] = np.where(grouped['clicks'] > 0, 
                              100 * grouped['conversions'] / grouped['clicks'], 0)
    grouped['cpa'] = np.where(grouped['conversions'] > 0, 
                              grouped['adv_cost'] / grouped['conversions'], 0)
    grouped['mnet_roas'] = np.where(grouped['max_cost'] > 0, 
                                    grouped['payout'] / grouped['max_cost'], 0)
    grouped['adv_roas'] = np.where(grouped['adv_cost'] > 0, 
                                   grouped['payout'] / grouped['adv_cost'], 0)
    
    return grouped

# =========================================================
# LAYOUT
# =========================================================

layout = dbc.Container(fluid=True, style={'backgroundColor': '#111'}, children=[
    html.H2("Top Level Overview", style={'color': '#5dade2', 'textAlign': 'center', 'padding': '20px'}),
    
    # ADD DATE RANGE HERE (before other filters)
    dbc.Row([
        dbc.Col([html.Label("Date Range:", style={'color': 'white', 'fontWeight': 'bold'}),
                 dcc.DatePickerRange(
                     id='ov_date_range',
                     start_date=default_start,
                     end_date=default_end,
                     min_date_allowed=min_date,
                     max_date_allowed=max_date,
                     display_format='DD-MM-YYYY',
                     style={'marginBottom': '10px'}
                 )
                ], width=6)
    ], style={'marginBottom': '20px'}),    
    # Filters
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id='ov_adv_dd',
            multi=True,
            placeholder="Select Advertiser(s)",
            options=[{'label': x, 'value': x} for x in ALL_ADVERTISERS],
            style={'color': 'black'}
        ), width=6),
        dbc.Col(dcc.Dropdown(
            id='ov_camp_type_dd',
            multi=True,
            placeholder="Select Campaign Type(s)",
            options=[{'label': x, 'value': x} for x in ALL_CAMPAIGN_TYPES],
            style={'color': 'black'}
        ), width=6)
    ], style={'marginBottom': '20px'}),
    
    html.Hr(style={'borderColor': '#444'}),
    
    # Aggregated Stats Card
    html.Div(id='ov_agg_stats', style={'marginBottom': '20px'}),
    
    html.Hr(style={'borderColor': '#444'}),
    
    # Campaign Table with DataTable (MUCH FASTER!)
    html.H4("Campaign Performance", style={'color': '#5dade2', 'marginTop': '20px'}),
    html.P("Use pagination controls below. Click on a campaign row to see day-wise breakdown.", 
           style={'color': '#aaa', 'fontSize': '12px'}),
    
    dcc.Loading(
        html.Div(id='ov_campaign_table_container'),
        type="circle",
        color="#5dade2"
    ),
    
    # Selected Campaign Day-wise Breakdown
    dbc.Collapse(
        dbc.Card([
            dbc.CardHeader([
                html.H5(id='ov_day_breakdown_title', style={'color': '#5dade2', 'display': 'inline-block', 'marginBottom': '0'}),
                dbc.Button("✕ Close", id="close_day_breakdown", size="sm", color="secondary", 
                           style={'float': 'right'})
            ], style={'backgroundColor': '#333'}),
            dbc.CardBody(id='ov_day_breakdown_table', style={'backgroundColor': '#1a1a1a'})
        ], style={'marginTop': '10px'}),
        id='ov_day_breakdown_collapse',
        is_open=False
    ),
    
    html.Hr(style={'borderColor': '#444'}),
    
    # Day-on-Day Trends
    html.H4("Day-on-Day Trends", style={'color': '#5dade2', 'marginTop': '20px'}),
    html.P("Select/deselect metrics to compare. Select campaigns to filter (none selected = overall view)", 
           style={'color': '#aaa', 'fontSize': '12px'}),
    
    dbc.Row([
        dbc.Col([
            html.Label("Select Metrics:", style={'color': 'white', 'fontWeight': 'bold'}),
            dcc.Checklist(
                id='ov_metrics_checklist',
                options=[
                    {'label': ' Impressions', 'value': 'impressions'},
                    {'label': ' Clicks', 'value': 'clicks'},
                    {'label': ' Conversions', 'value': 'conversions'},
                    {'label': ' CTR %', 'value': 'ctr'},
                    {'label': ' CVR %', 'value': 'cvr'},
                    {'label': ' CPA', 'value': 'cpa'},
                    {'label': ' Mnet ROAS', 'value': 'mnet_roas'},
                    {'label': ' Advertiser ROAS', 'value': 'adv_roas'}
                ],
                value=['clicks', 'conversions', 'cvr'],
                labelStyle={'display': 'inline-block', 'marginRight': '15px', 'color': 'white'},
                style={'marginBottom': '15px'}
            )
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("Select Campaigns:", 
                      style={'color': 'white', 'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='ov_campaign_multi_dd',
                multi=True,
                placeholder="All Campaigns (or select specific)",
                options=[],
                style={'color': 'black'}
            )
        ], width=12)
    ], style={'marginBottom': '20px'}),
    
    dcc.Loading(
        dcc.Graph(id='ov_daily_trends', config={'displayModeBar': True}),
        type="circle",
        color="#5dade2"
    )
])

# =========================================================
# CALLBACKS
# =========================================================
@callback(
    [Output('ov_agg_stats', 'children'),
     Output('ov_campaign_table_container', 'children'),
     Output('ov_campaign_multi_dd', 'options')],
    [Input('ov_adv_dd', 'value'),
     Input('ov_camp_type_dd', 'value'),
     Input('ov_date_range', 'start_date'),
     Input('ov_date_range', 'end_date')]
)
def update_campaign_table(advertisers, campaign_types, start_date, end_date):
    """Update campaign table with pagination and conditional formatting"""
    # Filter data
    filtered_df = df.copy()
    # Check if base data exists
    if len(filtered_df) == 0:
        return (
            html.Div("⚠️ No data loaded from source", style={'color': '#ff0000', 'padding': '20px'}),
            html.Div("Please check data source connection", style={'color': '#aaa'}),
            []
    )
    
    # Date filtering with normalization
    if start_date and end_date:
        start_dt = pd.to_datetime(start_date).normalize()
        end_dt = pd.to_datetime(end_date).normalize()
        filtered_df['Day_normalized'] = filtered_df['Day'].dt.normalize()
        filtered_df = filtered_df[(filtered_df['Day_normalized'] >= start_dt) & (filtered_df['Day_normalized'] <= end_dt)]
        filtered_df = filtered_df.drop('Day_normalized', axis=1)
    
    if advertisers:
        filtered_df = filtered_df[filtered_df['advertiser'].isin(advertisers)]
    if campaign_types:
        filtered_df = filtered_df[filtered_df['campaign_type'].isin(campaign_types)]
    
    if len(filtered_df) == 0:
        return (
            html.Div("No data available", style={'color': '#ff0000'}),
            html.Div("No campaigns to display", style={'color': '#aaa'}),
            []
        )
    
    # Calculate overall stats (for comparison)
    total_impressions = filtered_df['impressions'].sum()
    total_clicks = filtered_df['clicks'].sum()
    total_conversions = filtered_df['conversions'].sum()
    total_adv_cost = filtered_df['adv_cost'].sum()
    total_max_cost = filtered_df['max_cost'].sum()
    total_payout = filtered_df['payout'].sum()
    
    agg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
    agg_cvr = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
    agg_cpa = (total_adv_cost / total_conversions) if total_conversions > 0 else 0
    agg_mnet_roas = (total_payout / total_max_cost) if total_max_cost > 0 else 0
    agg_adv_roas = (total_payout / total_adv_cost) if total_adv_cost > 0 else 0
    
    # Stats display
    stats_display = dbc.Card([
        dbc.CardBody([
            html.H4("Overall Aggregated Stats", style={'color': '#17a2b8', 'marginBottom': '15px'}),
            dbc.Row([
                dbc.Col([html.Strong("Impressions: "), html.Span(f"{int(total_impressions):,}", style={'color': '#5dade2'})], width=2),
                dbc.Col([html.Strong("Clicks: "), html.Span(f"{int(total_clicks):,}", style={'color': '#5dade2'})], width=2),
                dbc.Col([html.Strong("Conversions: "), html.Span(f"{total_conversions:.2f}", style={'color': '#5dade2'})], width=2),
                dbc.Col([html.Strong("CTR: "), html.Span(f"{agg_ctr:.2f}%", style={'color': '#00ff00'})], width=1),
                dbc.Col([html.Strong("CVR: "), html.Span(f"{agg_cvr:.2f}%", style={'color': '#00ff00'})], width=1),
                dbc.Col([html.Strong("CPA: "), html.Span(f"${agg_cpa:.2f}", style={'color': '#ffcc00'})], width=2),
                dbc.Col([html.Strong("Mnet ROAS: "), html.Span(f"{agg_mnet_roas:.2f}", style={'color': '#00ff00'})], width=1),
                dbc.Col([html.Strong("Adv ROAS: "), html.Span(f"{agg_adv_roas:.2f}", style={'color': '#00ff00'})], width=1)
            ])
        ])
    ], style={'backgroundColor': '#222', 'border': '1px solid #444', 'color': '#aaa'})
    
    # Aggregate by campaign
    campaign_agg = aggregate_metrics(filtered_df, ['campaign'])
    campaign_agg = campaign_agg.round(2)
    campaign_agg = campaign_agg.sort_values('payout', ascending=False)
    
    # Create DataTable with conditional formatting
    table = dash_table.DataTable(
        id='campaign_table',
        data=campaign_agg.to_dict('records'),
        columns=[
            {'name': 'Campaign', 'id': 'campaign'},
            {'name': 'Impressions', 'id': 'impressions', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
            {'name': 'Clicks', 'id': 'clicks', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
            {'name': 'Conversions', 'id': 'conversions', 'type': 'numeric', 'format': {'specifier': ',.2f'}},
            {'name': 'CTR %', 'id': 'ctr', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'CVR %', 'id': 'cvr', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'CPA', 'id': 'cpa', 'type': 'numeric', 'format': {'specifier': '$.2f'}},
            {'name': 'Mnet ROAS', 'id': 'mnet_roas', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Adv ROAS', 'id': 'adv_roas', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Payout', 'id': 'payout', 'type': 'numeric', 'format': {'specifier': '$.2f'}},
        ],
        page_size=20,
        page_action='native',
        sort_action='native',
        filter_action='native',
        row_selectable='single',
        selected_rows=[],
        style_table={'overflowX': 'auto'},
        style_header={
            'backgroundColor': '#17a2b8',
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_data={
            'backgroundColor': '#222',
            'color': 'white'
        },
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#2a2a2a'},
            {'if': {'state': 'selected'}, 'backgroundColor': '#17a2b8', 'border': '1px solid white'},
            
            # CVR: green if > avg, red if < avg
            {'if': {'filter_query': f'{{cvr}} > {agg_cvr}', 'column_id': 'cvr'}, 
             'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{cvr}} < {agg_cvr}', 'column_id': 'cvr'}, 
             'color': '#ff0000', 'fontWeight': 'bold'},
            
            # CTR: green if > avg, red if < avg
            {'if': {'filter_query': f'{{ctr}} > {agg_ctr}', 'column_id': 'ctr'}, 
             'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{ctr}} < {agg_ctr}', 'column_id': 'ctr'}, 
             'color': '#ff0000', 'fontWeight': 'bold'},
            
            # CPA: green if < avg (lower is better), red if > avg
            {'if': {'filter_query': f'{{cpa}} < {agg_cpa}', 'column_id': 'cpa'}, 
             'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{cpa}} > {agg_cpa}', 'column_id': 'cpa'}, 
             'color': '#ff0000', 'fontWeight': 'bold'},
            
            # Mnet ROAS: green if > avg, red if < avg
            {'if': {'filter_query': f'{{mnet_roas}} > {agg_mnet_roas}', 'column_id': 'mnet_roas'}, 
             'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{mnet_roas}} < {agg_mnet_roas}', 'column_id': 'mnet_roas'}, 
             'color': '#ff0000', 'fontWeight': 'bold'},
            
            # Adv ROAS: green if > avg, red if < avg
            {'if': {'filter_query': f'{{adv_roas}} > {agg_adv_roas}', 'column_id': 'adv_roas'}, 
             'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{adv_roas}} < {agg_adv_roas}', 'column_id': 'adv_roas'}, 
             'color': '#ff0000', 'fontWeight': 'bold'},
        ],
        style_cell={'textAlign': 'left', 'padding': '10px', 'fontSize': '12px'}
    )
    
    campaign_options = [{'label': c, 'value': c} for c in sorted(filtered_df['campaign'].unique())]
    
    return stats_display, table, campaign_options

@callback(
    [Output('ov_day_breakdown_collapse', 'is_open'),
     Output('ov_day_breakdown_title', 'children'),
     Output('ov_day_breakdown_table', 'children'),
     Output('campaign_table', 'selected_rows')],
    [Input('campaign_table', 'selected_rows'),
     Input('campaign_table', 'data'),
     Input('close_day_breakdown', 'n_clicks'),
     Input('ov_adv_dd', 'value'),
     Input('ov_camp_type_dd', 'value'),
     Input('ov_date_range', 'start_date'),
     Input('ov_date_range', 'end_date')],
    prevent_initial_call=True
)
def show_day_breakdown(selected_rows, table_data, close_clicks, advertisers, campaign_types, start_date, end_date):
    """Show collapsible day-wise breakdown when a campaign row is clicked"""
    from dash import callback_context
    
    ctx = callback_context
    if not ctx.triggered:
        return False, "", html.Div(), []
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Close if close button clicked
    if trigger_id == 'close_day_breakdown':
        return False, "", html.Div(), []
    
    # Open if row selected
    if not selected_rows or not table_data:
        return False, "", html.Div(), []
    
    selected_campaign = table_data[selected_rows[0]]['campaign']
    
    # Filter data
    filtered_df = df.copy()
    
    # Date filtering with normalization
    if start_date and end_date:
        start_dt = pd.to_datetime(start_date).normalize()
        end_dt = pd.to_datetime(end_date).normalize()
        filtered_df['Day_normalized'] = filtered_df['Day'].dt.normalize()
        filtered_df = filtered_df[(filtered_df['Day_normalized'] >= start_dt) & (filtered_df['Day_normalized'] <= end_dt)]
        filtered_df = filtered_df.drop('Day_normalized', axis=1)
    
    if advertisers:
        filtered_df = filtered_df[filtered_df['advertiser'].isin(advertisers)]
    if campaign_types:
        filtered_df = filtered_df[filtered_df['campaign_type'].isin(campaign_types)]
    
    # Get day-wise data for selected campaign
    campaign_days = filtered_df[filtered_df['campaign'] == selected_campaign].copy()
    day_agg = aggregate_metrics(campaign_days, ['Day_str'])
    day_agg['Day_sort'] = pd.to_datetime(day_agg['Day_str'], format='%d-%m-%Y')
    day_agg = day_agg.sort_values('Day_sort', ascending=False)
    day_agg = day_agg.drop('Day_sort', axis=1)
    day_agg = day_agg.round(2)
    
    if len(day_agg) == 0:
        return False, "", html.Div(), []
    
    # Create day-wise table
    day_table = dash_table.DataTable(
        data=day_agg.to_dict('records'),
        columns=[
            {'name': 'Date', 'id': 'Day_str'},
            {'name': 'Impressions', 'id': 'impressions', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
            {'name': 'Clicks', 'id': 'clicks', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
            {'name': 'Conversions', 'id': 'conversions', 'type': 'numeric', 'format': {'specifier': ',.2f'}},
            {'name': 'CTR %', 'id': 'ctr', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'CVR %', 'id': 'cvr', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'CPA', 'id': 'cpa', 'type': 'numeric', 'format': {'specifier': '$.2f'}},
            {'name': 'Mnet ROAS', 'id': 'mnet_roas', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Adv ROAS', 'id': 'adv_roas', 'type': 'numeric', 'format': {'specifier': '.2f'}},
        ],
        page_size=15,
        page_action='native',
        sort_action='native',
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': '#444', 'color': 'white', 'fontWeight': 'bold'},
        style_data={'backgroundColor': '#1a1a1a', 'color': '#aaa', 'fontSize': '11px'},
        style_cell={'textAlign': 'left', 'padding': '8px'}
    )
    
    return True, f"📅 Day-wise Breakdown: {selected_campaign}", day_table, selected_rows

@callback(
    Output('ov_daily_trends', 'figure'),
    [Input('ov_adv_dd', 'value'),
     Input('ov_camp_type_dd', 'value'),
     Input('ov_metrics_checklist', 'value'),
     Input('ov_campaign_multi_dd', 'value'),
     Input('ov_date_range', 'start_date'),  # ADD THIS
     Input('ov_date_range', 'end_date')]
)
def update_daily_trends(advertisers, campaign_types, selected_metrics, selected_campaigns, start_date, end_date):
    """Update daily trends chart"""
    # Filter data
    trend_df = df.copy()
    
    # Date filtering
    # Date filtering with normalization
    # Date filtering with normalization
    if start_date and end_date:
        start_dt = pd.to_datetime(start_date).normalize()
        end_dt = pd.to_datetime(end_date).normalize()
        trend_df['Day_normalized'] = trend_df['Day'].dt.normalize()
        trend_df = trend_df[(trend_df['Day_normalized'] >= start_dt) & (trend_df['Day_normalized'] <= end_dt)]
        trend_df = trend_df.drop('Day_normalized', axis=1)
    
    if advertisers:
        trend_df = trend_df[trend_df['advertiser'].isin(advertisers)]
    if campaign_types:
        trend_df = trend_df[trend_df['campaign_type'].isin(campaign_types)]
    if selected_campaigns:
        trend_df = trend_df[trend_df['campaign'].isin(selected_campaigns)]
    
    if len(trend_df) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="No data available",
            plot_bgcolor='#111',
            paper_bgcolor='#111',
            font=dict(color='white')
        )
        return fig
    
    daily_agg = aggregate_metrics(trend_df, ['Day'])
    daily_agg = daily_agg.sort_values('Day')
    
    fig = go.Figure()
    
    metric_config = {
        'impressions': {'name': 'Impressions', 'color': '#1f77b4'},
        'clicks': {'name': 'Clicks', 'color': '#ff7f0e'},
        'conversions': {'name': 'Conversions', 'color': '#2ca02c'},
        'ctr': {'name': 'CTR %', 'color': '#d62728'},
        'cvr': {'name': 'CVR %', 'color': '#9467bd'},
        'cpa': {'name': 'CPA', 'color': '#8c564b'},
        'mnet_roas': {'name': 'Mnet ROAS', 'color': '#e377c2'},
        'adv_roas': {'name': 'Advertiser ROAS', 'color': '#7f7f7f'}
    }
    
    if not selected_metrics:
        selected_metrics = ['clicks']
    
    # Normalize metrics
    normalized_data = daily_agg.copy()
    for metric in selected_metrics:
        if metric in normalized_data.columns:
            max_val = normalized_data[metric].max()
            min_val = normalized_data[metric].min()
            if max_val > min_val:
                normalized_data[f'{metric}_norm'] = ((normalized_data[metric] - min_val) / (max_val - min_val)) * 100
            else:
                normalized_data[f'{metric}_norm'] = 50
    
    for metric in selected_metrics:
        if metric in metric_config and f'{metric}_norm' in normalized_data.columns:
            config = metric_config[metric]
            hover_text = [f"{config['name']}: {actual:.2f}<br>Date: {date.strftime('%d-%m-%Y')}"
                         for actual, date in zip(daily_agg[metric], daily_agg['Day'])]
            
            fig.add_trace(go.Scatter(
                x=normalized_data['Day'],
                y=normalized_data[f'{metric}_norm'],
                name=config['name'],
                mode='lines+markers',
                line=dict(color=config['color'], width=2),
                marker=dict(size=8),
                hovertext=hover_text,
                hoverinfo='text'
            ))
    
    title = 'Daily Metrics Trend - '
    if selected_campaigns:
        title += f"{len(selected_campaigns)} Campaign(s) Selected"
    else:
        title += "Overall"
    
    fig.update_layout(
        title=title,
        plot_bgcolor='#111',
        paper_bgcolor='#111',
        font=dict(color='white'),
        hovermode='x unified',
        height=600,
        xaxis=dict(title='Date', showgrid=True, gridcolor='#333'),
        yaxis=dict(title='Normalized Scale (0-100)', showgrid=True, gridcolor='#333'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    
    return fig



