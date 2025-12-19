import dash
from dash import dcc, html, Input, Output, State, callback, dash_table, clientside_callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Register page
dash.register_page(__name__, path='/overview', name='Overview')

# =========================================================
# LOAD DATA
# =========================================================
# =========================================================
# LOAD DATA FROM GOOGLE DRIVE
# =========================================================
import io
import requests

OVERVIEW_FILE_ID = "13sSmNN7f2e1FkCji6TVCA9BaGRDfneQO"
OVERVIEW_URL = f"https://drive.google.com/uc?export=download&id={OVERVIEW_FILE_ID}"

try:
    response = requests.get(OVERVIEW_URL)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
except Exception as e:
    print(f"Error loading Overview data: {e}")
    df = pd.DataFrame()
# =========================================================
# DATA PREPROCESSING
# =========================================================
# Parse date column (format: 15-12-2025)
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
    df[c] = pd.to_numeric(df.get(c, 0), errors='coerce').fillna(0)

# Calculate metrics
df['ctr'] = np.where(df['impressions'] > 0, 100 * df['clicks'] / df['impressions'], 0)
df['cvr'] = np.where(df['clicks'] > 0, 100 * df['conversions'] / df['clicks'], 0)
df['cpa'] = np.where(df['conversions'] > 0, df['adv_cost'] / df['conversions'], 0)
df['mnet_roas'] = np.where(df['max_cost'] > 0, df['payout'] / df['max_cost'], 0)
df['adv_roas'] = np.where(df['adv_cost'] > 0, df['payout'] / df['adv_cost'], 0)

# Pre-compute options
ALL_ADVERTISERS = sorted(df['advertiser'].dropna().unique())
ALL_CAMPAIGN_TYPES = sorted(df['campaign_type'].dropna().unique())

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def aggregate_metrics(data, group_cols):
    """Aggregate metrics with weighted averages"""
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
    
    # Campaign Table with Drill-down
    html.H4("Campaign Performance (Click ▶ to expand day-wise stats)", 
            style={'color': '#5dade2', 'marginTop': '20px'}),
    html.Div(id='ov_campaign_table_container'),
    
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
                    {'label': ' Advertiser ROAS', 'value': 'adv_roas'},
                    {'label': ' Payout', 'value': 'payout'},
                    {'label': ' Advertiser Cost', 'value': 'adv_cost'},
                    {'label': ' Max System Cost', 'value': 'max_cost'},
                    {'label': ' Actual Advertiser Payout', 'value': 'actual_payout'}
                ],
                value=['clicks', 'conversions', 'cvr'],
                labelStyle={'display': 'inline-block', 'marginRight': '15px', 'color': 'white'},
                style={'marginBottom': '15px'}
            )
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("Select Campaigns (scroll to see all):", 
                      style={'color': 'white', 'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.Checklist(
                id='ov_campaign_checklist',
                options=[],  # Will be populated by callback
                value=[],
                labelStyle={'display': 'block', 'marginBottom': '5px', 'color': 'white'},
                style={'maxHeight': '500px', 'overflowY': 'auto', 'padding': '10px',
                       'backgroundColor': '#222', 'borderRadius': '5px'}
            )
        ], width=2),
        dbc.Col([
            dcc.Loading(
                dcc.Graph(id='ov_daily_trends', config={'displayModeBar': True}),
                type="circle",
                color="#5dade2"
            )
        ], width=10)
    ])
])

# =========================================================
# CALLBACKS
# =========================================================
@callback(
    [Output('ov_agg_stats', 'children'),
     Output('ov_campaign_table_container', 'children'),
     Output('ov_daily_trends', 'figure'),
     Output('ov_campaign_checklist', 'options')],
    [Input('ov_adv_dd', 'value'),
     Input('ov_camp_type_dd', 'value'),
     Input('ov_metrics_checklist', 'value'),
     Input('ov_campaign_checklist', 'value')]
)
def update_overview(advertisers, campaign_types, selected_metrics, selected_campaigns):
    # Filter data
    filtered_df = df.copy()
    
    if advertisers:
        filtered_df = filtered_df[filtered_df['advertiser'].isin(advertisers)]
    if campaign_types:
        filtered_df = filtered_df[filtered_df['campaign_type'].isin(campaign_types)]
    
    if len(filtered_df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            plot_bgcolor='#111',
            paper_bgcolor='#111',
            font=dict(color='white')
        )
        return (
            html.Div("No data available", style={'color': '#ff0000'}),
            html.Div("No campaigns to display", style={'color': '#aaa'}),
            empty_fig,
            []

        )
    
    # Calculate overall aggregated stats
    total_impressions = filtered_df['impressions'].sum()
    total_clicks = filtered_df['clicks'].sum()
    total_conversions = filtered_df['conversions'].sum()
    total_adv_cost = filtered_df['adv_cost'].sum()
    total_max_cost = filtered_df['max_cost'].sum()
    total_payout = filtered_df['payout'].sum()
    total_actual_payout = filtered_df['actual_payout'].sum()
    
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
                dbc.Col([html.Strong("Impressions: ", style={'color': '#aaa'}),
                        html.Span(f"{int(total_impressions):,}", style={'color': '#5dade2', 'fontSize': '16px'})], width=2),
                dbc.Col([html.Strong("Clicks: ", style={'color': '#aaa'}),
                        html.Span(f"{int(total_clicks):,}", style={'color': '#5dade2', 'fontSize': '16px'})], width=2),
                dbc.Col([html.Strong("Conversions: ", style={'color': '#aaa'}),
                        html.Span(f"{total_conversions:.2f}", style={'color': '#5dade2', 'fontSize': '16px'})], width=2),
                dbc.Col([html.Strong("CTR: ", style={'color': '#aaa'}),
                        html.Span(f"{agg_ctr:.2f}%", style={'color': '#00ff00', 'fontSize': '16px'})], width=1),
                dbc.Col([html.Strong("CVR: ", style={'color': '#aaa'}),
                        html.Span(f"{agg_cvr:.2f}%", style={'color': '#00ff00', 'fontSize': '16px'})], width=1),
                dbc.Col([html.Strong("CPA: ", style={'color': '#aaa'}),
                        html.Span(f"${agg_cpa:.2f}", style={'color': '#ffcc00', 'fontSize': '16px'})], width=2),
                dbc.Col([html.Strong("Mnet ROAS: ", style={'color': '#aaa'}),
                        html.Span(f"{agg_mnet_roas:.2f}", style={'color': '#00ff00', 'fontSize': '16px'})], width=1),
                dbc.Col([html.Strong("Adv ROAS: ", style={'color': '#aaa'}),
                        html.Span(f"{agg_adv_roas:.2f}", style={'color': '#00ff00', 'fontSize': '16px'})], width=1)
            ])
        ])
    ], style={'backgroundColor': '#222', 'border': '1px solid #444'})
    
    # Aggregate by campaign
    campaign_agg = aggregate_metrics(filtered_df, ['campaign'])
    campaign_agg = campaign_agg.round(2)
    
    # Create expandable table
    table_rows = []
    
    for idx, row in campaign_agg.iterrows():
        campaign_name = row['campaign']
        unique_id = f"camp_{idx}"
        
        # Main campaign row
        campaign_row = html.Tr([
            html.Td(
                html.Button('▶', 
                           id={'type': 'expand-btn', 'index': unique_id},
                           n_clicks=0,
                           style={'backgroundColor': '#17a2b8', 'border': 'none', 
                                  'color': 'white', 'cursor': 'pointer', 'padding': '5px 10px',
                                  'borderRadius': '3px', 'fontSize': '14px'}),
                style={'width': '50px', 'textAlign': 'center', 'padding': '8px'}
            ),
            html.Td(campaign_name, style={'fontWeight': 'bold', 'color': '#5dade2', 'padding': '8px'}),
            html.Td(f"{int(row['impressions']):,}", style={'padding': '8px'}),
            html.Td(f"{int(row['clicks']):,}", style={'padding': '8px'}),
            html.Td(f"{row['conversions']:.2f}", style={'padding': '8px'}),
            html.Td(f"{row['ctr']:.2f}%", style={'padding': '8px'}),
            html.Td(f"{row['cvr']:.2f}%", style={'padding': '8px'}),
            html.Td(f"${row['cpa']:.2f}", style={'padding': '8px'}),
            html.Td(f"{row['mnet_roas']:.2f}", style={'padding': '8px'}),
            html.Td(f"{row['adv_roas']:.2f}", style={'padding': '8px'}),
            html.Td(f"${row['payout']:.2f}", style={'padding': '8px'}),
            html.Td(f"${row['adv_cost']:.2f}", style={'padding': '8px'}),
            html.Td(f"${row['max_cost']:.2f}", style={'padding': '8px'}),
            html.Td(f"${row['actual_payout']:.2f}", style={'padding': '8px'})
        ], style={'backgroundColor': '#222', 'borderBottom': '1px solid #444'})
        
        table_rows.append(campaign_row)
        
        # Day-wise breakdown rows
        # Day-wise breakdown rows
        campaign_days = filtered_df[filtered_df['campaign'] == campaign_name].copy()
        day_agg = aggregate_metrics(campaign_days, ['Day_str'])
        day_agg['Day_sort'] = pd.to_datetime(day_agg['Day_str'], format='%d-%m-%Y')
        day_agg = day_agg.sort_values('Day_sort', ascending=False)  # Latest first
        
        for day_idx, day_row in day_agg.iterrows():
            day_tr = html.Tr([
                html.Td('', style={'padding': '8px'}),
                html.Td(f"  └─ {day_row['Day_str']}", 
                       style={'paddingLeft': '30px', 'color': '#aaa', 'fontSize': '11px', 'padding': '8px'}),
                html.Td(f"{int(day_row['impressions']):,}", style={'padding': '8px', 'fontSize': '11px'}),
                html.Td(f"{int(day_row['clicks']):,}", style={'padding': '8px', 'fontSize': '11px'}),
                html.Td(f"{day_row['conversions']:.2f}", style={'padding': '8px', 'fontSize': '11px'}),
                html.Td(f"{day_row['ctr']:.2f}%", style={'padding': '8px', 'fontSize': '11px'}),
                html.Td(f"{day_row['cvr']:.2f}%", style={'padding': '8px', 'fontSize': '11px'}),
                html.Td(f"${day_row['cpa']:.2f}", style={'padding': '8px', 'fontSize': '11px'}),
                html.Td(f"{day_row['mnet_roas']:.2f}", style={'padding': '8px', 'fontSize': '11px'}),
                html.Td(f"{day_row['adv_roas']:.2f}", style={'padding': '8px', 'fontSize': '11px'}),
                html.Td(f"${day_row['payout']:.2f}", style={'padding': '8px', 'fontSize': '11px'}),
                html.Td(f"${day_row['adv_cost']:.2f}", style={'padding': '8px', 'fontSize': '11px'}),
                html.Td(f"${day_row['max_cost']:.2f}", style={'padding': '8px', 'fontSize': '11px'}),
                html.Td(f"${day_row['actual_payout']:.2f}", style={'padding': '8px', 'fontSize': '11px'})
            ], 
            id={'type': 'day-row', 'index': unique_id},
            style={'backgroundColor': '#1a1a1a', 'borderBottom': '1px solid #333', 'display': 'none'})
            
            table_rows.append(day_tr)
    
    campaign_table = html.Div([
        html.Table([
            html.Thead(html.Tr([
                html.Th('', style={'width': '50px', 'padding': '10px'}),
                html.Th('Campaign', style={'padding': '10px'}),
                html.Th('Impressions', style={'padding': '10px'}),
                html.Th('Clicks', style={'padding': '10px'}),
                html.Th('Conversions', style={'padding': '10px'}),
                html.Th('CTR %', style={'padding': '10px'}),
                html.Th('CVR %', style={'padding': '10px'}),
                html.Th('CPA', style={'padding': '10px'}),
                html.Th('Mnet ROAS', style={'padding': '10px'}),
                html.Th('Adv ROAS', style={'padding': '10px'}),
                html.Th('Payout', style={'padding': '10px'}),
                html.Th('Adv Cost', style={'padding': '10px'}),
                html.Th('Max Cost', style={'padding': '10px'}),
                html.Th('Actual Payout', style={'padding': '10px'})
            ], style={'backgroundColor': '#111', 'color': '#17a2b8', 'fontSize': '13px'})),
            html.Tbody(table_rows)
        ], style={'width': '100%', 'borderCollapse': 'collapse', 'color': 'white', 'fontSize': '12px'})
    ], style={'overflowX': 'auto'})
    
    # Daily trends chart
    
    campaign_options = [{'label': c, 'value': c} for c in sorted(filtered_df['campaign'].unique())]
    
    # Daily trends chart - filter by selected campaigns
    trend_df = filtered_df.copy()
    if selected_campaigns:
        trend_df = trend_df[trend_df['campaign'].isin(selected_campaigns)]
    
    daily_agg = aggregate_metrics(trend_df, ['Day'])
    daily_agg = daily_agg.sort_values('Day')
    
    fig = go.Figure()
    
    # Metric configurations
    metric_config = {
        'impressions': {'name': 'Impressions', 'color': '#1f77b4'},
        'clicks': {'name': 'Clicks', 'color': '#ff7f0e'},
        'conversions': {'name': 'Conversions', 'color': '#2ca02c'},
        'ctr': {'name': 'CTR %', 'color': '#d62728'},
        'cvr': {'name': 'CVR %', 'color': '#9467bd'},
        'cpa': {'name': 'CPA', 'color': '#8c564b'},
        'mnet_roas': {'name': 'Mnet ROAS', 'color': '#e377c2'},
        'adv_roas': {'name': 'Advertiser ROAS', 'color': '#7f7f7f'},
        'payout': {'name': 'Payout', 'color': '#bcbd22'},
        'adv_cost': {'name': 'Advertiser Cost', 'color': '#17becf'},
        'max_cost': {'name': 'Max System Cost', 'color': '#ff9896'},
        'actual_payout': {'name': 'Actual Advertiser Payout', 'color': '#c5b0d5'}
    }
    
    if not selected_metrics:
        selected_metrics = ['clicks']
    
    # Normalize metrics to 0-100 scale for better comparison
    normalized_data = daily_agg.copy()
    for metric in selected_metrics:
        if metric in normalized_data.columns and metric in metric_config:
            max_val = normalized_data[metric].max()
            min_val = normalized_data[metric].min()
            if max_val > min_val:
                normalized_data[f'{metric}_norm'] = ((normalized_data[metric] - min_val) / (max_val - min_val)) * 100
            else:
                normalized_data[f'{metric}_norm'] = 50  # midpoint if all values same
    
    for metric in selected_metrics:
        if metric in metric_config and f'{metric}_norm' in normalized_data.columns:
            config = metric_config[metric]
            
            # Create hover text with actual values
            hover_text = [
                f"{config['name']}: {actual:.2f}<br>Date: {date.strftime('%d-%m-%Y')}"
                for actual, date in zip(daily_agg[metric], daily_agg['Day'])
            ]
            
            fig.add_trace(go.Scatter(
                x=normalized_data['Day'],
                y=normalized_data[f'{metric}_norm'],
                name=config['name'],
                mode='lines+markers',
                line=dict(color=config['color'], width=2),
                marker=dict(size=8, color=config['color']),
                hovertext=hover_text,
                hoverinfo='text'
            ))
    
    # Determine title based on campaign selection
    if selected_campaigns:
        if len(selected_campaigns) == 1:
            chart_title = f'Daily Metrics Trend - {selected_campaigns[0]}'
        else:
            chart_title = f'Daily Metrics Trend - {len(selected_campaigns)} Campaigns Selected'
    else:
        chart_title = 'Daily Metrics Trend - Overall'
    
    fig.update_layout(
        title=chart_title,
        plot_bgcolor='#111',
        paper_bgcolor='#111',
        font=dict(color='white'),
        hovermode='x unified',
        height=600,
        xaxis=dict(title='Date', showgrid=True, gridcolor='#333'),
        yaxis=dict(title='Normalized Scale (0-100)', showgrid=True, gridcolor='#333'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(0,0,0,0.5)'
        ),
        annotations=[
            dict(
                text='All metrics normalized to 0-100 scale for comparison. Hover for actual values.',
                xref='paper', yref='paper',
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=10, color='#aaa'),
                xanchor='center'
            )
        ]
    )
    
    return stats_display, campaign_table, fig, campaign_options

# Clientside callback for expand/collapse
clientside_callback(
    """
    function(n_clicks_list) {
        var triggered = dash_clientside.callback_context.triggered;
        if (!triggered || triggered.length === 0) {
            return window.dash_clientside.no_update;
        }
        
        var btn_id = triggered[0].prop_id.split('.')[0];
        var parsed = JSON.parse(btn_id);
        var unique_id = parsed.index;
        
        // Find button and toggle text
        var buttons = document.querySelectorAll('[id*="expand-btn"]');
        buttons.forEach(function(btn) {
            if (btn.id.includes(unique_id)) {
                btn.textContent = btn.textContent === '▶' ? '▼' : '▶';
            }
        });
        
        // Toggle day rows
        var dayRows = document.querySelectorAll('[id*="day-row"]');
        dayRows.forEach(function(row) {
            if (row.id.includes(unique_id)) {
                row.style.display = row.style.display === 'none' ? 'table-row' : 'none';
            }
        });
        
        return window.dash_clientside.no_update;
    }
    """,
    Output({'type': 'expand-btn', 'index': dash.dependencies.ALL}, 'n_clicks'),
    Input({'type': 'expand-btn', 'index': dash.dependencies.ALL}, 'n_clicks'),
    prevent_initial_call=True
)
