import dash
from dash import dcc, html, Input, Output, State, dash_table, callback, register_page, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash.dependencies import MATCH
dash.register_page(__name__, path="/creative", name="Creative & SERP Analysis")

# =========================================================
# DEFINE CONSTANTS FIRST
# =========================================================
# Define attribute columns for Creative & SERP Analysis
CS_ATTRIBUTE_COLUMNS = [
    'bg_color',
    'font_color',
    'cta_present',
    'cta_color',
    'logo_present',
    'web_results_present'
]

# Color mapping for visual display
COLOR_MAP = {
    'Green': '#00ff00',
    'Grey': '#808080',
    'Gray': '#808080',
    'Black': '#000000',
    'White': '#ffffff',
    'Red': '#ff0000',
    'Purple': '#800080',
    'Pink': '#ffc0cb',
    'Dark Blue': '#00008b',
    'Turquoise': '#40e0d0',
    'Turqoise': '#40e0d0',  # Handle typo
    'Yellow': '#ffff00'
}

# Define attribute columns for Ad Title Analysis
ATTRIBUTE_COLUMNS = [
    'specificity',
    'attention_trigger',
    'tone',
    'trust_signal',
    'framing',
    'character_count',
    'word_count',
    'is_number_present'
]

# =========================================================
# LOAD DATA
# =========================================================
# =========================================================
# LOAD DATA
# =========================================================
# =========================================================
# LOAD DATA FROM GOOGLE DRIVE
# =========================================================
import io
import requests

# File 1: Creative & SERP Data
CREATIVE_SERP_FILE_ID = "1mGUAmrQJAxxbiiIAD39LKIKsl1IMWANz"
CREATIVE_SERP_URL = f"https://drive.google.com/uc?export=download&id={CREATIVE_SERP_FILE_ID}"

# File 2: Ad Title Data
AD_TITLE_FILE_ID = "13sSmNN7f2e1FkCji6TVCA9BaGRDfneQO"
AD_TITLE_URL = f"https://drive.google.com/uc?export=download&id={AD_TITLE_FILE_ID}"

# Load Creative & SERP data
try:
    response = requests.get(CREATIVE_SERP_URL)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
except Exception as e:
    print(f"Error loading Creative & SERP data: {e}")
    df = pd.DataFrame()

# Column mapping for Creative & SERP data
COL_MAP = {
    'Advertiser': 'advertiser',
    'Campaign Type': 'campaign_type',
    'Campaign': 'campaign',
    'CM Template': 'creative',
    'CM Serp Template (CM)': 'serp',
    'Background Color': 'bg_color',
    'Main Font Color': 'font_color',
    'CTA Present?': 'cta_present',
    'CTA Color': 'cta_color',
    'Adv Logo Present?': 'logo_present',
    'Web Results Present?': 'web_results_present',
    'Clicks': 'clicks',
    'Ad Impressions': 'impressions',
    'Weighted Conversion': 'conversions',
    'Advertiser Cost': 'adv_cost',
    'Max System Cost': 'max_cost',
    'Advertiser Value': 'adv_value',
    'Actual Advertiser Payout': 'actual_adv_payout',
    'Mnet ROAS': 'mnet_roas',
    'Advertiser ROAS': 'adv_roas'
}
df = df.rename(columns={c: COL_MAP[c] for c in df.columns if c in COL_MAP})

# Load Ad Title data
# Load Ad Title data
AD_TITLE_ERROR = None
AD_TITLE_COLUMNS = []
try:
    response = requests.get(AD_TITLE_URL)
    response.raise_for_status()
    ad_df = pd.read_csv(io.StringIO(response.text), encoding='latin1')    
    AD_TITLE_COLUMNS = list(ad_df.columns)  # Store original columns for debugging
    
    # Column mapping for Ad Title data
    AD_COL_MAP = {
        'Campaign Type': 'campaign_type',
        'Advertiser': 'advertiser',
        'Campaign': 'campaign',
        'Ad Title': 'ad_title',
        'Specificity': 'specificity',
        'Attention Trigger': 'attention_trigger',
        'Tone': 'tone',
        'Trust Signal': 'trust_signal',
        'Framing': 'framing',
        'Character Count': 'character_count',
        'Word Count': 'word_count',
        'Is Number Present?': 'is_number_present',
        'Ad Impressions': 'impressions',
        'Clicks': 'clicks',
        'Weighted Conversion': 'conversions',
        'CVR': 'cvr',
        'CTR': 'ctr',
        'CPA': 'cpa',
        'Advertiser Cost': 'adv_cost',
        'Max System Cost': 'max_cost',
        'Advertiser Value': 'actual_adv_payout',
        'Mnet ROAS': 'mnet_roas',
        'Adv ROAS': 'adv_roas'
    }
    ad_df = ad_df.rename(columns={c: AD_COL_MAP[c] for c in ad_df.columns if c in AD_COL_MAP})
    
    # Convert numeric columns
    for c in ['impressions','clicks','conversions','adv_cost','max_cost','actual_adv_payout']:
        ad_df[c] = pd.to_numeric(ad_df.get(c, 0), errors='coerce').fillna(0)
    
    # Calculate metrics if not present
    if 'ctr' not in ad_df.columns:
        ad_df['ctr'] = np.where(ad_df['impressions']>0, 100*ad_df['clicks']/ad_df['impressions'], np.nan)
    if 'cvr' not in ad_df.columns:
        ad_df['cvr'] = np.where(ad_df['clicks']>0, 100*ad_df['conversions']/ad_df['clicks'], np.nan)
    if 'cpa' not in ad_df.columns:
        ad_df['cpa'] = np.where(ad_df['conversions']>0, ad_df['adv_cost']/ad_df['conversions'], np.nan)
    if 'mnet_roas' not in ad_df.columns:
        ad_df['mnet_roas'] = np.where(ad_df['max_cost']>0, ad_df['actual_adv_payout']/ad_df['max_cost'], np.nan)
    if 'adv_roas' not in ad_df.columns:
        ad_df['adv_roas'] = np.where(ad_df['adv_cost']>0, ad_df['actual_adv_payout']/ad_df['adv_cost'], np.nan)
    
    AD_TITLE_AVAILABLE = True
except Exception as e:
    ad_df = pd.DataFrame()
    AD_TITLE_AVAILABLE = False
    AD_TITLE_ERROR = str(e)

# Add campaign_type if missing
if 'campaign_type' not in df.columns:
    df['campaign_type'] = 'All'

# Replace "null" strings with actual NaN values for attribute columns
for col in CS_ATTRIBUTE_COLUMNS:
    if col in df.columns:
        df[col] = df[col].replace('null', np.nan)

# Convert numeric columns
for c in ['clicks','impressions','conversions','adv_cost','max_cost','adv_value']:
    df[c] = pd.to_numeric(df.get(c, 0), errors='coerce').fillna(0)

# Calculate metrics for Creative & SERP
df['ctr'] = np.where(df['impressions']>0, 100*df['clicks']/df['impressions'], np.nan)
df['cvr'] = np.where(df['clicks']>0, 100*df['conversions']/df['clicks'], np.nan)
df['cpa'] = np.where(df['conversions']>0, df['adv_cost']/df['conversions'], np.nan)

# Convert ROAS columns if they exist, otherwise calculate them
for c in ['mnet_roas', 'adv_roas', 'actual_adv_payout']:
    if c in df.columns:
        df[c] = pd.to_numeric(df.get(c, 0), errors='coerce').fillna(0)

# Calculate ROAS if not present in CSV
if 'mnet_roas' not in df.columns:
    df['mnet_roas'] = np.where(df['max_cost']>0, df['adv_value']/df['max_cost'], 0)

if 'adv_roas' not in df.columns:
    # Check if actual_adv_payout exists, if not use adv_value
    if 'actual_adv_payout' in df.columns:
        df['adv_roas'] = np.where(df['adv_cost']>0, df['actual_adv_payout']/df['adv_cost'], 0)
    else:
        df['adv_roas'] = np.where(df['adv_cost']>0, df['adv_value']/df['adv_cost'], 0)

if 'actual_adv_payout' not in df.columns:
    df['actual_adv_payout'] = df['adv_value']  # Use adv_value as fallback
    
# Pre-compute options
ALL_ADVERTISERS = sorted(df['advertiser'].dropna().unique())
ALL_CAMPAIGN_TYPES = sorted(df['campaign_type'].dropna().unique())
ALL_CAMPAIGNS = sorted(df['campaign'].dropna().unique())
# =========================================================
# TABLE STYLE
# =========================================================
TABLE_STYLE = {
    'style_cell': {
        'textAlign': 'left',
        'backgroundColor': '#222',
        'color': 'white',
        'border': '1px solid #444',
        'fontSize': '12px',
        'padding': '10px',
        'minWidth': '80px',
        'maxWidth': '300px',
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
        'whiteSpace': 'normal'
    },    'style_header': {
        'backgroundColor': '#111',
        'fontWeight': 'bold',
        'border': '1px solid #444',
        'color': '#17a2b8'
    },
    'style_data': {
        'border': '1px solid #444'
    },
    'style_data_conditional': []
}

# =========================================================
# LAYOUT
# =========================================================
layout = dbc.Container(
    fluid=True, 
    style={'backgroundColor': '#111', 'minHeight': '100vh', 'padding': '20px'}, 
    children=[

        html.H2(
            "Campaign Performance Analysis Dashboard",
            style={'color': '#17a2b8', 'marginBottom': '30px', 'textAlign': 'center'}
        ),

        # Filters
        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id='cs_adv_dd',
                multi=True,
                placeholder="Select Advertiser(s)",
                options=[{'label':x,'value':x} for x in ALL_ADVERTISERS],
                style={'color': 'black'}
            ), width=4),
            dbc.Col(dcc.Dropdown(
                id='cs_camp_type_dd',
                multi=True,
                placeholder="Select Campaign Type(s)",
                options=[],
                style={'color': 'black'}
            ), width=4),
            dbc.Col(dcc.Dropdown(
                id='cs_camp_dd',
                multi=True,
                placeholder="Select Campaign(s)",
                options=[{'label':x,'value':x} for x in ALL_CAMPAIGNS],
                style={'color': 'black'}
            ), width=4)
        ], style={'marginBottom': '30px'}),

        html.Hr(style={'borderColor': '#444'}),

        # Tabs
        dcc.Tabs(
            id='main_tabs', 
            value='tab-creative-serp', 
            children=[

                # Tab 1: Creative & SERP Analysis
                dcc.Tab(
                    label='Creative & SERP Analysis',
                    value='tab-creative-serp',
                    style={'backgroundColor': '#222', 'color': '#aaa'},
                    selected_style={'backgroundColor': '#17a2b8', 'color': 'white'},
                    children=[
                        dbc.Row([
                            dbc.Col([
                                html.Div(
                                    style={'padding': '20px'},
                                    children=[
                                        # Aggregated Stats
                                        html.Div(id='cs_agg_stats', style={'marginBottom': '20px'}),

                                        html.Hr(style={'borderColor': '#444'}),

                                        # Creative Tables
                                        html.H4("Creative Performance", style={'color': '#5dade2'}),
                                        html.H5("Best 5 Creatives", style={'color': '#00ff00'}),
                                        dash_table.DataTable(
                                            id='best_creatives',
                                            columns=[
                                                {'name': 'Creative', 'id': 'creative'},
                                                {'name': 'Clicks', 'id': 'clicks'},
                                                {'name': 'Conv', 'id': 'conversions'},
                                                {'name': 'CVR %', 'id': 'cvr'},
                                                {'name': 'CTR %', 'id': 'ctr'},
                                                {'name': 'CPA', 'id': 'cpa'},
                                                {'name': 'Mnet ROAS', 'id': 'mnet_roas'},
                                                {'name': 'Adv ROAS', 'id': 'adv_roas'},
                                                {'name': 'Adv Cost', 'id': 'adv_cost'},
                                                {'name': 'Adv Payout', 'id': 'actual_adv_payout'},
                                                {'name': 'Max Cost', 'id': 'max_cost'}
                                                ],
                                            style_cell=TABLE_STYLE['style_cell'],
                                            style_header=TABLE_STYLE['style_header'],
                                            style_data=TABLE_STYLE['style_data'],
                                            style_data_conditional=[]  # To be updated dynamically via callback
                                        ),
                                        html.H5("Worst 5 Creatives", style={'color': '#ff0000'}),
                                        dash_table.DataTable(
                                            id='worst_creatives',
                                            columns=[
                                                {'name': 'Creative', 'id': 'creative'},
                                                {'name': 'Clicks', 'id': 'clicks'},
                                                {'name': 'Conv', 'id': 'conversions'},
                                                {'name': 'CVR %', 'id': 'cvr'},
                                                {'name': 'CTR %', 'id': 'ctr'},
                                                {'name': 'CPA', 'id': 'cpa'},
                                                {'name': 'Mnet ROAS', 'id': 'mnet_roas'},
                                                {'name': 'Adv ROAS', 'id': 'adv_roas'},
                                                {'name': 'Adv Cost', 'id': 'adv_cost'},
                                                {'name': 'Adv Payout', 'id': 'actual_adv_payout'},
                                                {'name': 'Max Cost', 'id': 'max_cost'}
                                                ],
                                            style_cell=TABLE_STYLE['style_cell'],
                                            style_header=TABLE_STYLE['style_header'],
                                            style_data=TABLE_STYLE['style_data'],
                                            style_data_conditional=[]
                                            ),
                                        html.Hr(style={'borderColor': '#444'}),

                                # SERP Tables
                                        html.H4("SERP Performance", style={'color': '#5dade2', 'marginTop': '20px'}),
                                        html.H5("Best 5 SERPs", style={'color': '#00ff00'}),
                                        dash_table.DataTable(
                                            id='best_serps',
                                            columns=[
                                                {'name': 'SERP', 'id': 'serp'},
                                                {'name': 'Clicks', 'id': 'clicks'},
                                                {'name': 'Conv', 'id': 'conversions'},
                                                {'name': 'CVR %', 'id': 'cvr'},
                                                {'name': 'CTR %', 'id': 'ctr'},
                                                {'name': 'CPA', 'id': 'cpa'},
                                                {'name': 'Mnet ROAS', 'id': 'mnet_roas'},
                                                {'name': 'Adv ROAS', 'id': 'adv_roas'},
                                                {'name': 'Adv Cost', 'id': 'adv_cost'},
                                                {'name': 'Adv Payout', 'id': 'actual_adv_payout'},
                                                {'name': 'Max Cost', 'id': 'max_cost'}
                                                ],
                                            style_cell=TABLE_STYLE['style_cell'],
                                            style_header=TABLE_STYLE['style_header'],
                                            style_data=TABLE_STYLE['style_data'],
                                            style_data_conditional=[]
                                            ),
                                        html.H5("Worst 5 SERPs", style={'color': '#ff0000'}),
                                        dash_table.DataTable(
                                            id='worst_serps',
                                            columns=[
                                                {'name': 'SERP', 'id': 'serp'},
                                                {'name': 'Clicks', 'id': 'clicks'},
                                                {'name': 'Conv', 'id': 'conversions'},
                                                {'name': 'CVR %', 'id': 'cvr'},
                                                {'name': 'CTR %', 'id': 'ctr'},
                                                {'name': 'CPA', 'id': 'cpa'},
                                                {'name': 'Mnet ROAS', 'id': 'mnet_roas'},
                                                {'name': 'Adv ROAS', 'id': 'adv_roas'},
                                                {'name': 'Adv Cost', 'id': 'adv_cost'},
                                                {'name': 'Adv Payout', 'id': 'actual_adv_payout'},
                                                {'name': 'Max Cost', 'id': 'max_cost'}
                                                ],
                                            style_cell=TABLE_STYLE['style_cell'],
                                            style_header=TABLE_STYLE['style_header'],
                                            style_data=TABLE_STYLE['style_data'],
                                            style_data_conditional=[]
                                            ),
                                        html.Hr(style={'borderColor': '#444'}),
                                # Creative-SERP Pair Tables
                                       html.H4("Creative-SERP Pair Performance", style={'color': '#5dade2', 'marginTop': '20px'}),
                                       html.H5("Best 5 Pairs", style={'color': '#00ff00'}),
                                       dash_table.DataTable(
                                           id='best_pairs',
                                           columns=[
                                               {'name': 'Creative', 'id': 'creative'},
                                               {'name': 'SERP', 'id': 'serp'},
                                               {'name': 'Clicks', 'id': 'clicks'},
                                               {'name': 'Conv', 'id': 'conversions'},
                                               {'name': 'CVR %', 'id': 'cvr'},
                                               {'name': 'CTR %', 'id': 'ctr'},
                                               {'name': 'CPA', 'id': 'cpa'},
                                               {'name': 'Mnet ROAS', 'id': 'mnet_roas'},
                                               {'name': 'Adv ROAS', 'id': 'adv_roas'},
                                               {'name': 'Adv Cost', 'id': 'adv_cost'},
                                               {'name': 'Adv Payout', 'id': 'actual_adv_payout'},
                                               {'name': 'Max Cost', 'id': 'max_cost'}
                                               ],
                                           style_cell=TABLE_STYLE['style_cell'],
                                           style_header=TABLE_STYLE['style_header'],
                                           style_data=TABLE_STYLE['style_data'],
                                           style_data_conditional=[]
                                           ),
                                       html.H5("Worst 5 Pairs", style={'color': '#ff0000'}),
                                       dash_table.DataTable(
                                           id='worst_pairs',
                                           columns=[
                                               {'name': 'Creative', 'id': 'creative'},
                                               {'name': 'SERP', 'id': 'serp'},
                                               {'name': 'Clicks', 'id': 'clicks'},
                                               {'name': 'Conv', 'id': 'conversions'},
                                               {'name': 'CVR %', 'id': 'cvr'},
                                               {'name': 'CTR %', 'id': 'ctr'},
                                               {'name': 'CPA', 'id': 'cpa'},
                                               {'name': 'Mnet ROAS', 'id': 'mnet_roas'},
                                               {'name': 'Adv ROAS', 'id': 'adv_roas'},
                                               {'name': 'Adv Cost', 'id': 'adv_cost'},
                                               {'name': 'Adv Payout', 'id': 'actual_adv_payout'},
                                               {'name': 'Max Cost', 'id': 'max_cost'}
                                               ],
                                           style_cell=TABLE_STYLE['style_cell'],
                                           style_header=TABLE_STYLE['style_header'],
                                           style_data=TABLE_STYLE['style_data'],
                                           style_data_conditional=[]
                                    ),
                                        html.Hr(style={'borderColor': '#444'}),
                                        
                                        # Creative & SERP Attributes Section
                                        html.H4("Creative & SERP Attributes Analysis", style={'color': '#5dade2', 'marginTop': '20px'}),
                                        html.Div(id='cs_attributes_content')
                                    ]
                                )
                            ])
                        ])
                    ]
                ),

                # Tab 2: Ad Title Analysis
                dcc.Tab(
                    label='Ad Title Analysis',
                    value='tab-ad-title',
                    style={'backgroundColor': '#222', 'color': '#aaa'},
                    selected_style={'backgroundColor': '#17a2b8', 'color': 'white'},
                    children=[html.Div(id='ad_title_content', style={'padding': '20px'})]
                )
            ]
        )
        ]
    )


# =========================================================
# CALLBACKS
# =========================================================

# Update filter dropdowns
@callback(
    [Output('cs_camp_type_dd', 'options'),
     Output('cs_camp_dd', 'options')],
    [Input('cs_adv_dd', 'value'),
     Input('cs_camp_type_dd', 'value')]
)
def update_filters(advs, camp_types):
    d = df.copy()
    if advs:
        d = d[d['advertiser'].isin(advs)]
    campaign_types = sorted(d['campaign_type'].dropna().unique())
    
    if camp_types:
        d = d[d['campaign_type'].isin(camp_types)]
    campaigns = sorted(d['campaign'].dropna().unique())
    
    return (
        [{'label': x, 'value': x} for x in campaign_types],
        [{'label': x, 'value': x} for x in campaigns]
    )

# Update Creative & SERP Analysis
@callback(
    [Output('cs_agg_stats', 'children'),
     Output('best_creatives', 'data'),
     Output('best_creatives', 'style_data_conditional'),
     Output('worst_creatives', 'data'),
     Output('worst_creatives', 'style_data_conditional'),
     Output('best_serps', 'data'),
     Output('best_serps', 'style_data_conditional'),
     Output('worst_serps', 'data'),
     Output('worst_serps', 'style_data_conditional'),
     Output('best_pairs', 'data'),
     Output('best_pairs', 'style_data_conditional'),
     Output('worst_pairs', 'data'),
     Output('worst_pairs', 'style_data_conditional')],
    [Input('cs_adv_dd', 'value'),
     Input('cs_camp_type_dd', 'value'),
     Input('cs_camp_dd', 'value')]
)
def update_creative_serp(advs, camp_types, camps):
    # Filter data
    d = df.copy()
    if advs:
        d = d[d['advertiser'].isin(advs)]
    if camp_types:
        d = d[d['campaign_type'].isin(camp_types)]
    if camps:
        d = d[d['campaign'].isin(camps)]
    
    avg_cvr = d['cvr'].mean()

    # Calculate aggregated stats
    total_clicks = d['clicks'].sum()
    total_impressions = d['impressions'].sum()
    total_conversions = d['conversions'].sum()
    total_adv_cost = d['adv_cost'].sum()
    total_max_cost = d['max_cost'].sum()
    total_adv_value = d['adv_value'].sum()
    
    agg_cvr = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
    agg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
    agg_cpa = (total_adv_cost / total_conversions) if total_conversions > 0 else 0
    agg_roas = (total_adv_value / total_max_cost) if total_max_cost > 0 else 0
    total_actual_adv_payout = d['actual_adv_payout'].sum()
    agg_mnet_roas = (total_actual_adv_payout / total_max_cost) if total_max_cost > 0 else 0
    agg_adv_roas = (total_actual_adv_payout / total_adv_cost) if total_adv_cost > 0 else 0
    
    stats_display = dbc.Card([
        dbc.CardBody([
            html.H4("Aggregated Stats", style={'color': '#17a2b8', 'marginBottom': '15px'}),
            dbc.Row([
                dbc.Col([html.Strong("Clicks: ", style={'color': '#aaa'}), 
                        html.Span(f"{int(total_clicks):,}", style={'color': '#5dade2', 'fontSize': '18px'})], width=2),
                dbc.Col([html.Strong("Conversions: ", style={'color': '#aaa'}), 
                        html.Span(f"{total_conversions:.2f}", style={'color': '#5dade2', 'fontSize': '18px'})], width=2),
                dbc.Col([html.Strong("CVR: ", style={'color': '#aaa'}), 
                        html.Span(f"{agg_cvr:.2f}%", style={'color': '#00ff00', 'fontSize': '18px'})], width=2),
                dbc.Col([html.Strong("CTR: ", style={'color': '#aaa'}), 
                        html.Span(f"{agg_ctr:.2f}%", style={'color': '#00ff00', 'fontSize': '18px'})], width=2),
                dbc.Col([html.Strong("CPA: ", style={'color': '#aaa'}), 
                        html.Span(f"${agg_cpa:.2f}", style={'color': '#ffcc00', 'fontSize': '18px'})], width=2),
                dbc.Col([html.Strong("ROAS: ", style={'color': '#aaa'}), 
                        html.Span(f"{agg_roas:.2f}", style={'color': '#00ff00', 'fontSize': '18px'})], width=2)
            ])
        ])
    ], style={'backgroundColor': '#222', 'border': '1px solid #444'})
    # Helper function for color coding
    def add_color_conditional(data_list, agg_cvr, agg_ctr, agg_mnet_roas, agg_adv_roas, agg_cpa):
        """Add style_data_conditional for metrics vs aggregates"""
        if not data_list:
            return data_list
        
        conditional = [
            # CVR - green if above average, red if below
            {'if': {'filter_query': f'{{cvr}} > {agg_cvr}', 'column_id': 'cvr'}, 
             'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{cvr}} <= {agg_cvr}', 'column_id': 'cvr'}, 
             'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
            
            # CTR - green if above average, red if below
            {'if': {'filter_query': f'{{ctr}} > {agg_ctr}', 'column_id': 'ctr'}, 
             'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{ctr}} <= {agg_ctr}', 'column_id': 'ctr'}, 
             'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
            
            # CPA - green if BELOW average (lower is better), red if above
            {'if': {'filter_query': f'{{cpa}} < {agg_cpa}', 'column_id': 'cpa'}, 
             'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{cpa}} >= {agg_cpa}', 'column_id': 'cpa'}, 
             'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
            
            # Mnet ROAS - green if above average, red if below
            {'if': {'filter_query': f'{{mnet_roas}} > {agg_mnet_roas}', 'column_id': 'mnet_roas'}, 
             'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{mnet_roas}} <= {agg_mnet_roas}', 'column_id': 'mnet_roas'}, 
             'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
            
            # Adv ROAS - green if above average, red if below
            {'if': {'filter_query': f'{{adv_roas}} > {agg_adv_roas}', 'column_id': 'adv_roas'}, 
             'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{adv_roas}} <= {agg_adv_roas}', 'column_id': 'adv_roas'}, 
             'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'}
        ]
        
        return data_list, conditional
    # Helper function for scoring
    def get_best_worst(d, group_col, min_clicks=3):
        g = d.groupby(group_col, dropna=True).agg(
            clicks=('clicks','sum'),
            impressions=('impressions','sum'),
            conversions=('conversions','sum'),
            adv_cost=('adv_cost','sum'),
            max_cost=('max_cost','sum'),
            adv_value=('adv_value','sum'),
            actual_adv_payout=('actual_adv_payout','sum'),  # ADD
            mnet_roas=('mnet_roas','mean'),  # ADD - use mean for ROAS
            adv_roas=('adv_roas','mean')  # ADD
            ).reset_index()
        
        g['ctr'] = np.where(g['impressions']>0, 100*g['clicks']/g['impressions'], np.nan)
        g['cvr'] = np.where(g['clicks']>0, 100*g['conversions']/g['clicks'], np.nan)
        g['cpa'] = np.where(g['conversions']>0, g['adv_cost']/g['conversions'], np.nan)
        
        g = g.dropna(subset=['cvr'])
        
        # BEST
        best_candidates = g[(g['clicks'] >= min_clicks) & (g['cvr'] > 0)].copy()
        if len(best_candidates) > 0:
            best_candidates['score'] = best_candidates['cvr'] * np.log1p(best_candidates['clicks'])
            best_df = best_candidates.sort_values('score', ascending=False).head(5)
            best_ids = set(best_df[group_col].tolist())
        else:
            best_df = pd.DataFrame()
            best_ids = set()
        
        # WORST
        worst_candidates = g[
            (g['clicks'] >= min_clicks) & 
            (g['cvr'] <= 0.6) &
            (~g[group_col].isin(best_ids))
        ].copy()
        if len(worst_candidates) > 0:
            worst_df = worst_candidates.sort_values('clicks', ascending=False).head(5)
        else:
            worst_df = pd.DataFrame()
        
        return best_df.round(2).to_dict('records'), worst_df.round(2).to_dict('records')
    
    # Get results
    best_creatives, worst_creatives = get_best_worst(d, 'creative', min_clicks=3)
    best_serps, worst_serps = get_best_worst(d, 'serp', min_clicks=3)
    
    # For pairs
    d['pair'] = d['creative'].astype(str) + ' | ' + d['serp'].astype(str)
    best_pairs_raw, worst_pairs_raw = get_best_worst(d, 'pair', min_clicks=3)
    
    # Split pairs back
    def split_pairs(data):
        for item in data:
            if 'pair' in item:
                parts = item['pair'].split(' | ')
                item['creative'] = parts[0]
                item['serp'] = parts[1] if len(parts) > 1 else ''
                del item['pair']
        return data
    
    best_pairs = split_pairs(best_pairs_raw)
    worst_pairs = split_pairs(worst_pairs_raw)
    
    # Apply color coding to all tables
    best_creatives, bc_conditional = add_color_conditional(best_creatives, agg_cvr, agg_ctr, agg_mnet_roas, agg_adv_roas, agg_cpa)
    worst_creatives, wc_conditional = add_color_conditional(worst_creatives, agg_cvr, agg_ctr, agg_mnet_roas, agg_adv_roas, agg_cpa)
    best_serps, bs_conditional = add_color_conditional(best_serps, agg_cvr, agg_ctr, agg_mnet_roas, agg_adv_roas, agg_cpa)
    worst_serps, ws_conditional = add_color_conditional(worst_serps, agg_cvr, agg_ctr, agg_mnet_roas, agg_adv_roas, agg_cpa)
    best_pairs, bp_conditional = add_color_conditional(best_pairs, agg_cvr, agg_ctr, agg_mnet_roas, agg_adv_roas, agg_cpa)
    worst_pairs, wp_conditional = add_color_conditional(worst_pairs, agg_cvr, agg_ctr, agg_mnet_roas, agg_adv_roas, agg_cpa)
    return (stats_display, 
            best_creatives, bc_conditional,
            worst_creatives, wc_conditional,
            best_serps, bs_conditional,
            worst_serps, ws_conditional,
            best_pairs, bp_conditional,
            worst_pairs, wp_conditional)

# Update Ad Title Analysis Content
# Update Creative & SERP Attributes Analysis
@callback(
    Output('cs_attributes_content', 'children'),
    [Input('cs_adv_dd', 'value'),
     Input('cs_camp_type_dd', 'value'),
     Input('cs_camp_dd', 'value')]
)
def update_cs_attributes(advs, camp_types, camps):
    # Filter data
    d = df.copy()
    if advs:
        d = d[d['advertiser'].isin(advs)]
    if camp_types:
        d = d[d['campaign_type'].isin(camp_types)]
    if camps:
        d = d[d['campaign'].isin(camps)]
    
    if len(d) == 0:
        return html.Div([
            html.P("No data available for selected filters.", style={'color': '#aaa'})
        ])
    
    # Calculate overall averages for conditional formatting (with NaN handling)
    overall_avg_cvr = d['cvr'].mean() if not d['cvr'].isna().all() else 0
    overall_avg_ctr = d['ctr'].mean() if not d['ctr'].isna().all() else 0
    overall_avg_cpa = d['cpa'].mean() if not d['cpa'].isna().all() else 0
    overall_avg_mnet_roas = d['mnet_roas'].mean() if not d['mnet_roas'].isna().all() else 0
    overall_avg_adv_roas = d['adv_roas'].mean() if not d['adv_roas'].isna().all() else 0
    
    # Create sections for each attribute
    attribute_sections = []
    
    for attr in CS_ATTRIBUTE_COLUMNS:
        # Check if column exists and has data
        if attr not in d.columns:
            continue
        
        # Check if column has any non-null values
        if d[attr].isna().all() or len(d[attr].dropna()) == 0:
            continue
        
        attr_display_name = attr.replace('_', ' ').title()
        
        # Aggregate by attribute
        agg_data = d.groupby(attr, dropna=True).agg(
            impressions=('impressions', 'sum'),
            clicks=('clicks', 'sum'),
            conversions=('conversions', 'sum'),
            adv_cost=('adv_cost', 'sum'),
            max_cost=('max_cost', 'sum'),
            actual_adv_payout=('actual_adv_payout', 'sum')
        ).reset_index()
        
        # Skip if no data after grouping
        if len(agg_data) == 0:
            continue
        
        # Calculate metrics
        agg_data['ctr'] = np.where(agg_data['impressions']>0, 100*agg_data['clicks']/agg_data['impressions'], 0)
        agg_data['cvr'] = np.where(agg_data['clicks']>0, 100*agg_data['conversions']/agg_data['clicks'], 0)
        agg_data['cpa'] = np.where(agg_data['conversions']>0, agg_data['adv_cost']/agg_data['conversions'], 0)
        agg_data['mnet_roas'] = np.where(agg_data['max_cost']>0, agg_data['actual_adv_payout']/agg_data['max_cost'], 0)
        agg_data['adv_roas'] = np.where(agg_data['adv_cost']>0, agg_data['actual_adv_payout']/agg_data['adv_cost'], 0)
        
        # Round numbers
        agg_data = agg_data.round(2)
        
        # Create conditional styling for metrics
        style_conditional = [
            # CVR - green if ABOVE average, red if BELOW
            {'if': {'filter_query': f'{{cvr}} > {overall_avg_cvr:.2f}', 'column_id': 'cvr'}, 
             'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{cvr}} <= {overall_avg_cvr:.2f}', 'column_id': 'cvr'}, 
             'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
            
            # CTR - green if ABOVE average, red if BELOW
            {'if': {'filter_query': f'{{ctr}} > {overall_avg_ctr:.2f}', 'column_id': 'ctr'}, 
             'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{ctr}} <= {overall_avg_ctr:.2f}', 'column_id': 'ctr'}, 
             'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
            
            # CPA - green if BELOW average (lower is better), red if ABOVE
            {'if': {'filter_query': f'{{cpa}} < {overall_avg_cpa:.2f}', 'column_id': 'cpa'}, 
             'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{cpa}} >= {overall_avg_cpa:.2f}', 'column_id': 'cpa'}, 
             'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
            
            # Mnet ROAS - green if ABOVE average, red if BELOW
            {'if': {'filter_query': f'{{mnet_roas}} > {overall_avg_mnet_roas:.2f}', 'column_id': 'mnet_roas'}, 
             'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{mnet_roas}} <= {overall_avg_mnet_roas:.2f}', 'column_id': 'mnet_roas'}, 
             'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
            
            # Adv ROAS - green if ABOVE average, red if BELOW
            {'if': {'filter_query': f'{{adv_roas}} > {overall_avg_adv_roas:.2f}', 'column_id': 'adv_roas'}, 
             'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{adv_roas}} <= {overall_avg_adv_roas:.2f}', 'column_id': 'adv_roas'}, 
             'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
            
            # Make attribute column clickable-looking
            {'if': {'column_id': attr}, 
             'cursor': 'pointer', 'fontWeight': 'bold'}
        ]
        
        # Add color highlighting for color columns (KEEP THE ACTUAL COLORS!)
        if attr in ['bg_color', 'font_color', 'cta_color']:
            for idx, row in agg_data.iterrows():
                color_value = row[attr]
                if pd.notna(color_value) and color_value in COLOR_MAP:
                    bg = COLOR_MAP[color_value]
                    txt = 'white' if color_value in ['Black', 'Dark Blue', 'Purple', 'Red'] else 'black'
                    style_conditional.append({
                        'if': {'filter_query': f'{{{attr}}} = "{color_value}"', 'column_id': attr},
                        'backgroundColor': bg,
                        'color': txt,
                        'fontWeight': 'bold'
                    })
        
        # Define columns (NO arrow column)
        columns = [
            {'name': attr_display_name + ' ▼', 'id': attr},  # Arrow in header
            {'name': 'Impressions', 'id': 'impressions'},
            {'name': 'Clicks', 'id': 'clicks'},
            {'name': 'CTR %', 'id': 'ctr'},
            {'name': 'CVR %', 'id': 'cvr'},
            {'name': 'CPA', 'id': 'cpa'},
            {'name': 'Mnet ROAS', 'id': 'mnet_roas'},
            {'name': 'Adv ROAS', 'id': 'adv_roas'}
        ]
        
        attribute_sections.append(
            html.Div([
                html.H5(f"{attr_display_name} Performance", 
                        style={'color': '#ffcc00', 'marginTop': '25px', 'marginBottom': '10px'}),
                html.P(f"Overall Averages: CVR: {overall_avg_cvr:.2f}%, CTR: {overall_avg_ctr:.2f}%, CPA: ${overall_avg_cpa:.2f}, Mnet ROAS: {overall_avg_mnet_roas:.2f}, Adv ROAS: {overall_avg_adv_roas:.2f}",
                       style={'color': '#888', 'fontSize': '11px', 'marginBottom': '10px'}),
                dash_table.DataTable(
                    id={'type': 'cs-attr-table', 'index': attr},
                    columns=columns,
                    data=agg_data.to_dict('records'),
                    style_cell={
                        'textAlign': 'left',
                        'backgroundColor': '#222',
                        'color': 'white',
                        'border': '1px solid #444',
                        'fontSize': '12px',
                        'padding': '10px',
                        'minWidth': '60px',
                        'maxWidth': '200px',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis'
                    },
                    style_header={
                        'backgroundColor': '#111',
                        'fontWeight': 'bold',
                        'border': '1px solid #444',
                        'color': '#17a2b8'
                    },
                    style_data_conditional=style_conditional,
                    row_selectable='single',
                    selected_rows=[]
                ),
                html.Div(id={'type': 'cs-drilldown-container', 'index': attr}, 
                        style={'marginTop': '10px'})
            ])
        )
    
    # If no attributes have data, show message
    if len(attribute_sections) == 0:
        return html.Div([
            html.P("No attribute data available.", 
                   style={'color': '#ffcc00', 'backgroundColor': '#331100', 'padding': '15px', 'borderRadius': '5px'})
        ])
    
    return html.Div(attribute_sections)
# Drill-down callback for Creative & SERP attributes
@callback(
    Output({'type': 'cs-drilldown-container', 'index': MATCH}, 'children'),
    Input({'type': 'cs-attr-table', 'index': MATCH}, 'selected_rows'),
    State({'type': 'cs-attr-table', 'index': MATCH}, 'data'),
    State({'type': 'cs-attr-table', 'index': MATCH}, 'id'),
    [State('cs_adv_dd', 'value'),
     State('cs_camp_type_dd', 'value'),
     State('cs_camp_dd', 'value')],
    prevent_initial_call=True
)
def update_cs_drilldown(selected_rows, table_data, table_id, advs, camp_types, camps):
    if not selected_rows:
        return html.Div()
    
    attr = table_id['index']
    
    # Filter data
    d = df.copy()
    if advs:
        d = d[d['advertiser'].isin(advs)]
    if camp_types:
        d = d[d['campaign_type'].isin(camp_types)]
    if camps:
        d = d[d['campaign'].isin(camps)]
    
    # Get selected value and its averages
    selected_value = table_data[selected_rows[0]][attr]
    avg_cvr = table_data[selected_rows[0]]['cvr']
    avg_ctr = table_data[selected_rows[0]]['ctr']
    avg_cpa = table_data[selected_rows[0]]['cpa']
    avg_mnet_roas = table_data[selected_rows[0]]['mnet_roas']
    avg_adv_roas = table_data[selected_rows[0]]['adv_roas']
    
    # Filter for this attribute value
    detail_data = d[d[attr] == selected_value].copy()
    
    # Create Creative-SERP pairs
    detail_data['pair'] = detail_data['creative'].astype(str) + ' | ' + detail_data['serp'].astype(str)
    
    # Aggregate by pair
    pair_agg = detail_data.groupby('pair', dropna=True).agg(
        impressions=('impressions', 'sum'),
        clicks=('clicks', 'sum'),
        conversions=('conversions', 'sum'),
        adv_cost=('adv_cost', 'sum'),
        max_cost=('max_cost', 'sum'),
        actual_adv_payout=('actual_adv_payout', 'sum')
    ).reset_index()
    
    # Calculate metrics
    pair_agg['ctr'] = np.where(pair_agg['impressions']>0, 100*pair_agg['clicks']/pair_agg['impressions'], 0)
    pair_agg['cvr'] = np.where(pair_agg['clicks']>0, 100*pair_agg['conversions']/pair_agg['clicks'], 0)
    pair_agg['cpa'] = np.where(pair_agg['conversions']>0, pair_agg['adv_cost']/pair_agg['conversions'], 0)
    pair_agg['mnet_roas'] = np.where(pair_agg['max_cost']>0, pair_agg['actual_adv_payout']/pair_agg['max_cost'], 0)
    pair_agg['adv_roas'] = np.where(pair_agg['adv_cost']>0, pair_agg['actual_adv_payout']/pair_agg['adv_cost'], 0)
    
    # Split pair back to creative and serp
    pair_agg[['creative', 'serp']] = pair_agg['pair'].str.split(' | ', expand=True)
    pair_agg = pair_agg.drop('pair', axis=1)
    
    # Sort by conversions
    pair_agg = pair_agg.sort_values('conversions', ascending=False)
    
    # Round
    pair_agg = pair_agg.round(2)
    
    # Create conditional styling
    style_conditional = [
        {'if': {'filter_query': f'{{cvr}} > {avg_cvr}', 'column_id': 'cvr'}, 
         'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
        {'if': {'filter_query': f'{{cvr}} <= {avg_cvr}', 'column_id': 'cvr'}, 
         'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
        {'if': {'filter_query': f'{{ctr}} > {avg_ctr}', 'column_id': 'ctr'}, 
         'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
        {'if': {'filter_query': f'{{ctr}} <= {avg_ctr}', 'column_id': 'ctr'}, 
         'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
        {'if': {'filter_query': f'{{cpa}} < {avg_cpa}', 'column_id': 'cpa'}, 
         'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
        {'if': {'filter_query': f'{{cpa}} >= {avg_cpa}', 'column_id': 'cpa'}, 
         'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
        {'if': {'filter_query': f'{{mnet_roas}} > {avg_mnet_roas}', 'column_id': 'mnet_roas'}, 
         'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
        {'if': {'filter_query': f'{{mnet_roas}} <= {avg_mnet_roas}', 'column_id': 'mnet_roas'}, 
         'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
        {'if': {'filter_query': f'{{adv_roas}} > {avg_adv_roas}', 'column_id': 'adv_roas'}, 
         'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
        {'if': {'filter_query': f'{{adv_roas}} <= {avg_adv_roas}', 'column_id': 'adv_roas'}, 
         'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'}
    ]
    
    attr_display = attr.replace('_', ' ').title()
    
    return html.Div([
        dbc.Button("▲ Collapse", 
                   id={'type': 'cs-collapse-btn', 'index': attr}, 
                   color="info", 
                   size="sm", 
                   style={'marginBottom': '10px'}),
        html.H6(f"Creative-SERP Pairs for {attr_display}: {selected_value}", 
                style={'color': '#17a2b8', 'marginBottom': '10px'}),
        html.P([
            f"Category Averages: ",
            html.Span(f"CVR: {avg_cvr:.2f}% ", style={'color': '#ffcc00'}),
            html.Span(f"CTR: {avg_ctr:.2f}% ", style={'color': '#ffcc00'}),
            html.Span(f"CPA: ${avg_cpa:.2f} ", style={'color': '#ffcc00'}),
            html.Span(f"Mnet ROAS: {avg_mnet_roas:.2f} ", style={'color': '#ffcc00'}),
            html.Span(f"Adv ROAS: {avg_adv_roas:.2f}", style={'color': '#ffcc00'})
        ], style={'color': '#aaa', 'fontSize': '12px', 'marginBottom': '10px'}),
        dash_table.DataTable(
            columns=[
                {'name': 'Creative', 'id': 'creative'},
                {'name': 'SERP', 'id': 'serp'},
                {'name': 'Impressions', 'id': 'impressions'},
                {'name': 'Clicks', 'id': 'clicks'},
                {'name': 'Conversions', 'id': 'conversions'},
                {'name': 'CTR %', 'id': 'ctr'},
                {'name': 'CVR %', 'id': 'cvr'},
                {'name': 'CPA', 'id': 'cpa'},
                {'name': 'Mnet ROAS', 'id': 'mnet_roas'},
                {'name': 'Adv ROAS', 'id': 'adv_roas'}
            ],
            data=pair_agg.to_dict('records'),
            style_cell={
                'textAlign': 'left',
                'backgroundColor': '#1a1a1a',
                'color': 'white',
                'border': '1px solid #555',
                'fontSize': '11px',
                'padding': '8px',
                'minWidth': '70px',
                'maxWidth': '300px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis'
            },
            style_header={
                'backgroundColor': '#0d0d0d',
                'fontWeight': 'bold',
                'border': '1px solid #555',
                'color': '#17a2b8'
            },
            style_data_conditional=style_conditional,
            page_size=15
        )
    ], style={
        'backgroundColor': '#0a0a0a', 
        'padding': '15px', 
        'borderRadius': '5px', 
        'border': '2px solid #17a2b8',
        'marginLeft': '20px'
    })


# Collapse callback for Creative & SERP attributes
@callback(
    Output({'type': 'cs-attr-table', 'index': MATCH}, 'selected_rows'),
    Input({'type': 'cs-collapse-btn', 'index': MATCH}, 'n_clicks'),
    prevent_initial_call=True
)
def collapse_cs_drilldown(n_clicks):
    return []

@callback(
    Output('ad_title_content', 'children'),
    [Input('cs_adv_dd', 'value'),
     Input('cs_camp_type_dd', 'value'),
     Input('cs_camp_dd', 'value')]
)
def update_ad_title_content(advs, camp_types, camps):
    if not AD_TITLE_AVAILABLE:
        error_msg = f"Error loading Ad Title data: {AD_TITLE_ERROR}" if AD_TITLE_ERROR else "Ad Title data file not found."
        
        debug_info = []
        if AD_TITLE_COLUMNS:
            debug_info = [
                html.H5("Columns found in file:", style={'color': '#17a2b8', 'marginTop': '15px'}),
                html.Ul([html.Li(col, style={'color': '#aaa', 'fontSize': '12px'}) for col in AD_TITLE_COLUMNS])
            ]
        
        return html.Div([
            html.H4("Ad Title Analysis", style={'color': '#ff0000'}),
            html.P(f"Google Drive File ID: {AD_TITLE_FILE_ID}", style={'color': '#aaa'}),
            html.P(error_msg, style={'color': '#ffcc00', 'backgroundColor': '#331100', 'padding': '10px', 'borderRadius': '5px'}),
            html.P("Please check:", style={'color': '#aaa', 'marginTop': '10px'}),
            html.Ul([
                html.Li("File exists in the same directory as your script", style={'color': '#aaa'}),
                html.Li("File name matches exactly (case-sensitive)", style={'color': '#aaa'}),
                html.Li("File has the correct column names", style={'color': '#aaa'})
            ]),
            *debug_info
        ])
    
    # Filter data
    d = ad_df.copy()
    if advs:
        d = d[d['advertiser'].isin(advs)]
    if camp_types:
        d = d[d['campaign_type'].isin(camp_types)]
    if camps:
        d = d[d['campaign'].isin(camps)]
    
    if len(d) == 0:
        return html.Div([
            html.H4("Ad Title Analysis", style={'color': '#ff0000'}),
            html.P("No data available for selected filters.", style={'color': '#aaa'})
        ])
    
    # Calculate overall average CVR
    overall_avg_cvr = d['cvr'].mean()
    
    # Create tables for each attribute
    attribute_sections = []
    
    for attr in ATTRIBUTE_COLUMNS:
        if attr not in d.columns:
            continue
        
        attr_display_name = attr.replace('_', ' ').title()
        
        # Aggregate by attribute
        agg_data = d.groupby(attr, dropna=True).agg(
            impressions=('impressions', 'sum'),
            clicks=('clicks', 'sum'),
            conversions=('conversions', 'sum'),
            adv_cost=('adv_cost', 'sum'),
            max_cost=('max_cost', 'sum'),
            actual_adv_payout=('actual_adv_payout', 'sum')
        ).reset_index()
        
        # Calculate metrics
        agg_data['ctr'] = np.where(agg_data['impressions']>0, 100*agg_data['clicks']/agg_data['impressions'], 0)
        agg_data['cvr'] = np.where(agg_data['clicks']>0, 100*agg_data['conversions']/agg_data['clicks'], 0)
        agg_data['cpa'] = np.where(agg_data['conversions']>0, agg_data['adv_cost']/agg_data['conversions'], 0)
        agg_data['mnet_roas'] = np.where(agg_data['max_cost']>0, agg_data['actual_adv_payout']/agg_data['max_cost'], 0)
        agg_data['adv_roas'] = np.where(agg_data['adv_cost']>0, agg_data['actual_adv_payout']/agg_data['adv_cost'], 0)
        
        # Format numbers
        agg_data['impressions'] = agg_data['impressions'].apply(lambda x: f"{int(x):,}")
        agg_data['clicks'] = agg_data['clicks'].apply(lambda x: f"{int(x):,}")
        agg_data['ctr'] = agg_data['ctr'].apply(lambda x: f"{x:.2f}%")
        agg_data['cvr'] = agg_data['cvr'].apply(lambda x: f"{x:.2f}%")
        agg_data['cpa'] = agg_data['cpa'].apply(lambda x: f"${x:.2f}")
        agg_data['mnet_roas'] = agg_data['mnet_roas'].apply(lambda x: f"{x:.2f}")
        agg_data['adv_roas'] = agg_data['adv_roas'].apply(lambda x: f"{x:.2f}")
        agg_data['adv_cost'] = agg_data['adv_cost'].apply(lambda x: f"${x:.2f}")
        agg_data['actual_adv_payout'] = agg_data['actual_adv_payout'].apply(lambda x: f"${x:.2f}")
        agg_data['max_cost'] = agg_data['max_cost'].apply(lambda x: f"${x:.2f}")
        
        # Create table
        table_id = f'table_{attr}'
        drill_down_id = f'drilldown_{attr}'
        
        attribute_sections.append(
            html.Div([
                html.H5(f"{attr_display_name} Performance", style={'color': '#5dade2', 'marginTop': '30px'}),
                dash_table.DataTable(
                    id={'type': 'attr-table', 'index': attr},
                    columns=[
                        {'name': attr_display_name, 'id': attr},
                        {'name': 'Impressions', 'id': 'impressions'},
                        {'name': 'Clicks', 'id': 'clicks'},
                        {'name': 'CTR', 'id': 'ctr'},
                        {'name': 'CVR', 'id': 'cvr'},
                        {'name': 'CPA', 'id': 'cpa'},
                        {'name': 'Mnet ROAS', 'id': 'mnet_roas'},
                        {'name': 'Adv ROAS', 'id': 'adv_roas'},
                        {'name': 'Advertiser Cost', 'id': 'adv_cost'},
                        {'name': 'Actual Adv Payout', 'id': 'actual_adv_payout'},
                        {'name': 'Max System Cost', 'id': 'max_cost'}
                        ],
                    data=agg_data.to_dict('records'),
                    style_cell={
                        'textAlign': 'left',
                        'backgroundColor': '#222',
                        'color': 'white',
                        'border': '1px solid #444',
                        'fontSize': '11px',
                        'padding': '8px',
                        'minWidth': '80px',
                        'maxWidth': '200px',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis'
                        },
                    style_header={
                        'backgroundColor': '#111',
                        'fontWeight': 'bold',
                        'border': '1px solid #444',
                        'color': '#17a2b8'
                        },
                    style_data_conditional=[],
                    row_selectable='single',
                    selected_rows=[],
                    hidden_columns=[]
                    ),
                html.Div(id={'type': 'drilldown-container', 'index': attr})
            ])
        )
    
    return html.Div([
        html.H3("Ad Title Performance Analysis", style={'color': '#17a2b8', 'marginBottom': '20px'}),
        html.P(f"Overall Average CVR: {overall_avg_cvr:.2f}%", 
               style={'color': '#00ff00', 'fontSize': '16px', 'fontWeight': 'bold'}),
        html.Hr(style={'borderColor': '#444'}),
        *attribute_sections
    ])

# Dynamic drill-down callbacks for each attribute

@callback(
    Output({'type': 'drilldown-container', 'index': MATCH}, 'children'),
    Input({'type': 'attr-table', 'index': MATCH}, 'selected_rows'),
    State({'type': 'attr-table', 'index': MATCH}, 'data'),
    State({'type': 'attr-table', 'index': MATCH}, 'id'),
    [State('cs_adv_dd', 'value'),
     State('cs_camp_type_dd', 'value'),
     State('cs_camp_dd', 'value')],
    prevent_initial_call=True
)
def update_drilldown_expand(selected_rows, table_data, table_id, advs, camp_types, camps):
    if not selected_rows or not AD_TITLE_AVAILABLE:
        return html.Div()
    
    attr = table_id['index']
    
    # Filter data
    d = ad_df.copy()
    if advs:
        d = d[d['advertiser'].isin(advs)]
    if camp_types:
        d = d[d['campaign_type'].isin(camp_types)]
    if camps:
        d = d[d['campaign'].isin(camps)]
    
    # Get selected value and its average
    selected_value = table_data[selected_rows[0]][attr]
    avg_cvr = float(table_data[selected_rows[0]]['cvr'].replace('%', ''))
    avg_ctr = float(table_data[selected_rows[0]]['ctr'].replace('%', ''))
    avg_cpa = float(table_data[selected_rows[0]]['cpa'].replace('$', ''))
    avg_mnet_roas = float(table_data[selected_rows[0]]['mnet_roas'])
    avg_adv_roas = float(table_data[selected_rows[0]]['adv_roas'])
    
    # Filter for this attribute value
    detail_data = d[d[attr] == selected_value].copy()
    
    # Aggregate by ad_title
    title_agg = detail_data.groupby('ad_title', dropna=True).agg(
        impressions=('impressions', 'sum'),
        clicks=('clicks', 'sum'),
        conversions=('conversions', 'sum'),
        adv_cost=('adv_cost', 'sum'),
        max_cost=('max_cost', 'sum'),
        actual_adv_payout=('actual_adv_payout', 'sum')
    ).reset_index()
    
    # Calculate metrics
    title_agg['ctr'] = np.where(title_agg['impressions']>0, 100*title_agg['clicks']/title_agg['impressions'], 0)
    title_agg['cvr'] = np.where(title_agg['clicks']>0, 100*title_agg['conversions']/title_agg['clicks'], 0)
    title_agg['cpa'] = np.where(title_agg['conversions']>0, title_agg['adv_cost']/title_agg['conversions'], 0)
    title_agg['mnet_roas'] = np.where(title_agg['max_cost']>0, title_agg['actual_adv_payout']/title_agg['max_cost'], 0)
    title_agg['adv_roas'] = np.where(title_agg['adv_cost']>0, title_agg['actual_adv_payout']/title_agg['adv_cost'], 0)
    
    # Create conditional styling
    style_conditional = [
        {'if': {'filter_query': f'{{cvr}} > {avg_cvr}', 'column_id': 'cvr'}, 
         'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
        {'if': {'filter_query': f'{{cvr}} <= {avg_cvr}', 'column_id': 'cvr'}, 
         'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
        {'if': {'filter_query': f'{{ctr}} > {avg_ctr}', 'column_id': 'ctr'}, 
         'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
        {'if': {'filter_query': f'{{ctr}} <= {avg_ctr}', 'column_id': 'ctr'}, 
         'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
        {'if': {'filter_query': f'{{cpa}} < {avg_cpa}', 'column_id': 'cpa'}, 
         'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
        {'if': {'filter_query': f'{{cpa}} >= {avg_cpa}', 'column_id': 'cpa'}, 
         'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
        {'if': {'filter_query': f'{{mnet_roas}} > {avg_mnet_roas}', 'column_id': 'mnet_roas'}, 
         'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
        {'if': {'filter_query': f'{{mnet_roas}} <= {avg_mnet_roas}', 'column_id': 'mnet_roas'}, 
         'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
        {'if': {'filter_query': f'{{adv_roas}} > {avg_adv_roas}', 'column_id': 'adv_roas'}, 
         'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
        {'if': {'filter_query': f'{{adv_roas}} <= {avg_adv_roas}', 'column_id': 'adv_roas'}, 
         'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'}
    ]
    
    return html.Div([
        dbc.Button("Collapse", id={'type': 'collapse-btn', 'index': attr}, 
                   color="secondary", size="sm", style={'marginBottom': '10px'}),
        html.H6(f"Ad Titles for {attr.replace('_', ' ').title()}: {selected_value}", 
                style={'color': '#ffcc00', 'marginBottom': '10px'}),
        html.P([
            f"Category Averages: ",
            html.Span(f"CVR: {avg_cvr:.2f}% ", style={'color': '#17a2b8'}),
            html.Span(f"CTR: {avg_ctr:.2f}% ", style={'color': '#17a2b8'}),
            html.Span(f"CPA: ${avg_cpa:.2f} ", style={'color': '#17a2b8'}),
            html.Span(f"Mnet ROAS: {avg_mnet_roas:.2f} ", style={'color': '#17a2b8'}),
            html.Span(f"Adv ROAS: {avg_adv_roas:.2f}", style={'color': '#17a2b8'})
        ], style={'color': '#aaa', 'fontSize': '12px'}),
        dash_table.DataTable(
            columns=[
                {'name': 'Ad Title', 'id': 'ad_title'},
                {'name': 'Impressions', 'id': 'impressions'},
                {'name': 'Clicks', 'id': 'clicks'},
                {'name': 'CTR %', 'id': 'ctr'},
                {'name': 'CVR %', 'id': 'cvr'},
                {'name': 'CPA', 'id': 'cpa'},
                {'name': 'Mnet ROAS', 'id': 'mnet_roas'},
                {'name': 'Adv ROAS', 'id': 'adv_roas'}
            ],
            data=title_agg.round(2).to_dict('records'),
            style_cell={
                'textAlign': 'left',
                'backgroundColor': '#222',
                'color': 'white',
                'border': '1px solid #444',
                'fontSize': '11px',
                'padding': '8px',
                'minWidth': '80px',
                'maxWidth': '400px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis'
            },
            style_header={
                'backgroundColor': '#111',
                'fontWeight': 'bold',
                'border': '1px solid #444',
                'color': '#17a2b8'
            },
            style_data_conditional=style_conditional,
            page_size=10
        )
    ], style={'backgroundColor': '#1a1a1a', 'padding': '15px', 'borderRadius': '5px', 'border': '1px solid #444'})


@callback(
    Output({'type': 'attr-table', 'index': MATCH}, 'selected_rows'),
    Input({'type': 'collapse-btn', 'index': MATCH}, 'n_clicks'),
    prevent_initial_call=True
)
def collapse_drilldown(n_clicks):
    return []






