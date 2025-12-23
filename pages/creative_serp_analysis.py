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
    'logo_present'
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
    'is_number_present'
]
# Educational descriptions for Ad Title attributes
ATTRIBUTE_DESCRIPTIONS = {
    'specificity': {
        'title': '',
        'description': 'Measures clarity and detail in the ad title.\n\n• High: Contains specific numbers, facts, or detailed information that clearly communicates value\n• Medium: Has some specific elements but could be more detailed\n• Low: Vague or generic messaging without concrete details',
        'values': {
            'High': 'Title provides specific, concrete information (e.g., "Save $50 on...", "3 Easy Steps to...")',
            'Medium': 'Title has moderate detail but could be more specific',
            'Low': 'Generic or vague title lacking concrete details'
        }
    },
    'attention_trigger': {
        'title': '',
        'description': 'Measures how compelling the hook is - how likely users are to click.\n\n• High: Creates strong curiosity or intrigue without revealing everything; uses sensational language\n• Medium: Has some engaging elements but moderate appeal\n• Low: Straightforward title with minimal curiosity-building elements',
        'values': {
            'High': 'Strong hook with curiosity gap or sensational words (e.g., "Shocking Truth About...", "What They Don\'t Tell You...")',
            'Medium': 'Moderate appeal with some engaging elements',
            'Low': 'Direct, straightforward title with minimal intrigue'
        }
    },
    'tone': {
        'title': '',
        'description': 'Identifies the underlying emotion conveyed in the ad title.\n\n• Negative/Fear: Emphasizes problems, risks, or what could go wrong\n• Positive/Aspirational: Focuses on benefits, improvements, or positive outcomes\n• Neutral/Informational: Objective, fact-based without strong emotional slant',
        'values': {
            'Negative/Fear': 'Highlights problems, warnings, or negative consequences (e.g., "Avoid These Mistakes...", "Warning Signs of...")',
            'Positive/Aspirational': 'Emphasizes benefits and positive outcomes (e.g., "Achieve Your Dreams...", "Feel Better Than Ever")',
            'Neutral/Informational': 'Objective, informational tone without strong emotion'
        }
    },
    'trust_signal': {
        'title': 'Credibility Signal',
        'description': 'Measures what type of authority or proof the title leverages.\n\n• Expert-Based: References doctors, scientists, professionals, or expert opinions\n• Social-Proof-Based: Mentions celebrities, testimonials, or popular acceptance\n• None: No specific credibility signals present',
        'values': {
            'Expert-Based': 'Quotes experts, doctors, or professionals (e.g., "Doctors Recommend...", "Expert-Approved...")',
            'Social-Proof-Based': 'References celebrities, crowds, or social validation (e.g., "Thousands Trust...", "As Seen on TV")',
            'None': 'No specific trust signals or authority references'
        }
    },
    'framing': {
        'title': 'Message Framing',
        'description': 'Identifies whether the title focuses on problems, solutions, or context.\n\n• Problem-Based: Highlights pain points, challenges, or issues\n• Solution-Based: Presents answers, remedies, or ways to fix problems\n• Context-Based: Provides background information or educational content',
        'values': {
            'Problem-Based': 'Focuses on issues or pain points (e.g., "Struggling With...", "Tired of...")',
            'Solution-Based': 'Presents solutions or remedies (e.g., "How to Fix...", "The Solution to...")',
            'Context-Based': 'Provides informational or educational context'
        }
    },
    'character_count': {
        'title': 'Character Length',
        'description': 'Groups ad titles by their character count to analyze optimal length.\n\nDifferent lengths serve different purposes:\n• Shorter titles (1-30): Quick, punchy messages\n• Medium titles (31-60): Balanced detail and brevity\n• Longer titles (60+): Detailed, informative messages',
        'values': {
            '1-5': 'Very short - typically symbols or brand names',
            '6-10': 'Short and punchy',
            '11-20': 'Brief but can convey simple message',
            '21-30': 'Concise with moderate detail',
            '31-40': 'Balanced length for most ad titles',
            '41-50': 'Detailed messaging',
            '51-60': 'Extended detail',
            '60+': 'Long-form titles with comprehensive information'
        }
    },
    'is_number_present': {
        'title': 'Numeric Elements',
        'description': 'Indicates whether the title contains numbers.\n\nNumbers can:\n• Increase specificity and credibility\n• Attract attention with concrete data\n• Set clear expectations (e.g., "7 Tips", "$50 Off", "In 30 Days")',
        'values': {
            'Yes': 'Title contains numbers - often increases click-through by providing specific, measurable information',
            'No': 'Title does not contain numbers - relies on other persuasive elements'
        }
    }
}
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
CREATIVE_SERP_FILE_ID = "1BwA_h5Z5Hmo_WKTh6a4a0Jc-Ep2btgk9"
CREATIVE_SERP_URL = f"https://drive.google.com/uc?export=download&id={CREATIVE_SERP_FILE_ID}"

# File 2: Ad Title Data
AD_TITLE_FILE_ID = "1qD-G1W1gWT3uFiE_6u30XSGD1Vm-F5KJ"
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
    'Text CTA Present?': 'cta_present',
    'CTA Color': 'cta_color',
    'Adv Logo Present?': 'logo_present',
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
    def create_char_bucket(count):
        if pd.isna(count):
            return 'Unknown'
        count = int(count)
        if count <= 0:
            return '0'
        elif count <= 5:
            return '1-5'
        elif count <= 10:
            return '6-10'
        elif count <= 15:
            return '11-15'
        elif count <= 20:
            return '16-20'
        elif count <= 25:
            return '21-25'
        elif count <= 30:
            return '26-30'
        elif count <= 35:
            return '31-35'
        elif count <= 40:
            return '36-40'
        elif count <= 45:
            return '41-45'
        elif count <= 50:
            return '46-50'
        elif count <= 60:
            return '51-60'
        elif count <= 70:
            return '61-70'
        elif count <= 80:
            return '71-80'
        elif count <= 90:
            return '81-90'
        elif count <= 100:
            return '91-100'
        else:
            return '100+'
    if 'character_count' in ad_df.columns:
        ad_df['character_count'] = ad_df['character_count'].apply(create_char_bucket)
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
        html.Div(id='shared_agg_stats', style={'marginBottom': '20px'}),


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
                                        
                                        html.Hr(style={'borderColor': '#444'}),

                                        html.H4("Creative Performance", style={'color': '#5dade2'}),
                                        html.P("📊 Analyze creative templates by performance metrics", 
                                               style={'color': '#aaa', 'fontSize': '12px', 'marginBottom': '15px'}),
                                        dbc.Row([
                                            dbc.Col([
                                                html.Label("Performance Type:", style={'color': 'white'}),
                                                dcc.Dropdown(id='creative_table_type', value='best',
                                                             options=[{'label': '✓ Best Performers', 'value': 'best'},
                                                                      {'label': '✗ Worst Performers', 'value': 'worst'}],
                                                             style={'color': 'black'})
                                            ], width=3),
                                            dbc.Col([
                                                html.Label("Number of Items:", style={'color': 'white'}),
                                                dcc.Dropdown(id='creative_table_count', value=5,
                                                             options=[{'label': str(x), 'value': x} for x in [5, 10, 15, 20]],
                                                             style={'color': 'black'})
                                            ], width=2),
                                            dbc.Col([
                                                html.Label("Sort By:", style={'color': 'white'}),
                                                dcc.Dropdown(id='creative_table_sort', value='cvr',
                                                             options=[{'label': 'CVR', 'value': 'cvr'},
                                                                      {'label': 'CTR', 'value': 'ctr'},
                                                                      {'label': 'Clicks', 'value': 'clicks'},
                                                                      {'label': 'CPA', 'value': 'cpa'},
                                                                      {'label': 'ROAS', 'value': 'mnet_roas'}],
                                                             style={'color': 'black'})
                                            ], width=2)
                                        ], style={'marginBottom': '15px'}),
                                        dash_table.DataTable(
                                            id='dynamic_creatives',
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
                                        
                                        # Creative Tables
                                        html.Hr(style={'borderColor': '#444'}),

                                # SERP Tables
                                        html.H4("SERP Performance", style={'color': '#5dade2', 'marginTop': '20px'}),
                                        html.P("📄 Analyze SERP templates by performance metrics", 
                                               style={'color': '#aaa', 'fontSize': '12px', 'marginBottom': '15px'}),
                                        dbc.Row([
                                            dbc.Col([
                                                html.Label("Performance Type:", style={'color': 'white'}),
                                                dcc.Dropdown(id='serp_table_type', value='best',
                                                             options=[{'label': '✓ Best Performers', 'value': 'best'},
                                                                      {'label': '✗ Worst Performers', 'value': 'worst'}],
                                                             style={'color': 'black'})
                                            ], width=3),
                                            dbc.Col([
                                                html.Label("Number of Items:", style={'color': 'white'}),
                                                dcc.Dropdown(id='serp_table_count', value=5,
                                                             options=[{'label': str(x), 'value': x} for x in [5, 10, 15, 20]],
                                                             style={'color': 'black'})
                                            ], width=2),
                                            dbc.Col([
                                                html.Label("Sort By:", style={'color': 'white'}),
                                                dcc.Dropdown(id='serp_table_sort', value='cvr',
                                                             options=[{'label': 'CVR', 'value': 'cvr'},
                                                                      {'label': 'CTR', 'value': 'ctr'},
                                                                      {'label': 'Clicks', 'value': 'clicks'},
                                                                      {'label': 'CPA', 'value': 'cpa'},
                                                                      {'label': 'ROAS', 'value': 'mnet_roas'}],
                                                             style={'color': 'black'})
                                            ], width=2)
                                        ], style={'marginBottom': '15px'}),
                                        dash_table.DataTable(
                                            id='dynamic_serps',
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
                                        html.H4("Creative-SERP Pair Performance", style={'color': '#5dade2', 'marginTop': '20px'}),
                                        html.P("🔗 Analyze creative-SERP combinations", 
                                               style={'color': '#aaa', 'fontSize': '12px', 'marginBottom': '15px'}),
                                        dbc.Row([
                                            dbc.Col([
                                                html.Label("Performance Type:", style={'color': 'white'}),
                                                dcc.Dropdown(id='pair_table_type', value='best',
                                                             options=[{'label': '✓ Best Performers', 'value': 'best'},
                                                                      {'label': '✗ Worst Performers', 'value': 'worst'}],
                                                             style={'color': 'black'})
                                            ], width=3),
                                            dbc.Col([
                                                html.Label("Number of Items:", style={'color': 'white'}),
                                                dcc.Dropdown(id='pair_table_count', value=5,
                                                             options=[{'label': str(x), 'value': x} for x in [5, 10, 15, 20]],
                                                             style={'color': 'black'})
                                            ], width=2),
                                            dbc.Col([
                                                html.Label("Sort By:", style={'color': 'white'}),
                                                dcc.Dropdown(id='pair_table_sort', value='cvr',
                                                             options=[{'label': 'CVR', 'value': 'cvr'},
                                                                      {'label': 'CTR', 'value': 'ctr'},
                                                                      {'label': 'Clicks', 'value': 'clicks'},
                                                                      {'label': 'CPA', 'value': 'cpa'},
                                                                      {'label': 'ROAS', 'value': 'mnet_roas'}],
                                                             style={'color': 'black'})
                                            ], width=2)
                                        ], style={'marginBottom': '15px'}),
                                        dash_table.DataTable(
                                            id='dynamic_pairs',
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
                                        # Creative-SERP Pair Tables
                                       
                                        
                                        html.Hr(style={'borderColor': '#444'}),
                                        
                                        # Creative & SERP Attributes Section
                                        html.H4("SERP Design Attributes Analysis", style={'color': '#5dade2', 'marginTop': '20px'}),
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
@callback(
    Output('shared_agg_stats', 'children'),
    [Input('cs_adv_dd', 'value'),
     Input('cs_camp_type_dd', 'value'),
     Input('cs_camp_dd', 'value'),
     Input('main_tabs', 'value')]
)
def update_shared_stats(advs, camp_types, camps, active_tab):
    # Use ad_df for Ad Title tab, df for Creative & SERP tab
    if active_tab == 'tab-ad-title' and AD_TITLE_AVAILABLE:
        d = ad_df.copy()
    else:
        d = df.copy()
    
    # Apply filters
    if advs:
        d = d[d['advertiser'].isin(advs)]
    if camp_types:
        d = d[d['campaign_type'].isin(camp_types)]
    if camps:
        d = d[d['campaign'].isin(camps)]
    
    # Calculate aggregated stats (same logic as before)
    total_clicks = d['clicks'].sum()
    total_impressions = d['impressions'].sum()
    total_conversions = d['conversions'].sum()
    total_adv_cost = d['adv_cost'].sum()
    total_max_cost = d['max_cost'].sum()
    total_actual_adv_payout = d['actual_adv_payout'].sum()
    
    agg_cvr = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
    agg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
    agg_cpa = (total_adv_cost / total_conversions) if total_conversions > 0 else 0
    agg_mnet_roas = (total_actual_adv_payout / total_max_cost) if total_max_cost > 0 else 0
    agg_adv_roas = (total_actual_adv_payout / total_adv_cost) if total_adv_cost > 0 else 0
    
    # Return the stats card (same as before)
    return dbc.Card([
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
                        html.Span(f"{agg_mnet_roas:.2f}", style={'color': '#00ff00', 'fontSize': '18px'})], width=2)
            ])
        ])
    ], style={'backgroundColor': '#222', 'border': '1px solid #444'})
# Update Creative & SERP Analysis
@callback(
    [Output('dynamic_creatives', 'data'),
     Output('dynamic_creatives', 'style_data_conditional'),
     Output('dynamic_serps', 'data'),
     Output('dynamic_serps', 'style_data_conditional'),
     Output('dynamic_pairs', 'data'),
     Output('dynamic_pairs', 'style_data_conditional')],
    [Input('cs_adv_dd', 'value'),
     Input('cs_camp_type_dd', 'value'),
     Input('cs_camp_dd', 'value'),
     Input('creative_table_type', 'value'),
     Input('creative_table_count', 'value'),
     Input('creative_table_sort', 'value'),
     Input('serp_table_type', 'value'),
     Input('serp_table_count', 'value'),
     Input('serp_table_sort', 'value'),
     Input('pair_table_type', 'value'),
     Input('pair_table_count', 'value'),
     Input('pair_table_sort', 'value')]
)
def update_creative_serp(advs, camp_types, camps, 
                        creative_type, creative_count, creative_sort,
                        serp_type, serp_count, serp_sort,
                        pair_type, pair_count, pair_sort):
    # Filter data
    d = df.copy()
    if advs:
        d = d[d['advertiser'].isin(advs)]
    if camp_types:
        d = d[d['campaign_type'].isin(camp_types)]
    if camps:
        d = d[d['campaign'].isin(camps)]
    
    # Calculate aggregated stats
    total_clicks = d['clicks'].sum()
    total_impressions = d['impressions'].sum()
    total_conversions = d['conversions'].sum()
    total_adv_cost = d['adv_cost'].sum()
    total_max_cost = d['max_cost'].sum()
    total_actual_adv_payout = d['actual_adv_payout'].sum()
    
    agg_cvr = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
    agg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
    agg_cpa = (total_adv_cost / total_conversions) if total_conversions > 0 else 0
    agg_mnet_roas = (total_actual_adv_payout / total_max_cost) if total_max_cost > 0 else 0
    agg_adv_roas = (total_actual_adv_payout / total_adv_cost) if total_adv_cost > 0 else 0
        
    def get_dynamic_table(d, group_col, table_type, table_count, table_sort, min_clicks=3):
        """Generate dynamic table based on filters"""
        g = d.groupby(group_col, dropna=True).agg(
            clicks=('clicks','sum'),
            impressions=('impressions','sum'),
            conversions=('conversions','sum'),
            adv_cost=('adv_cost','sum'),
            max_cost=('max_cost','sum'),
            actual_adv_payout=('actual_adv_payout','sum'),
            mnet_roas=('mnet_roas','mean'),
            adv_roas=('adv_roas','mean')
        ).reset_index()
        
        g['ctr'] = np.where(g['impressions']>0, 100*g['clicks']/g['impressions'], np.nan)
        g['cvr'] = np.where(g['clicks']>0, 100*g['conversions']/g['clicks'], np.nan)
        g['cpa'] = np.where(g['conversions']>0, g['adv_cost']/g['conversions'], np.nan)
        
        g = g.dropna(subset=['cvr'])
        
        if table_type == 'best':
            candidates = g[(g['clicks'] >= min_clicks) & (g['cvr'] > 0)].copy()
            if len(candidates) == 0:
                candidates = g[g['cvr'] > 0].copy()
            
            if len(candidates) > 0:
                if table_sort == 'cvr':
                    candidates['score'] = candidates['cvr'] * np.log1p(candidates['clicks'])
                    result_df = candidates.sort_values('score', ascending=False).head(table_count)
                else:
                    result_df = candidates.nlargest(table_count, table_sort)
            else:
                result_df = pd.DataFrame()
        else:  # worst
            candidates = g[(g['clicks'] >= min_clicks) & (g['cvr'] <= 0.6)].copy()
            if len(candidates) > 0:
                result_df = candidates.nlargest(table_count, 'clicks')
            else:
                result_df = pd.DataFrame()
        
        return result_df.round(2).to_dict('records')
    
    def add_color_conditional(agg_cvr, agg_ctr, agg_mnet_roas, agg_adv_roas, agg_cpa):
        """Generate conditional styling"""
        return [
            {'if': {'filter_query': f'{{cvr}} > {agg_cvr}', 'column_id': 'cvr'}, 
             'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{cvr}} <= {agg_cvr}', 'column_id': 'cvr'}, 
             'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{ctr}} > {agg_ctr}', 'column_id': 'ctr'}, 
             'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{ctr}} <= {agg_ctr}', 'column_id': 'ctr'}, 
             'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{cpa}} < {agg_cpa}', 'column_id': 'cpa'}, 
             'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{cpa}} >= {agg_cpa}', 'column_id': 'cpa'}, 
             'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{mnet_roas}} > {agg_mnet_roas}', 'column_id': 'mnet_roas'}, 
             'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{mnet_roas}} <= {agg_mnet_roas}', 'column_id': 'mnet_roas'}, 
             'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{adv_roas}} > {agg_adv_roas}', 'column_id': 'adv_roas'}, 
             'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{adv_roas}} <= {agg_adv_roas}', 'column_id': 'adv_roas'}, 
             'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'}
        ]
    
    # Get results
    dynamic_creatives = get_dynamic_table(d, 'creative', creative_type, creative_count, creative_sort)
    dynamic_serps = get_dynamic_table(d, 'serp', serp_type, serp_count, serp_sort)
    
    # For pairs
    d['pair'] = d['creative'].astype(str) + ' | ' + d['serp'].astype(str)
    dynamic_pairs_raw = get_dynamic_table(d, 'pair', pair_type, pair_count, pair_sort)
    
    # Split pairs back
    dynamic_pairs = []
    for item in dynamic_pairs_raw:
        if 'pair' in item:
            parts = item['pair'].split(' | ')
            item['creative'] = parts[0]
            item['serp'] = parts[1] if len(parts) > 1 else ''
            del item['pair']
        dynamic_pairs.append(item)
    
    conditional = add_color_conditional(agg_cvr, agg_ctr, agg_mnet_roas, agg_adv_roas, agg_cpa)
    
    return (dynamic_creatives, conditional,
            dynamic_serps, conditional,
            dynamic_pairs, conditional)
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
        
        # Custom display names
        ATTR_DISPLAY_NAMES = {
            'cta_present': 'Text CTA Present?',
            'bg_color': 'Background Color',
            'font_color': 'Main Font Color',
            'cta_color': 'CTA Color',
            'logo_present': 'Adv Logo Present?'
            }
        attr_display_name = ATTR_DISPLAY_NAMES.get(attr, attr.replace('_', ' ').title())
        
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
            {'if': {'state': 'active'},
             'backgroundColor': '#17a2b8', 'border': '2px solid #00ff00'},
            {'if': {'state': 'selected'},
             'backgroundColor': '#0d5e6b', 'border': '2px solid #00ff00'}
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
    
    # Aggregate by SERP (instead of Creative-SERP pairs)
    serp_agg = detail_data.groupby('serp', dropna=True).agg(
        impressions=('impressions', 'sum'),
        clicks=('clicks', 'sum'),
        conversions=('conversions', 'sum'),
        adv_cost=('adv_cost', 'sum'),
        max_cost=('max_cost', 'sum'),
        actual_adv_payout=('actual_adv_payout', 'sum')
    ).reset_index()
    
    # Calculate metrics
    serp_agg['ctr'] = np.where(serp_agg['impressions']>0, 100*serp_agg['clicks']/serp_agg['impressions'], 0)
    serp_agg['cvr'] = np.where(serp_agg['clicks']>0, 100*serp_agg['conversions']/serp_agg['clicks'], 0)
    serp_agg['cpa'] = np.where(serp_agg['conversions']>0, serp_agg['adv_cost']/serp_agg['conversions'], 0)
    serp_agg['mnet_roas'] = np.where(serp_agg['max_cost']>0, serp_agg['actual_adv_payout']/serp_agg['max_cost'], 0)
    serp_agg['adv_roas'] = np.where(serp_agg['adv_cost']>0, serp_agg['actual_adv_payout']/serp_agg['adv_cost'], 0)
    
    # Sort by clicks and take top 5
    serp_agg = serp_agg.sort_values('clicks', ascending=False).head(5)
    
    # Round
    serp_agg = serp_agg.round(2)
    
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
    
    ATTR_DISPLAY_NAMES = {
        'cta_present': 'Text CTA Present?',
        'bg_color': 'Background Color',
        'font_color': 'Main Font Color',
        'cta_color': 'CTA Color',
        'logo_present': 'Adv Logo Present?'
    }
    attr_display = ATTR_DISPLAY_NAMES.get(attr, attr.replace('_', ' ').title())
    
    return html.Div([
        dbc.Button("▲ Collapse", 
                   id={'type': 'cs-collapse-btn', 'index': attr}, 
                   color="info", 
                   size="sm", 
                   style={'marginBottom': '10px'}),
        html.H6(f"Top 5 SERPs by Clicks for {attr_display}: {selected_value}", 
                style={'color': '#17a2b8', 'marginBottom': '10px'}),
        dash_table.DataTable(
            columns=[
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
            data=serp_agg.to_dict('records'),
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
            page_size=5
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
        
        # Calculate overall averages for this attribute
        overall_avg_ctr = agg_data['ctr'].mean()
        overall_avg_cpa = agg_data['cpa'].mean()
        overall_avg_mnet_roas = agg_data['mnet_roas'].mean()
        overall_avg_adv_roas = agg_data['adv_roas'].mean()
        
        # Round numbers (don't format yet - need numeric for conditional styling)
        agg_data = agg_data.round(2)
        
        # Create table
        table_id = f'table_{attr}'
        drill_down_id = f'drilldown_{attr}'
        style_conditional = [
            {'if': {'filter_query': f'{{cvr}} > {overall_avg_cvr}', 'column_id': 'cvr'}, 
             'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{cvr}} <= {overall_avg_cvr}', 'column_id': 'cvr'}, 
             'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{ctr}} > {overall_avg_ctr}', 'column_id': 'ctr'}, 
             'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{ctr}} <= {overall_avg_ctr}', 'column_id': 'ctr'}, 
             'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{cpa}} < {overall_avg_cpa}', 'column_id': 'cpa'}, 
             'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{cpa}} >= {overall_avg_cpa}', 'column_id': 'cpa'}, 
             'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{mnet_roas}} > {overall_avg_mnet_roas}', 'column_id': 'mnet_roas'}, 
             'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{mnet_roas}} <= {overall_avg_mnet_roas}', 'column_id': 'mnet_roas'}, 
             'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{adv_roas}} > {overall_avg_adv_roas}', 'column_id': 'adv_roas'}, 
             'backgroundColor': '#1a4d2e', 'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{adv_roas}} <= {overall_avg_adv_roas}', 'column_id': 'adv_roas'}, 
             'backgroundColor': '#4d1a1a', 'color': '#ff0000', 'fontWeight': 'bold'}
        ]
        tooltip_header = {
            attr: ATTRIBUTE_DESCRIPTIONS.get(attr, {}).get('description', attr_display_name)
        }
        tooltip_data = []
        for idx, row in agg_data.iterrows():
            attr_value = row[attr]
            value_description = ATTRIBUTE_DESCRIPTIONS.get(attr, {}).get('values', {}).get(str(attr_value), '')
            
            tooltip_row = {
                attr: f"**{attr_value}**: {value_description}" if value_description else str(attr_value)
            }
            tooltip_data.append(tooltip_row)

        
        attribute_sections.append(
            html.Div([
                html.H5(f"{attr_display_name} Performance", style={'color': '#5dade2', 'marginTop': '30px'}),
                html.P([
                    f"💡 {ATTRIBUTE_DESCRIPTIONS.get(attr, {}).get('title', attr_display_name)}: ",
                    html.Span(ATTRIBUTE_DESCRIPTIONS.get(attr, {}).get('description', '').split('\n\n')[0], 
                             style={'color': '#aaa', 'fontSize': '11px', 'fontStyle': 'italic'})
                ], style={'marginBottom': '10px'}),
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
                    style_data_conditional=style_conditional,
                    row_selectable='single',
                    selected_rows=[],
                    hidden_columns=[],
                    # ADD TOOLTIPS HERE
                    tooltip_header=tooltip_header,
                    tooltip_data=tooltip_data,
                    tooltip_duration=None,
                    tooltip_delay=0,
                    css=[{
                        'selector': '.dash-table-tooltip',
                        'rule': '''
                            background-color: #1a1a1a;
                            border: 2px solid #17a2b8;
                            color: white;
                            font-size: 12px;
                            padding: 12px;
                            max-width: 400px;
                            white-space: pre-wrap;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                        '''
                    }]
                ),
                html.Div(id={'type': 'drilldown-container', 'index': attr})
            ])
        )
    
    return html.Div([
        html.H3("Ad Title Performance Analysis", style={'color': '#17a2b8', 'marginBottom': '20px'}),
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
    avg_cvr = table_data[selected_rows[0]]['cvr']
    avg_ctr = table_data[selected_rows[0]]['ctr']
    avg_cpa = table_data[selected_rows[0]]['cpa']
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











