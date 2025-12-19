import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import re
from collections import Counter
from itertools import combinations

import dash
from dash import dcc, html, Input, Output, dash_table, State, register_page, callback

import plotly.express as px
import plotly.graph_objects as go


register_page(__name__, path='/', name='URL')

# [REST OF YOUR CURRENT CODE - just remove the app.run() at bottom]
# ... all your existing code ...



# =========================================================
# CONFIG
# =========================================================
DATA_FILE = "URL_File_Full_02Dec2025_16Dec2025.csv"

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(DATA_FILE)

# =========================================================
# COLUMN NORMALIZATION
# =========================================================
COL_MAP = {
    'Advertiser': 'advertiser',
    'Campaign Type': 'campaign_type',
    'Campaign': 'campaign',
    'Clicks': 'clicks',
    'Ad Impressions': 'impressions',
    'Weighted Conversion': 'conversions',
    'Advertiser Cost': 'adv_cost',
    'Max System Cost': 'max_cost',
    'Advertiser Value': 'adv_value',
    'Original Publisher Url': 'url',
    'Domain': 'domain',
    'URL-Concept Contextuality': 'contextuality',
    'Sprig URL Category': 'sprig_url',
    'Sprig Domain Category': 'sprig_domain',
    'Mnet ROAS': 'mnet_roas',
    'Adv ROAS': 'adv_roas'
}
df = df.rename(columns={c: COL_MAP[c] for c in df.columns if c in COL_MAP})

for c in ['clicks','impressions','conversions','adv_cost','max_cost','adv_value','mnet_roas','adv_roas']:
    df[c] = pd.to_numeric(df.get(c, 0), errors='coerce').fillna(0)

# =========================================================
# METRICS
# =========================================================
df['ctr'] = np.where(df['impressions']>0, 100*df['clicks']/df['impressions'], np.nan)
df['cvr'] = np.where(df['clicks']>0, 100*df['conversions']/df['clicks'], np.nan)
df['cpa'] = np.where(df['conversions']>0, df['adv_cost']/df['conversions'], np.nan)

# =========================================================
# SPRIG PARSING
# =========================================================
def clean_sprig(x):
    if not isinstance(x, str):
        return np.nan
    cleaned = re.sub(r'\[.*?\]', '', x).strip()
    # Remove trailing " -" or " - " patterns
    cleaned = re.sub(r'\s*-\s*$', '', cleaned)
    return cleaned

def sprig_top(x):
    x = clean_sprig(x)
    return x.split('>')[0].strip() if isinstance(x,str) else np.nan

def sprig_final(x):
    x = clean_sprig(x)
    return x.split('>')[-1].strip() if isinstance(x,str) else np.nan

df['sprig_url_top'] = df['sprig_url'].apply(sprig_top)
df['sprig_url_final'] = df['sprig_url'].apply(sprig_final)
df['sprig_domain_top'] = df['sprig_domain'].apply(sprig_top)
df['sprig_domain_final'] = df['sprig_domain'].apply(sprig_final)

# =========================================================
# ENHANCED CONCEPT EXTRACTION
# =========================================================
STOP_WORDS = {
    'com','www','http','https','html','php','aspx','jsp','htm',
    'the','and','for','with','from','this','that','are','was','were',
    'been','have','has','had','will','would','could','should','can',
    'may','might','must','shall','being','am','is','how','why','what',
    'when','where','who','which','their','there','these','those','then',
    'than','them','they','about','after','all','also','any','because',
    'but','does','did','each','few','more','most','other','some','such',
    'through','into','during','before','after','above','below','between',
    'under','again','further','here','once','only','over','same','very',
    # Tracking parameters and technical terms
    'utm','gclid','fbclid','msclkid','gads','mnets','whitelist','blacklist',
    'tracking','param','params','ref','source','medium','campaign','term',
    'content','click','clicks','adid','adgroup','keyword','placement',
    'creative','network','device','match','type','target','audience', 'id', 'gad', 'campaignid',
    'medianet', 'mnet','topic','mnettopic','1','2','3','4','5','6','7','8','9','0'
}

def extract_parent_domain(url):
    """Extract parent domain like yahoo, finance, etc."""
    if not isinstance(url, str):
        return []
    match = re.search(r'://([^/]+)', url)
    if match:
        domain = match.group(1)
        parts = domain.split('.')
        return [p.lower() for p in parts if len(p) > 2]
    return []

def extract_concepts(url):
    """Extract single words and meaningful n-grams"""
    if not isinstance(url, str):
        return []
    
    # Remove protocol
    url = re.sub(r'https?://', '', url)
    
    # Get parent domain to exclude
    parent_domains = set(extract_parent_domain('http://' + url))
    
    # Extract all words (letters only, 3+ chars)
    words = re.findall(r'[a-zA-Z]{3,}', url.lower())
    
    # Filter: remove stop words, numbers, parent domains
    words = [w for w in words if w not in STOP_WORDS and w not in parent_domains and not w.isdigit()]
    
    concepts = []
    
    # Single words
    concepts.extend(words)
    
    # 2-word combinations (bigrams)
    if len(words) >= 2:
        for i in range(len(words) - 1):
            concepts.append(f"{words[i]} {words[i+1]}")
    
    # 3-word combinations (trigrams)
    if len(words) >= 3:
        for i in range(len(words) - 2):
            concepts.append(f"{words[i]} {words[i+1]} {words[i+2]}")
    
    return concepts

df['concepts'] = df['url'].apply(extract_concepts)

# Explode concepts for aggregation
df_concepts = df.explode('concepts').dropna(subset=['concepts'])
# Use the actual contextuality column from CSV
df_contextuality = df[df['contextuality'].notna()].copy()

# =========================================================
# WEIGHTED AGGREGATION
# =========================================================
def weighted_aggregate(df, col):
    g = df.groupby(col, dropna=True).agg(
        clicks=('clicks','sum'),
        impressions=('impressions','sum'),
        conversions=('conversions','sum'),
        adv_cost=('adv_cost','sum'),
        max_cost=('max_cost','sum'),
        adv_value=('adv_value','sum'),
        mnet_roas=('mnet_roas','mean'),
        adv_roas=('adv_roas','mean')
    ).reset_index()

    # Weighted metrics
    g['ctr'] = np.where(g['impressions']>0, 100*g['clicks']/g['impressions'], np.nan)
    g['cvr'] = np.where(g['clicks']>0, 100*g['conversions']/g['clicks'], np.nan)
    g['cpa'] = np.where(g['conversions']>0, g['adv_cost']/g['conversions'], np.nan)
    g['mnet_roas'] = np.where(g['max_cost']>0, g['adv_value']/g['max_cost'], np.nan)  # ADD THIS LINE


    return g
# =========================================================
# TOP 3 URLS BY CLICKS
# =========================================================
def top3_urls(df, group_col):
    out = {}
    for k, g in df.groupby(group_col):
        top = (
            g.groupby('url')
             .agg(clicks=('clicks','sum'), conversions=('conversions','sum'))
             .sort_values('clicks', ascending=False)
             .head(3)
        )
        lines = [f"{i+1}. {idx[:60]}... (Cl:{int(r.clicks)}, Cv:{int(r.conversions)})" 
                 for i,(idx,r) in enumerate(top.iterrows())]
        out[k] = "<br>".join(lines) if lines else "No URLs"
    return out

# =========================================================
# TREEMAP FUNCTIONS
# =========================================================
def create_treemap(g, metric_color, metric_sort, title, show_cvr_ctr=True, top_n=10, col_name='concepts', avg_metrics=None):
    g = g.dropna(subset=[metric_color, metric_sort])
    if len(g) == 0:
        fig = go.Figure()
        fig.update_layout(title="No data", plot_bgcolor='#111', paper_bgcolor='#111', font=dict(color='white'))
        return fig
    
    # Remove duplicates by grouping if needed
    if col_name in g.columns:
        g = g.groupby(col_name, dropna=True).agg({
            'clicks': 'sum',
            'impressions': 'sum',
            'conversions': 'sum',
            'adv_cost': 'sum',
            'max_cost': 'sum',
            'adv_value': 'sum',
            'mnet_roas': 'mean',
            'adv_roas': 'mean'
        }).reset_index()
        
        # Recalculate metrics
        g['ctr'] = np.where(g['impressions']>0, 100*g['clicks']/g['impressions'], np.nan)
        g['cvr'] = np.where(g['clicks']>0, 100*g['conversions']/g['clicks'], np.nan)
        g['cpa'] = np.where(g['conversions']>0, g['adv_cost']/g['conversions'], np.nan)
        g['mnet_roas'] = np.where(g['max_cost']>0, g['adv_value']/g['max_cost'], np.nan)
    
    # Sort by metric_sort DESCENDING and take top N by clicks
    g = g.nlargest(min(top_n, len(g)), 'clicks')
    g = g.sort_values(metric_sort, ascending=False).reset_index(drop=True)
    
    # Calculate color thresholds for metric_color
    avg = g[metric_color].mean()
    std = g[metric_color].std()
    good_threshold = avg + 0.5 * std
    
    # Create hover text based on treemap type
    # Create hover text based on treemap type
    label_col = col_name if col_name in g.columns else 'concepts'
    
    if show_cvr_ctr:
        g['hover_text'] = (
            '<b>' + g[label_col].astype(str) + '</b><br>' +
            'Clicks: ' + g['clicks'].astype(int).astype(str) + '<br>' +
            'CVR: ' + g['cvr'].round(2).astype(str) + '% (Avg: ' + (str(round(avg_metrics['cvr'], 2)) if avg_metrics else str(round(g['cvr'].mean(), 2))) + '%)<br>' +
            'CTR: ' + g['ctr'].round(2).astype(str) + '%'
        )
    else:
        g['hover_text'] = (
            '<b>' + g[label_col].astype(str) + '</b><br>' +
            'Clicks: ' + g['clicks'].astype(int).astype(str) + '<br>' +
            'CPA: ' + g['cpa'].round(2).astype(str) + '<br>' +
            'Mnet ROAS: ' + g['mnet_roas'].round(2).astype(str)
        )
    
    fig = go.Figure(go.Treemap(
        labels=g[label_col],
        parents=[''] * len(g),
        values=g['clicks'],
        marker=dict(
        colorscale=[[0, '#cc0000'], [0.5, '#ffcc00'], [1, '#00cc00']],
        cmid=avg,
        cmin=0,
        cmax=good_threshold,
        colorbar=dict(title=metric_color.upper(), tickfont=dict(color='white')),
        line=dict(width=2, color='#000')
    ),
        textposition='middle center',
        textfont=dict(size=14, color='black', family='Arial Black'), 
        hovertext=g['hover_text'],
        hoverinfo='text',
        marker_colors=g[metric_color],
        customdata=g[label_col]))
    # Add aggregated stats annotation
    if avg_metrics:
        if show_cvr_ctr:
            stats_text = f"Aggregated: CVR={avg_metrics['cvr']:.2f}% | CTR={avg_metrics['ctr']:.2f}%"
        else:
            stats_text = f"Aggregated: CPA=${avg_metrics['cpa']:.2f} | Mnet ROAS={avg_metrics['mnet_roas']:.2f}"
        
        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=0.5, y=-0.05,
            showarrow=False,
            font=dict(size=14, color='#5dade2', family='Arial Black'),
            align='center'
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#5dade2')),
        height=500,
        plot_bgcolor='#111',
        paper_bgcolor='#111',
        font=dict(color='white', size=12),
        margin=dict(b=80)  # Add bottom margin for stats

    )
    
    return fig

# =========================================================
# BUBBLE CHART
# =========================================================
def bubble_chart(g, x_col, metric, hover_map, title, top_n=5):
    g = g.dropna(subset=[metric])
    if len(g) == 0:
        fig = go.Figure()
        fig.update_layout(title=f"No data", plot_bgcolor='#111', paper_bgcolor='#111', font=dict(color='white'))
        return fig
    
    # Filter to top N by clicks
    g = g.nlargest(top_n, 'clicks').reset_index(drop=True)
    
    g['top_urls'] = g[x_col].map(hover_map)
    
    # Calculate statistics
    avg = g[metric].mean()
    std = g[metric].std()
    
    # Create hover text
    g['hover_text'] = (
        '<b>' + g[x_col].astype(str) + '</b><br>' +
        'Clicks: ' + g['clicks'].astype(int).astype(str) + '<br>' +
        'Conversions: ' + g['conversions'].round(2).astype(str) + '<br>' +
        'CVR: ' + g['cvr'].round(2).astype(str) + '%<br>' +
        'CTR: ' + g['ctr'].round(2).astype(str) + '%<br>' +
        'CPA: ' + g['cpa'].round(2).astype(str) + '<br>' +
        'Mnet ROAS: ' + g['mnet_roas'].round(2).astype(str) + '<br><br>' +
        '<b>Top 3 URLs:</b><br>' + g['top_urls']
    )
    
    # Use soft, aesthetic colors (pastel and muted tones)
    soft_colors = ['#87CEEB', '#98D8C8', '#F7DC6F', '#BB8FCE', '#F8B88B', 
                   '#85C1E2', '#52B2BF', '#F9E79F', '#AED6F1', '#FAD7A0']
    color_indices = np.arange(len(g)) % len(soft_colors)
    bubble_colors = [soft_colors[i] for i in color_indices]
    
    fig = go.Figure()
    
    # Calculate bubble sizes - make them larger and more spread out
# Calculate bubble sizes - make them larger and more spread out
    # Calculate bubble sizes using logarithmic scaling for better visual distribution
    # Calculate bubble sizes with adjusted scaling for better visibility
    max_size = 100
    min_size = 30
    if g['clicks'].max() > g['clicks'].min():
    # Use square root scaling for more realistic size differences
      sqrt_clicks = np.sqrt(g['clicks'])
      sizes = (sqrt_clicks - sqrt_clicks.min()) / (sqrt_clicks.max() - sqrt_clicks.min()) * (max_size - min_size) + min_size
    else:
        sizes = [max_size] * len(g)

# Store sizes for later use in y-axis calculation
    bubble_sizes = sizes
    fig.add_trace(go.Scatter(
        x=list(range(len(g))),  # Use indices for better spacing
        y=g[metric],
        mode='markers',  # Remove text mode
        marker=dict(
            size=sizes,
            color=bubble_colors,
            line=dict(width=3, color='white'),
            opacity=0.9
        ),
        text=g['hover_text'],
        hoverinfo='text',
        hoverlabel=dict(
            bgcolor='#222',
            font_size=12,
            font_color='white',
            align='left',
            namelength=-1
        )
    ))
    
    # Determine y-axis range - Allow negative values and show bubbles fully
    # Determine y-axis range - Account for bubble sizes to avoid cutting
    min_val = g[metric].min()
    max_val = g[metric].max()
    data_range = max_val - min_val if max_val != min_val else 1

# Convert largest bubble size from pixels to data units
# Assuming chart height of 600px, calculate percentage of data range
    max_bubble_radius = max(bubble_sizes) / 2  # radius in pixels
    chart_height_px = 600
    bubble_percentage = max_bubble_radius / chart_height_px
    range_buffer = data_range * bubble_percentage * 2.5  # 2.5x for safe margin
    
    if metric in ['cvr', 'ctr']:
        y_min = min_val - range_buffer
        y_max = min(100, max_val + range_buffer)
    
    elif metric == 'cpa':
        # CPA can show slightly negative for buffer
        y_min = min_val - range_buffer
        y_max = max_val + range_buffer
    elif metric == 'mnet_roas':
        # ROAS can be any value
        y_min = min(0, min_val - range_buffer)
        y_max = max_val + range_buffer
    else:
        y_min = min(0, min_val - range_buffer)
        y_max = max_val + range_buffer
    
    # Add reference lines with better labels
    fig.add_hline(y=avg, line_dash='solid', line_color='#ffffff', line_width=4, 
                  annotation_text='Avg', annotation_position='right',
                  annotation=dict(font=dict(size=13, color='#ffffff', family='Arial Black')))
    fig.add_hline(y=avg+std, line_dash='dash', line_color='#00ffff', line_width=4,
                  annotation_text='Avg +1σ', annotation_position='right',
                  annotation=dict(font=dict(size=13, color='#00ffff', family='Arial Black')))
    
    # Only show Avg -1σ if it's within the visible range
    if avg - std >= y_min and avg - std <= y_max:
        fig.add_hline(y=avg-std, line_dash='dash', line_color='#ff00ff', line_width=4,
                      annotation_text='Avg -1σ', annotation_position='right',
                      annotation=dict(font=dict(size=13, color='#ff00ff', family='Arial Black')))
    
    # Create custom x-axis labels
    x_labels = [f"{i+1}. {str(cat)[:30]}..." if len(str(cat)) > 30 else f"{i+1}. {cat}" 
                for i, cat in enumerate(g[x_col])]
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color='#5dade2')),
        showlegend=False,
        plot_bgcolor='#111',
        paper_bgcolor='#111',
        font=dict(color='white'),
        xaxis_title=None,
        yaxis_title=metric.upper(),
        height=600,
        margin=dict(l=80, r=80, t=80, b=120),  # More spacious margins
        hovermode='closest',
        hoverlabel=dict(
            bgcolor='#222',
            font_size=12,
            font_color='white'
        )
    )
    fig.update_xaxes(
        tickmode='array',
        tickvals=list(range(len(g))),
        ticktext=x_labels,
        tickangle=-45,
        showgrid=False,
        tickfont=dict(size=10)
    )
    fig.update_yaxes(
        showgrid=True, 
        gridcolor='#333',
        range=[y_min, y_max],
        type='linear'
    )
    
    return fig

# =========================================================
# PRE-COMPUTE OPTIONS
# =========================================================
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
        'padding': '10px'
    },
    'style_cell_conditional': [
        {
            'if': {'column_id': 'url'},
            'maxWidth': '40%',
            'whiteSpace': 'normal',
            'height': 'auto',
        }
    ],
    'style_header': {
        'backgroundColor': '#111',
        'fontWeight': 'bold',
        'border': '1px solid #444',
        'color': '#17a2b8'
    },
    'style_data': {
        'border': '1px solid #444',
        'whiteSpace': 'normal',
        'height': 'auto',
    }
}
# =========================================================
# DASH APP
# =========================================================

layout = dbc.Container(fluid=True, style={'backgroundColor': '#111'}, children=[
    html.H2("CPA Insight Dashboard", style={'color': '#5dade2', 'textAlign': 'center', 'padding': '20px'}),
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id='adv_dd',  
            multi=True, 
            placeholder="Select Advertiser(s)",
            options=[{'label':x,'value':x} for x in ALL_ADVERTISERS],
            style={'color': 'black'}
            ), width=6),
        dbc.Col(dcc.Dropdown(
            id='camp_type_dd',  
            multi=True, 
            placeholder="Select Campaign Type(s)",
            options=[],  # Will be populated by callback
            style={'color': 'black'}
            ), width=4),
        dbc.Col(dcc.Dropdown(
            id='camp_dd', 
            multi=True, 
            placeholder="Select Campaign(s)",
            options=[{'label':x,'value':x} for x in ALL_CAMPAIGNS],
            style={'color': 'black'}
            ), width=6)
    ], style={'marginBottom': '20px'}),
    
    # AGGREGATED STATS
    html.Div(id='agg_stats', style={'marginBottom': '20px'}),
    
    html.Hr(style={'borderColor': '#444'}),

    # TREEMAPS - FULL WIDTH, ONE BELOW OTHER with drill-down modal
    html.H4("Concept Analysis - Treemaps (Top 10, click to drill-down)", style={'color': '#17a2b8', 'marginTop': '20px'}),
    dcc.Loading(
        dcc.Graph(id='treemap_cvr_ctr', clickData=None),
        type="circle",  # or "default", "dot", "cube"
        color="#5dade2"
    ),
    dcc.Loading(
        dcc.Graph(id='treemap_roas_cpa', clickData=None),
        type="circle",
        color="#5dade2"
    ),
    
    # Modal for drill-down
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(id='drilldown_title')),
        dbc.ModalBody([
            dcc.Graph(id='drilldown_treemap')
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="close_modal", className="ms-auto", n_clicks=0)
        ),
    ], id="drilldown_modal", size="xl", is_open=False),

    html.Hr(style={'borderColor': '#444'}),

    # CONCEPT TABLES
    html.H4("Best & Worst 10 Concepts by CVR", style={'color': '#5dade2', 'marginTop': '20px'}),
    dbc.Row([
        dbc.Col([
            html.H5("Best 10: High CVR with Good Volume", style={'color': '#00ff00'}),
            html.P("Logic: CVR > 0%, Min 10 clicks, Scored by CVR × log(clicks)", style={'color': '#aaa', 'fontSize': '11px'}),
            dash_table.DataTable(
                id='best_concepts',
                columns=[
                    {'name': 'Concept', 'id': 'concepts'},
                    {'name': 'Clicks', 'id': 'clicks'},
                    {'name': 'Conv', 'id': 'conversions'},
                    {'name': 'CVR %', 'id': 'cvr'},
                    {'name': 'Avg CVR %', 'id': 'avg_cvr'},
                    {'name': 'CVR Deviation %', 'id': 'cvr_vs_avg'},
                    {'name': 'CTR %', 'id': 'ctr'},
                    {'name': 'CPA', 'id': 'cpa'},
                    {'name': 'Mnet ROAS', 'id': 'mnet_roas'},
                    {'name': 'Adv ROAS', 'id': 'adv_roas'},
                    {'name': 'Adv Cost', 'id': 'adv_cost'},
                    {'name': 'Max Cost', 'id': 'max_cost'}
                    ],
                style_data_conditional=[
                    {'if': {'filter_query': '{cvr_vs_avg} > 0', 'column_id': 'cvr_vs_avg'}, 'color': '#00ff00', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{cvr_vs_avg} < 0', 'column_id': 'cvr_vs_avg'}, 'color': '#ff0000', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{cvr_vs_avg} > 0', 'column_id': 'cvr'}, 'color': '#00ff00'},
                    {'if': {'filter_query': '{cvr_vs_avg} < 0', 'column_id': 'cvr'}, 'color': '#ff0000'},
                    {'if': {'filter_query': '{ctr_vs_avg} > 0', 'column_id': 'ctr'}, 'color': '#00ff00'},
                    {'if': {'filter_query': '{ctr_vs_avg} < 0', 'column_id': 'ctr'}, 'color': '#ff0000'},
                    {'if': {'filter_query': '{cpa_vs_avg} < 0', 'column_id': 'cpa'}, 'color': '#00ff00'},
                    {'if': {'filter_query': '{cpa_vs_avg} > 0', 'column_id': 'cpa'}, 'color': '#ff0000'},
                    {'if': {'filter_query': '{mnet_roas_vs_avg} > 0', 'column_id': 'mnet_roas'}, 'color': '#00ff00'},
                    {'if': {'filter_query': '{mnet_roas_vs_avg} < 0', 'column_id': 'mnet_roas'}, 'color': '#ff0000'}
                    ],
                **TABLE_STYLE
            )
        ], width=12),
        dbc.Col([
            html.H5("Worst 10: High Clicks with No/Low Conversions", style={'color': '#ff0000'}),
            html.P("Logic: CVR ≤ 0.6%, Min 10 clicks, Sorted by clicks (high to low)", style={'color': '#aaa', 'fontSize': '11px'}),
            dash_table.DataTable(
                id='worst_concepts',
                columns=[
                    {'name': 'Concept', 'id': 'concepts'},
                    {'name': 'Clicks', 'id': 'clicks'},
                    {'name': 'Conv', 'id': 'conversions'},
                    {'name': 'CVR %', 'id': 'cvr'},
                    {'name': 'Avg CVR %', 'id': 'avg_cvr'},
                    {'name': 'CVR Deviation %', 'id': 'cvr_vs_avg'},
                    {'name': 'CTR %', 'id': 'ctr'},
                    {'name': 'CPA', 'id': 'cpa'},
                    {'name': 'Mnet ROAS', 'id': 'mnet_roas'},
                    {'name': 'Adv ROAS', 'id': 'adv_roas'},
                    {'name': 'Adv Cost', 'id': 'adv_cost'},
                    {'name': 'Max Cost', 'id': 'max_cost'}
                    ],
                style_data_conditional=[
                    {'if': {'filter_query': '{cvr_vs_avg} > 0', 'column_id': 'cvr_vs_avg'}, 'color': '#00ff00', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{cvr_vs_avg} < 0', 'column_id': 'cvr_vs_avg'}, 'color': '#ff0000', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{cvr_vs_avg} > 0', 'column_id': 'cvr'}, 'color': '#00ff00'},
                    {'if': {'filter_query': '{cvr_vs_avg} < 0', 'column_id': 'cvr'}, 'color': '#ff0000'},
                    {'if': {'filter_query': '{ctr_vs_avg} > 0', 'column_id': 'ctr'}, 'color': '#00ff00'},
                    {'if': {'filter_query': '{ctr_vs_avg} < 0', 'column_id': 'ctr'}, 'color': '#ff0000'},
                    {'if': {'filter_query': '{cpa_vs_avg} < 0', 'column_id': 'cpa'}, 'color': '#00ff00'},
                    {'if': {'filter_query': '{cpa_vs_avg} > 0', 'column_id': 'cpa'}, 'color': '#ff0000'},
                    {'if': {'filter_query': '{mnet_roas_vs_avg} > 0', 'column_id': 'mnet_roas'}, 'color': '#00ff00'},
                    {'if': {'filter_query': '{mnet_roas_vs_avg} < 0', 'column_id': 'mnet_roas'}, 'color': '#ff0000'}
                ],
                **TABLE_STYLE
            )
        ], width=12)
    ]),

    html.Hr(style={'borderColor': '#444'}),

    # URL TABLES - FULL WIDTH
    html.H4("Best & Worst 5 URLs by CVR", style={'color': '#5dade2', 'marginTop': '20px'}),
    html.H5("Best 5: High CVR with Good Volume", style={'color': '#00ff00'}),
    html.P("Logic: CVR > 0%, Min 10 clicks, Scored by CVR × log(clicks)", style={'color': '#aaa', 'fontSize': '11px'}),
    dash_table.DataTable(
        id='best_urls',
        columns=[
            {'name': 'URL', 'id': 'url'},
            {'name': 'Clicks', 'id': 'clicks'},
            {'name': 'Conv', 'id': 'conversions'},
            {'name': 'CVR %', 'id': 'cvr'},
            {'name': 'Avg CVR %', 'id': 'avg_cvr'},
            {'name': 'CVR Deviation %', 'id': 'cvr_vs_avg'},
            {'name': 'CTR %', 'id': 'ctr'},
            {'name': 'CPA', 'id': 'cpa'},
            {'name': 'Mnet ROAS', 'id': 'mnet_roas'},
            {'name': 'Adv ROAS', 'id': 'adv_roas'},
            {'name': 'Adv Cost', 'id': 'adv_cost'},
            {'name': 'Max Cost', 'id': 'max_cost'}
            ],
        style_data_conditional=[
            {'if': {'filter_query': '{cvr_vs_avg} > 0', 'column_id': 'cvr_vs_avg'}, 'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': '{cvr_vs_avg} < 0', 'column_id': 'cvr_vs_avg'}, 'color': '#ff0000', 'fontWeight': 'bold'},
            {'if': {'filter_query': '{cvr_vs_avg} > 0', 'column_id': 'cvr'}, 'color': '#00ff00'},
            {'if': {'filter_query': '{cvr_vs_avg} < 0', 'column_id': 'cvr'}, 'color': '#ff0000'},
            {'if': {'filter_query': '{ctr_vs_avg} > 0', 'column_id': 'ctr'}, 'color': '#00ff00'},
            {'if': {'filter_query': '{ctr_vs_avg} < 0', 'column_id': 'ctr'}, 'color': '#ff0000'},
            {'if': {'filter_query': '{cpa_vs_avg} < 0', 'column_id': 'cpa'}, 'color': '#00ff00'},
            {'if': {'filter_query': '{cpa_vs_avg} > 0', 'column_id': 'cpa'}, 'color': '#ff0000'},
            {'if': {'filter_query': '{mnet_roas_vs_avg} > 0', 'column_id': 'mnet_roas'}, 'color': '#00ff00'},
            {'if': {'filter_query': '{mnet_roas_vs_avg} < 0', 'column_id': 'mnet_roas'}, 'color': '#ff0000'}
        ],
        **TABLE_STYLE
    ),
    html.Br(),
    html.H5("Worst 5: High Clicks with No/Low Conversions", style={'color': '#ff0000'}),
    html.P("Logic: CVR ≤ 0.6%, Min 10 clicks, Sorted by clicks (high to low)", style={'color': '#aaa', 'fontSize': '11px'}),
    dash_table.DataTable(
        id='worst_urls',
        columns=[
            {'name': 'URL', 'id': 'url'},
            {'name': 'Clicks', 'id': 'clicks'},
            {'name': 'Conv', 'id': 'conversions'},
            {'name': 'CVR %', 'id': 'cvr'},
            {'name': 'Avg CVR %', 'id': 'avg_cvr'},
            {'name': 'CVR Deviation %', 'id': 'cvr_vs_avg'},
            {'name': 'CTR %', 'id': 'ctr'},
            {'name': 'CPA', 'id': 'cpa'},
            {'name': 'Mnet ROAS', 'id': 'mnet_roas'},
            {'name': 'Adv ROAS', 'id': 'adv_roas'},
            {'name': 'Adv Cost', 'id': 'adv_cost'},
            {'name': 'Max Cost', 'id': 'max_cost'}
        ],
        style_data_conditional=[
            {'if': {'filter_query': '{cvr_vs_avg} > 0', 'column_id': 'cvr_vs_avg'}, 'color': '#00ff00', 'fontWeight': 'bold'},
            {'if': {'filter_query': '{cvr_vs_avg} < 0', 'column_id': 'cvr_vs_avg'}, 'color': '#ff0000', 'fontWeight': 'bold'},
            {'if': {'filter_query': '{cvr_vs_avg} > 0', 'column_id': 'cvr'}, 'color': '#00ff00'},
            {'if': {'filter_query': '{cvr_vs_avg} < 0', 'column_id': 'cvr'}, 'color': '#ff0000'},
            {'if': {'filter_query': '{ctr_vs_avg} > 0', 'column_id': 'ctr'}, 'color': '#00ff00'},
            {'if': {'filter_query': '{ctr_vs_avg} < 0', 'column_id': 'ctr'}, 'color': '#ff0000'},
            {'if': {'filter_query': '{cpa_vs_avg} < 0', 'column_id': 'cpa'}, 'color': '#00ff00'},
            {'if': {'filter_query': '{cpa_vs_avg} > 0', 'column_id': 'cpa'}, 'color': '#ff0000'},
            {'if': {'filter_query': '{mnet_roas_vs_avg} > 0', 'column_id': 'mnet_roas'}, 'color': '#00ff00'},
            {'if': {'filter_query': '{mnet_roas_vs_avg} < 0', 'column_id': 'mnet_roas'}, 'color': '#ff0000'}
        ],
        **TABLE_STYLE
    ),

    html.Hr(style={'borderColor': '#444'}),


    # CONTEXTUALITY BUBBLE CHARTS
    # CONTEXTUALITY TABLE
    # CONTEXTUALITY TABLE
    html.H4("URL-Concept Contextuality Analysis", style={'color': '#5dade2', 'marginTop': '20px'}),
    html.P("Shows performance by contextuality level. Click any row to see top 5 URL-concept pairs.", 
       style={'color': '#aaa', 'fontSize': '12px', 'marginBottom': '10px'}),
    dash_table.DataTable(
        id='contextuality_table',
        columns=[
        {'name': 'Contextuality', 'id': 'contextuality'},
        {'name': 'Impressions', 'id': 'impressions'},
        {'name': 'Clicks', 'id': 'clicks'},
        {'name': 'Conv', 'id': 'conversions'},
        {'name': 'CVR %', 'id': 'cvr'},
        {'name': 'CTR %', 'id': 'ctr'},
        {'name': 'CPA', 'id': 'cpa'},
        {'name': 'Mnet ROAS', 'id': 'mnet_roas'},
        {'name': 'Adv ROAS', 'id': 'adv_roas'}
    ],
        row_selectable='single',
        selected_rows=[],
        **TABLE_STYLE
),

# Inline drill-down section (collapsible)
    dbc.Collapse(
        dbc.Card([
            dbc.CardHeader([
                html.H5(id='ctx_drilldown_title', style={'color': '#17a2b8', 'display': 'inline-block', 'marginBottom': '0'}),
                dbc.Button("✕ Close", id="close_ctx_drilldown", size="sm", color="secondary", 
                           style={'float': 'right'})
                ], style={'backgroundColor': '#222', 'border': '1px solid #444'}),
            dbc.CardBody([
                dash_table.DataTable(
                    id='ctx_drilldown_table',
                    columns=[
                        {'name': 'URL', 'id': 'url'},
                        {'name': 'Concept', 'id': 'concept'},
                        {'name': 'Clicks', 'id': 'clicks'},
                        {'name': 'Conv', 'id': 'conversions'},
                        {'name': 'CVR %', 'id': 'cvr'},
                        {'name': 'CTR %', 'id': 'ctr'},
                        {'name': 'CPA', 'id': 'cpa'},
                        {'name': 'Mnet ROAS', 'id': 'mnet_roas'}
                ],
                    style_data_conditional=[
                        {'if': {'filter_query': '{cvr_vs_ctx} > 0', 'column_id': 'cvr'}, 'color': '#00ff00'},
                        {'if': {'filter_query': '{cvr_vs_ctx} < 0', 'column_id': 'cvr'}, 'color': '#ff0000'},
                        {'if': {'filter_query': '{ctr_vs_ctx} > 0', 'column_id': 'ctr'}, 'color': '#00ff00'},
                        {'if': {'filter_query': '{ctr_vs_ctx} < 0', 'column_id': 'ctr'}, 'color': '#ff0000'},
                        {'if': {'filter_query': '{cpa_vs_ctx} < 0', 'column_id': 'cpa'}, 'color': '#00ff00'},
                        {'if': {'filter_query': '{cpa_vs_ctx} > 0', 'column_id': 'cpa'}, 'color': '#ff0000'},
                        {'if': {'filter_query': '{mnet_roas_vs_ctx} > 0', 'column_id': 'mnet_roas'}, 'color': '#00ff00'},
                        {'if': {'filter_query': '{mnet_roas_vs_ctx} < 0', 'column_id': 'mnet_roas'}, 'color': '#ff0000'}
                        ],
                    **TABLE_STYLE
                    )
                ], style={'backgroundColor': '#222', 'padding': '15px'})
            ], style={'marginTop': '10px', 'marginBottom': '20px'}),
        id="ctx_drilldown_collapse",
        is_open=False
        ),
    html.Hr(style={'borderColor': '#444'}),
    # BUBBLE CHARTS - Sprig URL & Domain (Full) - 2x2
    html.Hr(style={'borderColor': '#444'}),
    html.H4("Sprig Categories Analysis", style={'color': '#5dade2', 'marginTop': '20px'}),
    dbc.Row([
        dbc.Col(html.Label("Number of items to show in treemaps:", style={'color': 'white'}), width=3),
        dbc.Col(dcc.Dropdown(
            id='sprig_count', 
            value=10,
            options=[{'label':f'Top {x}','value':x} for x in [5, 10, 15, 20, 30]],
            style={'color': 'black'}
            ), width=2)
        ], style={'marginBottom': '15px'}),

    html.Hr(style={'borderColor': '#444'}),
    html.H4("Sprig URL Top Categories - Treemaps (Top 10, click to drill-down to full category)", style={'color': '#17a2b8', 'marginTop': '20px'}),
    dcc.Loading(dcc.Graph(id='treemap_url_top_cvr_ctr'), type="circle", color="#5dade2"),
    dcc.Loading(dcc.Graph(id='treemap_url_top_roas_cpa'), type="circle", color="#5dade2"),
    html.Hr(style={'borderColor': '#444'}),
    html.H4("Sprig Domain Top Categories - Treemaps (Top 10, click to drill-down to full category)", style={'color': '#17a2b8', 'marginTop': '20px'}),
    dcc.Loading(dcc.Graph(id='treemap_dom_top_cvr_ctr'), type="circle", color="#5dade2"),
    dcc.Loading(dcc.Graph(id='treemap_dom_top_roas_cpa'), type="circle", color="#5dade2"),
    html.Hr(style={'borderColor': '#444'}),
    html.H4("Sprig URL Final Categories - Treemaps (Top 10, click to drill-down to full category)", style={'color': '#17a2b8', 'marginTop': '20px'}),
    dcc.Loading(dcc.Graph(id='treemap_url_final_cvr_ctr'), type="circle", color="#5dade2"),
    dcc.Loading(dcc.Graph(id='treemap_url_final_roas_cpa'), type="circle", color="#5dade2"),
    html.Hr(style={'borderColor': '#444'}),
    html.H4("Sprig Domain Final Categories - Treemaps (Top 10, click to drill-down to full category)", style={'color': '#17a2b8', 'marginTop': '20px'}),
    dcc.Loading(dcc.Graph(id='treemap_dom_final_cvr_ctr'), type="circle", color="#5dade2"),
    dcc.Loading(dcc.Graph(id='treemap_dom_final_roas_cpa'), type="circle", color="#5dade2"),
    html.Hr(style={'borderColor': '#444'}),

    # BUBBLE CHARTS - Sprig URL & Domain (Top) - 2x2
])

# =========================================================
# CALLBACKS
# =========================================================
@callback(
    [Output('camp_type_dd', 'options'),
     Output('camp_dd', 'options')],
    [Input('adv_dd', 'value'),
     Input('camp_type_dd', 'value')]
)
def update_filters(advs, camp_types):
    # Filter for Campaign Type dropdown
    d = df.copy()
    if advs:
        d = d[d['advertiser'].isin(advs)]
    campaign_types = sorted(d['campaign_type'].dropna().unique())
    
    # Filter for Campaign dropdown
    if camp_types:
        d = d[d['campaign_type'].isin(camp_types)]
    campaigns = sorted(d['campaign'].dropna().unique())
    
    return (
        [{'label': x, 'value': x} for x in campaign_types],
        [{'label': x, 'value': x} for x in campaigns]
    )

@callback(
    [Output('agg_stats', 'children'),
     Output('treemap_cvr_ctr', 'figure'),
     Output('treemap_roas_cpa', 'figure'),
     Output('best_concepts', 'data'),
     Output('worst_concepts', 'data'),
     Output('best_urls', 'data'),
     Output('worst_urls', 'data'),
     Output('contextuality_table', 'data'),
     Output('treemap_url_top_cvr_ctr', 'figure'),
     Output('treemap_url_top_roas_cpa', 'figure'),
     Output('treemap_dom_top_cvr_ctr', 'figure'),
     Output('treemap_dom_top_roas_cpa', 'figure'),
     Output('treemap_url_final_cvr_ctr', 'figure'),
     Output('treemap_url_final_roas_cpa', 'figure'),
     Output('treemap_dom_final_cvr_ctr', 'figure'),
     Output('treemap_dom_final_roas_cpa', 'figure')],
    [Input('adv_dd','value'),
     Input('camp_type_dd','value'),
     Input('camp_dd','value'),
     Input('sprig_count','value')]
)
def update_all(advs, camp_types, camps, sprig_count):    
    try:
        # Default counts if not set
        if sprig_count is None:
            sprig_count = 5
            
        # Filter data
        d = df.copy()
        d_concepts = df_concepts.copy()
        d_contextuality = df_contextuality.copy()
        
        if advs:
            d = d[d['advertiser'].isin(advs)]
            d_concepts = d_concepts[d_concepts['advertiser'].isin(advs)]
            d_contextuality = d_contextuality[d_contextuality['advertiser'].isin(advs)]
        if camp_types:
            d = d[d['campaign_type'].isin(camp_types)]
            d_concepts = d_concepts[d_concepts['campaign_type'].isin(camp_types)]
            d_contextuality = d_contextuality[d_contextuality['campaign_type'].isin(camp_types)]
        if camps:
            d = d[d['campaign'].isin(camps)]
            d_concepts = d_concepts[d_concepts['campaign'].isin(camps)]
            d_contextuality = d_contextuality[d_contextuality['campaign'].isin(camps)]

        # Calculate aggregated stats for filtered data
        total_clicks = d['clicks'].sum()
        total_impressions = d['impressions'].sum()
        total_conversions = d['conversions'].sum()
        total_adv_cost = d['adv_cost'].sum()
        total_max_cost = d['max_cost'].sum()
        total_adv_value = d['adv_value'].sum()
        
        agg_cvr = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
        agg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        agg_cpa = (total_adv_cost / total_conversions) if total_conversions > 0 else 0
        agg_mnet_roas = d['mnet_roas'].mean()
        agg_adv_roas = d['adv_roas'].mean()
        
        # Create stats display
        # Create stats display
        stats_display = dbc.Card([
            dbc.CardBody([
                html.H4("Aggregated Stats", style={'color': '#5dade2', 'marginBottom': '20px', 'fontSize': '22px', 'fontWeight': 'bold'}),        dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Strong("Clicks: ", style={'color': '#aaa'}),
                            html.Span(f"{int(total_clicks):,}", style={'color': '#5dade2', 'fontSize': '18px'})
                        ])
                    ], width=2),
                    dbc.Col([
                        html.Div([
                            html.Strong("Conversions: ", style={'color': '#aaa'}),
                            html.Span(f"{total_conversions:.2f}", style={'color': '#5dade2', 'fontSize': '18px'})
                        ])
                    ], width=2),
                    dbc.Col([
                        html.Div([
                            html.Strong("CVR: ", style={'color': '#aaa'}),
                            html.Span(f"{agg_cvr:.2f}%", style={'color': '#00ff00', 'fontSize': '18px'})
                        ])
                    ], width=2),
                    dbc.Col([
                        html.Div([
                            html.Strong("CTR: ", style={'color': '#aaa'}),
                            html.Span(f"{agg_ctr:.2f}%", style={'color': '#00ff00', 'fontSize': '18px'})
                        ])
                    ], width=2),
                    dbc.Col([
                        html.Div([
                            html.Strong("CPA: ", style={'color': '#aaa'}),
                            html.Span(f"${agg_cpa:.2f}", style={'color': '#ffcc00', 'fontSize': '18px'})
                        ])
                    ], width=2),
                    dbc.Col([
                        html.Div([
                            html.Strong("ROAS: ", style={'color': '#aaa'}),
                            html.Span(f"{agg_mnet_roas:.2f}", style={'color': '#00ff00', 'fontSize': '18px'})
                        ])
                    ], width=2)
                ])
            ])
        ], style={'backgroundColor': '#222', 'border': '1px solid #444', 'marginBottom': '30px'})

        # Empty figure template
        empty = go.Figure()
        empty.update_layout(title="No data", plot_bgcolor='#111', paper_bgcolor='#111', font=dict(color='white'))

        # Concept aggregation
        # Concept aggregation
        g_concept = weighted_aggregate(d_concepts, 'concepts')
        
        # Contextuality aggregation
        g_contextuality = weighted_aggregate(d_contextuality, 'contextuality')
        
        # Treemaps: CVR-CTR (sorted by CTR desc, colored by CVR) - TOP 10 DEFAULT
        # Get top 10 concepts by clicks first
        top_concepts = g_concept.nlargest(10, 'clicks')['concepts'].tolist()
        g_concept_top = g_concept[g_concept['concepts'].isin(top_concepts)]

# Treemaps: CVR-CTR (sorted by CTR desc, colored by CVR) - TOP 10 DEFAULT
        avg_metrics = {'cvr': agg_cvr, 'ctr': agg_ctr, 'cpa': agg_cpa, 'mnet_roas': agg_mnet_roas}
        tree_cvr_ctr = create_treemap(g_concept_top, 'cvr', 'ctr', 
                               'CVR vs CTR - Top 10 by Clicks (sorted by CTR ↓, colored by CVR, click to drill-down)', 
                               show_cvr_ctr=True, top_n=10, col_name='concepts', avg_metrics=avg_metrics)
# Treemaps: ROAS-CPA (sorted by CPA desc, colored by ROAS) - TOP 10 DEFAULT
        tree_roas_cpa = create_treemap(g_concept_top, 'mnet_roas', 'cpa', 
                                'ROAS vs CPA - Top 10 by Clicks (sorted by CPA ↓, colored by ROAS, click to drill-down)', 
                                show_cvr_ctr=False, top_n=10, col_name='concepts', avg_metrics=avg_metrics)
        
        # Concept tables with smart scoring
        all_concepts = g_concept.dropna(subset=['cvr']).copy()
        all_concepts['cvr_vs_avg'] = ((all_concepts['cvr'] - agg_cvr) / agg_cvr * 100).round(1) if agg_cvr > 0 else 0
        
        # BEST: Only non-zero CVR with min 10 clicks, scored by CVR * log(clicks)
        best_candidates = all_concepts[(all_concepts['clicks'] >= 10) & (all_concepts['cvr'] > 0)].copy()
        if len(best_candidates) == 0:
            best_candidates = all_concepts[all_concepts['cvr'] > 0].copy()
        
        if len(best_candidates) > 0:
            best_candidates['score'] = best_candidates['cvr'] * np.log1p(best_candidates['clicks'])
            best_concepts_df = best_candidates.sort_values('score', ascending=False).head(10)
            best_concept_ids = set(best_concepts_df['concepts'].tolist())
        else:
            best_concepts_df = pd.DataFrame()
            best_concept_ids = set()
        
        # WORST: CVR < 0.6% OR CVR = 0, with min 10 clicks, sorted by clicks descending
        # EXCLUDE concepts that are in the best list
        worst_candidates = all_concepts[
            (all_concepts['clicks'] >= 10) & 
            (all_concepts['cvr'] <= 0.6) &
            (~all_concepts['concepts'].isin(best_concept_ids))
        ].copy()
        if len(worst_candidates) == 0:
            worst_candidates = all_concepts[
                (all_concepts['cvr'] <= 0.6) &
                (~all_concepts['concepts'].isin(best_concept_ids))
            ].copy()
        
        if len(worst_candidates) > 0:
            worst_concepts_df = worst_candidates.sort_values('clicks', ascending=False).head(10)
        else:
            worst_concepts_df = pd.DataFrame()
        
        # Add avg CVR column and format for display
        best_concepts_display = best_concepts_df.copy() if len(best_concepts_df) > 0 else pd.DataFrame()
        worst_concepts_display = worst_concepts_df.copy() if len(worst_concepts_df) > 0 else pd.DataFrame()
        
        if len(best_concepts_display) > 0:
            best_concepts_display['avg_cvr'] = agg_cvr
            best_concepts_display['ctr_vs_avg'] = best_concepts_display['ctr'] - agg_ctr
            best_concepts_display['cpa_vs_avg'] = best_concepts_display['cpa'] - agg_cpa
            best_concepts_display['mnet_roas_vs_avg'] = best_concepts_display['mnet_roas'] - agg_mnet_roas
            best_concepts = best_concepts_display[['concepts', 'clicks', 'conversions', 'cvr', 'avg_cvr', 'cvr_vs_avg', 'ctr', 'ctr_vs_avg', 'cpa', 'cpa_vs_avg', 'mnet_roas', 'mnet_roas_vs_avg', 'adv_roas', 'adv_cost', 'max_cost']].round(2).to_dict('records')
        else:
            best_concepts = []
            
        if len(worst_concepts_display) > 0:
            worst_concepts_display['avg_cvr'] = agg_cvr
            worst_concepts_display['ctr_vs_avg'] = worst_concepts_display['ctr'] - agg_ctr
            worst_concepts_display['cpa_vs_avg'] = worst_concepts_display['cpa'] - agg_cpa
            worst_concepts_display['mnet_roas_vs_avg'] = worst_concepts_display['mnet_roas'] - agg_mnet_roas
            worst_concepts = worst_concepts_display[['concepts', 'clicks', 'conversions', 'cvr', 'avg_cvr', 'cvr_vs_avg', 'ctr', 'ctr_vs_avg', 'cpa', 'cpa_vs_avg', 'mnet_roas', 'mnet_roas_vs_avg', 'adv_roas', 'adv_cost', 'max_cost']].round(2).to_dict('records')
        else:
            worst_concepts = []
        
        # URL aggregation
        g_url = weighted_aggregate(d, 'url')
        
        # URL tables - same logic as concepts
        all_urls = g_url.dropna(subset=['cvr']).copy()
        all_urls['cvr_vs_avg'] = ((all_urls['cvr'] - agg_cvr) / agg_cvr * 100).round(1) if agg_cvr > 0 else 0
        
        # BEST URLs: Only non-zero CVR
        best_url_candidates = all_urls[(all_urls['clicks'] >= 10) & (all_urls['cvr'] > 0)].copy()
        if len(best_url_candidates) == 0:
            best_url_candidates = all_urls[all_urls['cvr'] > 0].copy()
        
        if len(best_url_candidates) > 0:
            best_url_candidates['score'] = best_url_candidates['cvr'] * np.log1p(best_url_candidates['clicks'])
            best_urls_df = best_url_candidates.sort_values('score', ascending=False).head(5)
            best_url_ids = set(best_urls_df['url'].tolist())
        else:
            best_urls_df = pd.DataFrame()
            best_url_ids = set()
        
        # WORST URLs: CVR < 0.6% OR CVR = 0, high clicks, EXCLUDE best URLs
        worst_url_candidates = all_urls[
            (all_urls['clicks'] >= 10) & 
            (all_urls['cvr'] <= 0.6) &
            (~all_urls['url'].isin(best_url_ids))
        ].copy()
        if len(worst_url_candidates) == 0:
            worst_url_candidates = all_urls[
                (all_urls['cvr'] <= 0.6) &
                (~all_urls['url'].isin(best_url_ids))
            ].copy()
        
        if len(worst_url_candidates) > 0:
            worst_urls_df = worst_url_candidates.sort_values('clicks', ascending=False).head(5)
        else:
            worst_urls_df = pd.DataFrame()
        
        # Add avg CVR column and format for display
        best_urls_display = best_urls_df.copy() if len(best_urls_df) > 0 else pd.DataFrame()
        worst_urls_display = worst_urls_df.copy() if len(worst_urls_df) > 0 else pd.DataFrame()
        
        if len(best_urls_display) > 0:
            best_urls_display['avg_cvr'] = agg_cvr
            best_urls_display['ctr_vs_avg'] = best_urls_display['ctr'] - agg_ctr
            best_urls_display['cpa_vs_avg'] = best_urls_display['cpa'] - agg_cpa
            best_urls_display['mnet_roas_vs_avg'] = best_urls_display['mnet_roas'] - agg_mnet_roas
            best_urls = best_urls_display[['url', 'clicks', 'conversions', 'cvr', 'avg_cvr', 'cvr_vs_avg', 'ctr', 'ctr_vs_avg', 'cpa', 'cpa_vs_avg', 'mnet_roas', 'mnet_roas_vs_avg', 'adv_roas', 'adv_cost', 'max_cost']].round(2).to_dict('records')
        else:
            best_urls = []
            
        if len(worst_urls_display) > 0:
            worst_urls_display['avg_cvr'] = agg_cvr
            worst_urls_display['ctr_vs_avg'] = worst_urls_display['ctr'] - agg_ctr
            worst_urls_display['cpa_vs_avg'] = worst_urls_display['cpa'] - agg_cpa
            worst_urls_display['mnet_roas_vs_avg'] = worst_urls_display['mnet_roas'] - agg_mnet_roas
            worst_urls = worst_urls_display[['url', 'clicks', 'conversions', 'cvr', 'avg_cvr', 'cvr_vs_avg', 'ctr', 'ctr_vs_avg', 'cpa', 'cpa_vs_avg', 'mnet_roas', 'mnet_roas_vs_avg', 'adv_roas', 'adv_cost', 'max_cost']].round(2).to_dict('records')
        else:
            worst_urls = []
        
        # Sprig aggregations
        g_url_top = weighted_aggregate(d, 'sprig_url_top')
        g_dom_top = weighted_aggregate(d, 'sprig_domain_top')
        g_url_final = weighted_aggregate(d, 'sprig_url_final')
        g_dom_final = weighted_aggregate(d, 'sprig_domain_final')
        
        # Hover maps
        h_contextuality = top3_urls(d_contextuality, 'contextuality')
        h_url_top = top3_urls(d, 'sprig_url_top')
        h_dom_top = top3_urls(d, 'sprig_domain_top')
        h_url_final = top3_urls(d, 'sprig_url_final')
        h_dom_final = top3_urls(d, 'sprig_domain_final')
        
        # Create all bubble charts with top_n parameter
        metrics = ['cvr', 'ctr', 'cpa', 'mnet_roas']
        
        # Show all contextuality items (no top_n limit)
        # Generate contextuality table data with drilldown arrows
        if len(g_contextuality) > 0:
            ctx_table_data = g_contextuality.copy()
    # Add arrow to contextuality value
            ctx_table_data['contextuality'] = '▶ ' + ctx_table_data['contextuality'].astype(str)
            ctx_table_data = ctx_table_data[['contextuality', 'impressions', 'clicks', 'conversions', 'cvr', 'ctr', 'cpa', 'mnet_roas', 'adv_roas']].round(2).to_dict('records')
        else:
            ctx_table_data = []
        # Create treemaps for Sprig categories
        tree_url_top_cvr_ctr = create_treemap(g_url_top, 'cvr', 'ctr', 
                               'Sprig URL Top: CVR vs CTR (sorted by CTR ↓, colored by CVR)', 
                               show_cvr_ctr=True, top_n=10, col_name='sprig_url_top', avg_metrics=avg_metrics)
        tree_url_top_roas_cpa = create_treemap(g_url_top, 'mnet_roas', 'cpa', 
                               'Sprig URL Top: ROAS vs CPA (sorted by CPA ↓, colored by ROAS)', 
                               show_cvr_ctr=False, top_n=10, col_name='sprig_url_top', avg_metrics=avg_metrics)
        tree_dom_top_cvr_ctr = create_treemap(g_dom_top, 'cvr', 'ctr', 
                               'Sprig Domain Top: CVR vs CTR (sorted by CTR ↓, colored by CVR)', 
                               show_cvr_ctr=True, top_n=10, col_name='sprig_domain_top', avg_metrics=avg_metrics)
        tree_dom_top_roas_cpa = create_treemap(g_dom_top, 'mnet_roas', 'cpa', 
                               'Sprig Domain Top: ROAS vs CPA (sorted by CPA ↓, colored by ROAS)', 
                               show_cvr_ctr=False, top_n=10, col_name='sprig_domain_top', avg_metrics=avg_metrics)
        tree_url_final_cvr_ctr = create_treemap(g_url_final, 'cvr', 'ctr', 
                               'Sprig URL Final: CVR vs CTR (sorted by CTR ↓, colored by CVR)', 
                               show_cvr_ctr=True, top_n=10, col_name='sprig_url_final', avg_metrics=avg_metrics)
        tree_url_final_roas_cpa = create_treemap(g_url_final, 'mnet_roas', 'cpa', 
                               'Sprig URL Final: ROAS vs CPA (sorted by CPA ↓, colored by ROAS)', 
                               show_cvr_ctr=False, top_n=10, col_name='sprig_url_final', avg_metrics=avg_metrics)
        tree_dom_final_cvr_ctr = create_treemap(g_dom_final, 'cvr', 'ctr', 
                               'Sprig Domain Final: CVR vs CTR (sorted by CTR ↓, colored by CVR)', 
                               show_cvr_ctr=True, top_n=10, col_name='sprig_domain_final', avg_metrics=avg_metrics)
        tree_dom_final_roas_cpa = create_treemap(g_dom_final, 'mnet_roas', 'cpa', 
                               'Sprig Domain Final: ROAS vs CPA (sorted by CPA ↓, colored by ROAS)', 
                               show_cvr_ctr=False, top_n=10, col_name='sprig_domain_final', avg_metrics=avg_metrics)
        return (
            stats_display,
            tree_cvr_ctr, tree_roas_cpa,
            best_concepts, worst_concepts,
            best_urls, worst_urls,
            ctx_table_data,  # Contextuality tables
            tree_url_top_cvr_ctr, tree_url_top_roas_cpa,
            tree_dom_top_cvr_ctr, tree_dom_top_roas_cpa,
            tree_url_final_cvr_ctr, tree_url_final_roas_cpa,
            tree_dom_final_cvr_ctr, tree_dom_final_roas_cpa)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        empty = go.Figure()
        empty.update_layout(title=f"Error: {str(e)}", plot_bgcolor='#111', paper_bgcolor='#111', font=dict(color='white'))
        empty_stats = html.Div("Error loading stats", style={'color': 'red'})
    
    # Return correct number of values: 1 stats + 2 treemaps + 4 tables + 12 figures = 19 total
        return (
            empty_stats,  # 1: agg_stats
            empty, empty,  # 2: treemap_cvr_ctr, treemap_roas_cpa
            [], [], [], [],  # 4: best_concepts, worst_concepts, best_urls, worst_urls (tables)
            [],  # 1: contextuality_table  # 4: ctx_cvr, ctx_ctr, ctx_cpa, ctx_mnet_roas
            empty, empty,  # 2: treemap_url_top_cvr_ctr, treemap_url_top_roas_cpa
            empty, empty,  # 2: treemap_dom_top_cvr_ctr, treemap_dom_top_roas_cpa
            empty, empty,  # 2: treemap_url_final_cvr_ctr, treemap_url_final_roas_cpa
            empty, empty   # 2: treemap_dom_final_cvr_ctr, treemap_dom_final_roas_cpa
    )
    
@callback(
    [Output("drilldown_modal", "is_open"),
     Output("drilldown_title", "children"),
     Output("drilldown_treemap", "figure"),
     Output("treemap_cvr_ctr", "clickData"),
     Output("treemap_roas_cpa", "clickData"),
     Output("treemap_url_top_cvr_ctr", "clickData"),
     Output("treemap_url_top_roas_cpa", "clickData"),
     Output("treemap_dom_top_cvr_ctr", "clickData"),
     Output("treemap_dom_top_roas_cpa", "clickData"),
     Output("treemap_url_final_cvr_ctr", "clickData"),
     Output("treemap_url_final_roas_cpa", "clickData"),
     Output("treemap_dom_final_cvr_ctr", "clickData"),
     Output("treemap_dom_final_roas_cpa", "clickData")],
    [Input("treemap_cvr_ctr", "clickData"),
     Input("treemap_roas_cpa", "clickData"),
     Input("treemap_url_top_cvr_ctr", "clickData"),
     Input("treemap_url_top_roas_cpa", "clickData"),
     Input("treemap_dom_top_cvr_ctr", "clickData"),
     Input("treemap_dom_top_roas_cpa", "clickData"),
     Input("treemap_url_final_cvr_ctr", "clickData"),
     Input("treemap_url_final_roas_cpa", "clickData"),
     Input("treemap_dom_final_cvr_ctr", "clickData"),
     Input("treemap_dom_final_roas_cpa", "clickData"),
     Input("close_modal", "n_clicks"),
     Input('adv_dd','value'),
     Input('camp_type_dd','value'),
     Input('camp_dd','value')],    
    [State("drilldown_modal", "is_open")],
    prevent_initial_call=True
)
def handle_treemap_click(click_cvr, click_roas, click_url_top_cvr, click_url_top_roas, 
                        click_dom_top_cvr, click_dom_top_roas, click_url_final_cvr, 
                        click_url_final_roas, click_dom_final_cvr, click_dom_final_roas,
                        close_clicks, advs, camp_types, camps, is_open):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return False, "", go.Figure(), None, None, None, None, None, None, None, None, None, None
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Close modal and clear clickData when close button clicked
    if trigger_id == "close_modal":
        return False, "", go.Figure(), None, None, None, None, None, None, None, None, None, None
    
    # Don't reopen if modal is already open
    if is_open:
        return False, "", go.Figure(), None, None, None, None, None, None, None, None, None, None
    
    # Determine which treemap was clicked and what to drill down to
    click_data = None
    drill_col = None
    drill_to_col = None
    
    if trigger_id == "treemap_cvr_ctr":
        click_data = click_cvr
        drill_col = 'concepts'
        drill_to_col = 'domain'
    elif trigger_id == "treemap_roas_cpa":
        click_data = click_roas
        drill_col = 'concepts'
        drill_to_col = 'domain'
    elif trigger_id == "treemap_url_top_cvr_ctr":
        click_data = click_url_top_cvr
        drill_col = 'sprig_url_top'
        drill_to_col = 'sprig_url'
    elif trigger_id == "treemap_url_top_roas_cpa":
        click_data = click_url_top_roas
        drill_col = 'sprig_url_top'
        drill_to_col = 'sprig_url'
    elif trigger_id == "treemap_dom_top_cvr_ctr":
        click_data = click_dom_top_cvr
        drill_col = 'sprig_domain_top'
        drill_to_col = 'sprig_domain'
    elif trigger_id == "treemap_dom_top_roas_cpa":
        click_data = click_dom_top_roas
        drill_col = 'sprig_domain_top'
        drill_to_col = 'sprig_domain'
    elif trigger_id == "treemap_url_final_cvr_ctr":
        click_data = click_url_final_cvr
        drill_col = 'sprig_url_final'
        drill_to_col = 'sprig_url'
    elif trigger_id == "treemap_url_final_roas_cpa":
        click_data = click_url_final_roas
        drill_col = 'sprig_url_final'
        drill_to_col = 'sprig_url'
    elif trigger_id == "treemap_dom_final_cvr_ctr":
        click_data = click_dom_final_cvr
        drill_col = 'sprig_domain_final'
        drill_to_col = 'sprig_domain'
    elif trigger_id == "treemap_dom_final_roas_cpa":
        click_data = click_dom_final_roas
        drill_col = 'sprig_domain_final'
        drill_to_col = 'sprig_domain'
    
    if not click_data or 'points' not in click_data or len(click_data['points']) == 0:
        return False, "", go.Figure(), None, None, None, None, None, None, None, None, None, None
    
    clicked_value = click_data['points'][0].get('label', '')
    
    if not clicked_value:
        return False, "", go.Figure(), None, None, None, None, None, None, None, None, None, None
    
    # Filter data
    d = df.copy()
    if advs:
        d = d[d['advertiser'].isin(advs)]
    if camp_types:
        d = d[d['campaign_type'].isin(camp_types)]
    if camps:
        d = d[d['campaign'].isin(camps)]
    
    # Filter based on the clicked value
    if drill_col == 'concepts':
        # For concepts, filter rows containing this concept
        d = d[d['concepts'].apply(lambda x: clicked_value in x if isinstance(x, list) else False)]
    else:
        # For sprig categories, filter directly
        d = d[d[drill_col] == clicked_value]
    
    # Aggregate by the drill-to column
    g_drill = weighted_aggregate(d, drill_to_col)
    
    if len(g_drill) == 0:
        fig = go.Figure()
        fig.update_layout(
            title=f"No data found for: {clicked_value}",
            plot_bgcolor='#111',
            paper_bgcolor='#111',
            font=dict(color='white')
        )
        return True, f"Drill-down for: {clicked_value}", fig, None, None, None, None, None, None, None, None, None, None
    
    # Calculate aggregated stats
    total_clicks = d['clicks'].sum()
    total_conversions = d['conversions'].sum()
    agg_cvr = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
    
    avg_metrics = {'cvr': agg_cvr}
    
    # Create treemap
    fig = create_treemap(
        g_drill, 
        'cvr', 
        'clicks', 
        f'Drill-down: "{clicked_value}" → {drill_to_col}',
        show_cvr_ctr=True,
        top_n=15,
        col_name=drill_to_col,
        avg_metrics=avg_metrics
    )
    
    # Return with all clickData set to None to clear them
    return True, f"Drill-down for: {clicked_value}", fig, None, None, None, None, None, None, None, None, None, None
@callback(
    [Output("ctx_drilldown_collapse", "is_open"),
     Output("ctx_drilldown_title", "children"),
     Output("ctx_drilldown_table", "data"),
     Output("contextuality_table", "selected_rows")],
    [Input("contextuality_table", "selected_rows"),
     Input("close_ctx_drilldown", "n_clicks"),     
     Input('adv_dd','value'),
     Input('camp_type_dd','value'),
     Input('camp_dd','value')],
    [State("contextuality_table", "data")],
    prevent_initial_call=True
)
def handle_contextuality_drilldown(selected_rows, close_clicks, advs, camp_types, camps, table_data):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return False, "", [], []
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Close drill-down
    if trigger_id == "close_ctx_drilldown":
        return False, "", [], []
    
    # Open drill-down
    if trigger_id == "contextuality_table" and selected_rows and len(selected_rows) > 0:
        row_idx = selected_rows[0]
        contextuality_value = table_data[row_idx]['contextuality'].replace('▶ ', '')  # Remove arrow
        ctx_cvr = table_data[row_idx]['cvr']
        
        # Filter data
        d = df.copy()
        if advs:
            d = d[d['advertiser'].isin(advs)]
        if camp_types:
            d = d[d['campaign_type'].isin(camp_types)]
        if camps:
            d = d[d['campaign'].isin(camps)]
        
        # Filter by contextuality
        d = d[d['contextuality'] == contextuality_value].copy()
        
        # Explode concepts and aggregate by URL-concept pairs
        d_exploded = d.explode('concepts').dropna(subset=['concepts'])
        
        pairs = d_exploded.groupby(['url', 'concepts']).agg({
            'clicks': 'sum',
            'impressions': 'sum',
            'conversions': 'sum',
            'adv_cost': 'sum',
            'max_cost': 'sum',
            'adv_value': 'sum',
            'mnet_roas': 'mean'
        }).reset_index()
        
        # Calculate metrics
        pairs['cvr'] = np.where(pairs['clicks']>0, 100*pairs['conversions']/pairs['clicks'], np.nan)
        pairs['ctr'] = np.where(pairs['impressions']>0, 100*pairs['clicks']/pairs['impressions'], np.nan)
        pairs['cpa'] = np.where(pairs['conversions']>0, pairs['adv_cost']/pairs['conversions'], np.nan)
        
        # Get top 5 by clicks
        top_pairs = pairs.nlargest(5, 'clicks').copy()
        
        # Calculate comparison columns for color coding
        top_pairs['cvr_vs_ctx'] = top_pairs['cvr'] - ctx_cvr
        top_pairs['ctr_vs_ctx'] = top_pairs['ctr'] - table_data[row_idx]['ctr']
        top_pairs['cpa_vs_ctx'] = top_pairs['cpa'] - table_data[row_idx]['cpa']
        top_pairs['mnet_roas_vs_ctx'] = top_pairs['mnet_roas'] - table_data[row_idx]['mnet_roas']
        
        # Rename concepts column to concept
        top_pairs = top_pairs.rename(columns={'concepts': 'concept'})
        
        # Format for display
        drilldown_data = top_pairs[['url', 'concept', 'clicks', 'conversions', 'cvr', 'cvr_vs_ctx', 'ctr', 'ctr_vs_ctx', 'cpa', 'cpa_vs_ctx', 'mnet_roas', 'mnet_roas_vs_ctx']].round(2).to_dict('records')
        
        return True, f"Top 5 URL-Concept Pairs for: {contextuality_value}", drilldown_data, selected_rows
    
    return False, "", [], []

# =========================================================
# RUN
# =========================================================
