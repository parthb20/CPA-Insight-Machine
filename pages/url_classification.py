import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import re
from collections import Counter
from itertools import combinations
from functools import lru_cache
import dash
from dash import dcc, html, Input, Output, dash_table, State, register_page, callback
import io
import requests
import plotly.express as px
import plotly.graph_objects as go
GDRIVE_FILE_ID = "1bRlaGhB2m_NNugf0iSjYFkff1kmqBytu"

# Convert to direct download link
GDRIVE_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

@lru_cache(maxsize=1)
def load_data_cached():
    """Load data once and cache it"""
    try:
        response = requests.get(GDRIVE_URL)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text))
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

# Replace line ~41 with:
df = load_data_cached()

register_page(__name__, path='/', name='URL')

# [REST OF YOUR CURRENT CODE - just remove the app.run() at bottom]
# ... all your existing code ...



# =========================================================
# CONFIG
# =========================================================
# =========================================================
# CONFIG & LOAD DATA FROM GOOGLE DRIVE
# =========================================================


# Google Drive file ID - REPLACE THIS with your actual file ID

# =========================================================
# LOAD DATA
# =========================================================


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
for col in ['advertiser', 'campaign_type', 'campaign', 'domain', 
            'sprig_url_top', 'sprig_url_final', 'sprig_domain_top', 'sprig_domain_final']:
    if col in df.columns:
        df[col] = df[col].astype('category')
for c in ['clicks','impressions','conversions','adv_cost','max_cost','adv_value','mnet_roas','adv_roas']:
    if c not in df.columns:
        df[c] = 0
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)


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
# PRE-FILTER: Only keep concepts that appear in at least 5 URLs
from collections import Counter
all_concepts = [c for concepts in df['concepts'] for c in concepts]
concept_counts = Counter(all_concepts)
valid_concepts = {c for c, count in concept_counts.items() if count >= 5}

# Filter concepts to only valid ones
df['concepts'] = df['concepts'].apply(lambda x: [c for c in x if c in valid_concepts])

# Explode concepts for aggregation
df_concepts = df.explode('concepts').dropna(subset=['concepts'])
# Use the actual contextuality column from CSV
# Map the column name first
if 'URL-Concept Contextuality' in df.columns and 'contextuality' not in df.columns:
    df['contextuality'] = df['URL-Concept Contextuality']

df_contextuality = df[df['contextuality'].notna()].copy()
    
def filter_dataframe(d, advs, camp_types, camps):
    mask = pd.Series(True, index=d.index)  # Start with all True
    if advs:
        mask &= d['advertiser'].isin(advs)
    if camp_types:
        mask &= d['campaign_type'].isin(camp_types)
    if camps:
        mask &= d['campaign'].isin(camps)
    return d[mask]  # No .copy()

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
    # Filter to top 10 by clicks BEFORE passing to create_treemap
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
    min_val = g[metric_color].min()
    max_val = g[metric_color].max()
    
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
    color_min = max(0, min_val - 0.2 * std)  # Slightly below minimum
    color_max = max_val + 0.2 * std
    fig = go.Figure(go.Treemap(
        labels=g[label_col],
        parents=[''] * len(g),
        values=g['clicks'],
        marker=dict(
        colorscale=[[0, '#cc0000'], [0.5, '#ffcc00'], [1, '#00cc00']],
        cmid=avg,
        cmin=color_min,  # Red at minimum
        cmax=color_max,
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
    html.P("Shows performance by contextuality level. Click any row to see top 5 URL-Campaign pairs.", 
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
    html.Div(id="ctx_drilldown_section", style={'marginTop': '10px'}),
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
    d = df
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

def create_stats_display(total_clicks, total_conversions, agg_cvr, agg_ctr, agg_cpa, agg_mnet_roas):
    """Extract stats display creation"""
    return dbc.Card([
        dbc.CardBody([
            html.H4("Aggregated Stats", style={'color': '#5dade2', 'marginBottom': '20px', 'fontSize': '22px', 'fontWeight': 'bold'}),
            dbc.Row([
                dbc.Col([html.Div([html.Strong("Clicks: ", style={'color': '#aaa'}), html.Span(f"{int(total_clicks):,}", style={'color': '#5dade2', 'fontSize': '18px'})])], width=2),
                dbc.Col([html.Div([html.Strong("Conversions: ", style={'color': '#aaa'}), html.Span(f"{total_conversions:.2f}", style={'color': '#5dade2', 'fontSize': '18px'})])], width=2),
                dbc.Col([html.Div([html.Strong("CVR: ", style={'color': '#aaa'}), html.Span(f"{agg_cvr:.2f}%", style={'color': '#00ff00', 'fontSize': '18px'})])], width=2),
                dbc.Col([html.Div([html.Strong("CTR: ", style={'color': '#aaa'}), html.Span(f"{agg_ctr:.2f}%", style={'color': '#00ff00', 'fontSize': '18px'})])], width=2),
                dbc.Col([html.Div([html.Strong("CPA: ", style={'color': '#aaa'}), html.Span(f"${agg_cpa:.2f}", style={'color': '#ffcc00', 'fontSize': '18px'})])], width=2),
                dbc.Col([html.Div([html.Strong("ROAS: ", style={'color': '#aaa'}), html.Span(f"{agg_mnet_roas:.2f}", style={'color': '#00ff00', 'fontSize': '18px'})])], width=2)
            ])
        ])
    ], style={'backgroundColor': '#222', 'border': '1px solid #444', 'marginBottom': '30px'})

def get_best_worst_items(g_agg, agg_cvr, agg_ctr, agg_cpa, agg_mnet_roas, col_name, limit):
    """Extract best/worst calculation logic"""
    all_items = g_agg.dropna(subset=['cvr']).copy()
    all_items['cvr_vs_avg'] = ((all_items['cvr'] - agg_cvr) / agg_cvr * 100).round(1) if agg_cvr > 0 else 0
    
    # Best items
    best_candidates = all_items[(all_items['clicks'] >= 10) & (all_items['cvr'] > 0)].copy()
    if len(best_candidates) == 0:
        best_candidates = all_items[all_items['cvr'] > 0].copy()
    
    if len(best_candidates) > 0:
        best_candidates['score'] = best_candidates['cvr'] * np.log1p(best_candidates['clicks'])
        best_df = best_candidates.sort_values('score', ascending=False).head(limit)
        best_ids = set(best_df[col_name].tolist())
    else:
        best_df = pd.DataFrame()
        best_ids = set()
    
    # Worst items
    worst_candidates = all_items[
        (all_items['clicks'] >= 10) & 
        (all_items['cvr'] <= 0.6) &
        (~all_items[col_name].isin(best_ids))
    ].copy()
    
    if len(worst_candidates) > 0:
        worst_df = worst_candidates.sort_values('clicks', ascending=False).head(limit)
    else:
        worst_df = pd.DataFrame()
    
    # Format for display
    best_display = format_table_data(best_df, agg_cvr, agg_ctr, agg_cpa, agg_mnet_roas, col_name)
    worst_display = format_table_data(worst_df, agg_cvr, agg_ctr, agg_cpa, agg_mnet_roas, col_name)
    
    return best_display, worst_display

def format_table_data(df_input, agg_cvr, agg_ctr, agg_cpa, agg_mnet_roas, col_name):
    """Format dataframe for table display"""
    if len(df_input) == 0:
        return []
    
    df = df_input.copy()
    df['avg_cvr'] = agg_cvr
    df['ctr_vs_avg'] = df['ctr'] - agg_ctr
    df['cpa_vs_avg'] = df['cpa'] - agg_cpa
    df['mnet_roas_vs_avg'] = df['mnet_roas'] - agg_mnet_roas
    
    return df[[col_name, 'clicks', 'conversions', 'cvr', 'avg_cvr', 'cvr_vs_avg', 'ctr', 'ctr_vs_avg', 'cpa', 'cpa_vs_avg', 'mnet_roas', 'mnet_roas_vs_avg', 'adv_roas', 'adv_cost', 'max_cost']].round(2).to_dict('records')

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
        if sprig_count is None:
            sprig_count = 5
        
        # Convert to tuples for hashing
        advs_tuple = tuple(advs) if advs else ()
        camp_types_tuple = tuple(camp_types) if camp_types else ()
        camps_tuple = tuple(camps) if camps else ()
        
        # Filter data ONCE
        d = filter_dataframe(df, advs, camp_types, camps)
        d_concepts = filter_dataframe(df_concepts, advs, camp_types, camps)
        d_contextuality = filter_dataframe(df_contextuality, advs, camp_types, camps)

        
        # Early return if no data
        if len(d) == 0:
            empty = go.Figure()
            empty.update_layout(title="No data", plot_bgcolor='#111', paper_bgcolor='#111', font=dict(color='white'))
            return (html.Div("No data"), empty, empty, [], [], [], [], [], empty, empty, empty, empty, empty, empty, empty, empty)
        
        # Calculate aggregated stats ONCE
        total_clicks = d['clicks'].sum()
        total_impressions = d['impressions'].sum()
        total_conversions = d['conversions'].sum()
        total_adv_cost = d['adv_cost'].sum()
        total_max_cost = d['max_cost'].sum()
        total_adv_value = d['adv_value'].sum()
        
        agg_cvr = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
        agg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        agg_cpa = (total_adv_cost / total_conversions) if total_conversions > 0 else 0
        agg_mnet_roas = (total_adv_value / total_max_cost) if total_max_cost > 0 else 0
        agg_adv_roas = d['adv_roas'].mean()
        
        avg_metrics = {'cvr': agg_cvr, 'ctr': agg_ctr, 'cpa': agg_cpa, 'mnet_roas': agg_mnet_roas}
        
        # Aggregate ONCE for each dimension (reuse these)
        g_concept = weighted_aggregate(d_concepts, 'concepts')
        g_url = weighted_aggregate(d, 'url')
        g_contextuality = weighted_aggregate(d_contextuality, 'contextuality')
        
        # Sprig aggregations
        g_url_top = weighted_aggregate(d, 'sprig_url_top')
        g_dom_top = weighted_aggregate(d, 'sprig_domain_top')
        g_url_final = weighted_aggregate(d, 'sprig_url_final')
        g_dom_final = weighted_aggregate(d, 'sprig_domain_final')
        
        # Stats display
        stats_display = create_stats_display(total_clicks, total_conversions, agg_cvr, agg_ctr, agg_cpa, agg_mnet_roas)
        
        # Treemaps (limit to top 10 for performance)
        top_concepts = g_concept.nlargest(10, 'clicks')['concepts'].tolist()
        g_concept_top = g_concept[g_concept['concepts'].isin(top_concepts)]
        
        tree_cvr_ctr = create_treemap(g_concept_top, 'cvr', 'ctr', 
                               'CVR vs CTR - Top 10 by Clicks', 
                               show_cvr_ctr=True, top_n=10, col_name='concepts', avg_metrics=avg_metrics)
        tree_roas_cpa = create_treemap(g_concept_top, 'mnet_roas', 'cpa', 
                                'ROAS vs CPA - Top 10 by Clicks', 
                                show_cvr_ctr=False, top_n=10, col_name='concepts', avg_metrics=avg_metrics)
        
        # Best/Worst concepts and URLs (extract into helper functions)
        best_concepts, worst_concepts = get_best_worst_items(g_concept, agg_cvr, agg_ctr, agg_cpa, agg_mnet_roas, 'concepts', 10)
        best_urls, worst_urls = get_best_worst_items(g_url, agg_cvr, agg_ctr, agg_cpa, agg_mnet_roas, 'url', 5)
        
        # Contextuality table
        if len(g_contextuality) > 0:
            ctx_table_data = g_contextuality.copy()
            ctx_table_data['contextuality'] = '▶ ' + ctx_table_data['contextuality'].astype(str)
            ctx_table_data = ctx_table_data[['contextuality', 'impressions', 'clicks', 'conversions', 'cvr', 'ctr', 'cpa', 'mnet_roas', 'adv_roas']].round(2).to_dict('records')
        else:
            ctx_table_data = []
        
        # Sprig treemaps
        tree_url_top_cvr_ctr = create_treemap(g_url_top, 'cvr', 'ctr', 'Sprig URL Top: CVR vs CTR', True, 10, 'sprig_url_top', avg_metrics)
        tree_url_top_roas_cpa = create_treemap(g_url_top, 'mnet_roas', 'cpa', 'Sprig URL Top: ROAS vs CPA', False, 10, 'sprig_url_top', avg_metrics)
        tree_dom_top_cvr_ctr = create_treemap(g_dom_top, 'cvr', 'ctr', 'Sprig Domain Top: CVR vs CTR', True, 10, 'sprig_domain_top', avg_metrics)
        tree_dom_top_roas_cpa = create_treemap(g_dom_top, 'mnet_roas', 'cpa', 'Sprig Domain Top: ROAS vs CPA', False, 10, 'sprig_domain_top', avg_metrics)
        tree_url_final_cvr_ctr = create_treemap(g_url_final, 'cvr', 'ctr', 'Sprig URL Final: CVR vs CTR', True, 10, 'sprig_url_final', avg_metrics)
        tree_url_final_roas_cpa = create_treemap(g_url_final, 'mnet_roas', 'cpa', 'Sprig URL Final: ROAS vs CPA', False, 10, 'sprig_url_final', avg_metrics)
        tree_dom_final_cvr_ctr = create_treemap(g_dom_final, 'cvr', 'ctr', 'Sprig Domain Final: CVR vs CTR', True, 10, 'sprig_domain_final', avg_metrics)
        tree_dom_final_roas_cpa = create_treemap(g_dom_final, 'mnet_roas', 'cpa', 'Sprig Domain Final: ROAS vs CPA', False, 10, 'sprig_domain_final', avg_metrics)
        
        return (
            stats_display,
            tree_cvr_ctr, tree_roas_cpa,
            best_concepts, worst_concepts,
            best_urls, worst_urls,
            ctx_table_data,
            tree_url_top_cvr_ctr, tree_url_top_roas_cpa,
            tree_dom_top_cvr_ctr, tree_dom_top_roas_cpa,
            tree_url_final_cvr_ctr, tree_url_final_roas_cpa,
            tree_dom_final_cvr_ctr, tree_dom_final_roas_cpa
        )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        empty = go.Figure()
        empty.update_layout(title=f"Error: {str(e)}", plot_bgcolor='#111', paper_bgcolor='#111', font=dict(color='white'))
        return (html.Div("Error", style={'color': 'red'}), empty, empty, [], [], [], [], [], empty, empty, empty, empty, empty, empty, empty, empty)    
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
        raise dash.exceptions.PreventUpdate

    
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
    d = filter_dataframe(df, advs, camp_types, camps)  # Use the optimized function
    
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
    Output("ctx_drilldown_section", "children"),
    [Input("contextuality_table", "selected_rows"),
     Input('adv_dd','value'),
     Input('camp_type_dd','value'),
     Input('camp_dd','value')],
    [State("contextuality_table", "data")],
    prevent_initial_call=True
)
def handle_contextuality_drilldown(selected_rows, advs, camp_types, camps, table_data):
    """Show inline URL-Campaign pairs when contextuality row is clicked"""
    
    # If no row selected, hide the section
    if not selected_rows or not table_data:
        return html.Div()
    
    # Get selected contextuality value (remove the arrow prefix)
    row_idx = selected_rows[0]
    contextuality_value = table_data[row_idx]['contextuality'].replace('▶ ', '').strip()
    ctx_cvr = table_data[row_idx]['cvr']
    ctx_ctr = table_data[row_idx]['ctr']
    ctx_cpa = table_data[row_idx]['cpa']
    ctx_mnet_roas = table_data[row_idx]['mnet_roas']
    
    # Filter data
    d = filter_dataframe(df, advs, camp_types, camps)
    
    # Make sure contextuality column exists
    if 'contextuality' not in d.columns:
        return html.Div("Contextuality column not found", style={'color': 'red', 'padding': '10px'})
    
    # Filter by selected contextuality value
    d = d[d['contextuality'] == contextuality_value].copy()
    
    if len(d) == 0:
        return dbc.Alert(
            f"No data found for contextuality: {contextuality_value}",
            color="warning",
            style={'marginTop': '10px'}
        )
    
    # Aggregate by URL-Campaign pairs
    pairs = d.groupby(['url', 'campaign'], dropna=False).agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'conversions': 'sum',
        'adv_cost': 'sum',
        'max_cost': 'sum',
        'adv_value': 'sum'
    }).reset_index()
    
    # Calculate metrics
    pairs['cvr'] = np.where(pairs['clicks']>0, 100*pairs['conversions']/pairs['clicks'], 0)
    pairs['ctr'] = np.where(pairs['impressions']>0, 100*pairs['clicks']/pairs['impressions'], 0)
    pairs['cpa'] = np.where(pairs['conversions']>0, pairs['adv_cost']/pairs['conversions'], 0)
    pairs['mnet_roas'] = np.where(pairs['max_cost']>0, pairs['adv_value']/pairs['max_cost'], 0)
    
    # Get top 5 by clicks
    top_pairs = pairs.nlargest(5, 'clicks').copy()
    
    if len(top_pairs) == 0:
        return html.Div()
    
    # Calculate comparison columns for color coding
    top_pairs['cvr_vs_ctx'] = top_pairs['cvr'] - ctx_cvr
    top_pairs['ctr_vs_ctx'] = top_pairs['ctr'] - ctx_ctr
    top_pairs['cpa_vs_ctx'] = top_pairs['cpa'] - ctx_cpa
    top_pairs['mnet_roas_vs_ctx'] = top_pairs['mnet_roas'] - ctx_mnet_roas
    
    # Format for display
    drilldown_data = top_pairs[['url', 'campaign', 'clicks', 'conversions', 'cvr', 'cvr_vs_ctx', 'ctr', 'ctr_vs_ctx', 'cpa', 'cpa_vs_ctx', 'mnet_roas', 'mnet_roas_vs_ctx']].round(2).to_dict('records')
    
    # Return expanded card with table
    return dbc.Card([
        dbc.CardHeader([
            html.H5(f"📊 Top 5 URL-Campaign Pairs for Contextuality: {contextuality_value}", 
                    style={'color': '#17a2b8', 'marginBottom': '0'}),
            html.P(f"Total pairs analyzed: {len(pairs)} | Showing top 5 by clicks", 
                   style={'color': '#aaa', 'fontSize': '11px', 'marginBottom': '0'})
        ], style={'backgroundColor': '#333', 'padding': '10px'}),
        dbc.CardBody([
            dash_table.DataTable(
                data=drilldown_data,
                columns=[
                    {'name': 'URL', 'id': 'url'},
                    {'name': 'Campaign', 'id': 'campaign'},
                    {'name': 'Clicks', 'id': 'clicks', 'type': 'numeric'},
                    {'name': 'Conv', 'id': 'conversions', 'type': 'numeric'},
                    {'name': 'CVR %', 'id': 'cvr', 'type': 'numeric'},
                    {'name': 'CTR %', 'id': 'ctr', 'type': 'numeric'},
                    {'name': 'CPA', 'id': 'cpa', 'type': 'numeric'},
                    {'name': 'Mnet ROAS', 'id': 'mnet_roas', 'type': 'numeric'}
                ],
                style_data_conditional=[
                    {'if': {'filter_query': '{cvr_vs_ctx} > 0', 'column_id': 'cvr'}, 'color': '#00ff00', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{cvr_vs_ctx} < 0', 'column_id': 'cvr'}, 'color': '#ff0000', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{ctr_vs_ctx} > 0', 'column_id': 'ctr'}, 'color': '#00ff00', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{ctr_vs_ctx} < 0', 'column_id': 'ctr'}, 'color': '#ff0000', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{cpa_vs_ctx} < 0', 'column_id': 'cpa'}, 'color': '#00ff00', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{cpa_vs_ctx} > 0', 'column_id': 'cpa'}, 'color': '#ff0000', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{mnet_roas_vs_ctx} > 0', 'column_id': 'mnet_roas'}, 'color': '#00ff00', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{mnet_roas_vs_ctx} < 0', 'column_id': 'mnet_roas'}, 'color': '#ff0000', 'fontWeight': 'bold'}
                ],
                page_size=5,
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': '#444', 'color': 'white', 'fontWeight': 'bold'},
                style_data={'backgroundColor': '#1a1a1a', 'color': 'white', 'fontSize': '11px'},
                style_cell={'textAlign': 'left', 'padding': '8px'},
                style_cell_conditional=[
                    {'if': {'column_id': 'url'}, 'maxWidth': '400px', 'overflow': 'hidden', 'textOverflow': 'ellipsis'}
                ]
            )
        ], style={'backgroundColor': '#1a1a1a', 'padding': '15px'})
    ], style={'marginTop': '10px', 'border': '2px solid #17a2b8'})


