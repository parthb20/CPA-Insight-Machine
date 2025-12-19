import dash
from dash import html
import dash_bootstrap_components as dbc

# Initialize app with multi-page support
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    use_pages=True,
    suppress_callback_exceptions=True
)

server = app.server

# Navigation bar - UPDATED WITH OVERVIEW PAGE
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Overview", href="/overview", active="exact")),
        dbc.NavItem(dbc.NavLink("URL Analysis", href="/", active="exact")),
        dbc.NavItem(dbc.NavLink("Creative & SERP", href="/creative", active="exact"))
    ],
    brand="CPA Insight Dashboard",
    brand_href="/overview",
    color="#17a2b8",
    dark=True,
    sticky="top",
    style={'marginBottom': '0px'}
)

# Main layout
app.layout = dbc.Container(
    fluid=True,
    style={'backgroundColor': '#111', 'minHeight': '100vh', 'padding': '0px'},
    children=[
        navbar,
        html.Div(
            dash.page_container,
            style={'padding': '20px'}
        )
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
