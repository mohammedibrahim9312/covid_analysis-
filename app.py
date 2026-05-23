import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

# ============================================================
# APP INITIALIZATION
# ============================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
app.title = "COVID-19 Dashboard"
server = app.server

# ============================================================
# COLOR PALETTE — unified pastel palette (reference for all charts)
# ============================================================
PASTEL_PALETTE = [
    "#87CEEB",  # light blue
    "#98FB98",  # light green
    "#DDA0DD",  # light purple / plum
    "#FFD700",  # light gold / yellow
    "#FFB6C1",  # light pink
    "#FFDAB9",  # peach
    "#B0E0E6",  # powder blue
    "#E6E6FA",  # lavender
    "#90EE90",  # light green (alt)
    "#F0E68C",  # khaki / light yellow
]

CHART_HEIGHT = 430

METRIC_LABELS = {
    "total_cases_per_million": "Total cases per million",
    "total_deaths_per_million": "Total deaths per million",
    "people_fully_vaccinated_per_hundred": "Fully vaccinated per 100 people",
    "case_fatality_rate": "Case fatality rate (%)",
}

DEFAULT_COUNTRIES = ["Egypt", "United States", "India", "Brazil", "United Kingdom"]

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def empty_figure(title, message="No data available for the selected filters."):
    fig = go.Figure()
    fig.update_layout(
        title={"text": title, "font": {"color": "black"}},
        height=CHART_HEIGHT,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "black"},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 15, "color": "black"},
            }
        ],
        margin={"l": 30, "r": 30, "t": 70, "b": 40},
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#E5E5E5", griddash="dot")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#E5E5E5", griddash="dot")
    return fig


def polish(fig, title, x_title=None, y_title=None):
    """Apply consistent style with dotted grid lines."""
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left", "font": {"color": "black"}},
        height=CHART_HEIGHT,
        margin={"l": 45, "r": 25, "t": 85, "b": 55},
        hovermode="closest",
        font={"family": "Arial, sans-serif", "size": 13, "color": "black"},
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={
            "font": {"color": "black"},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor="#E5E5E5", griddash="dot",
        tickfont={"color": "black"}, color="black"
    )
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor="#E5E5E5", griddash="dot",
        tickfont={"color": "black"}, color="black"
    )
    if x_title:
        fig.update_xaxes(title_text=x_title, title_font={"color": "black"})
    if y_title:
        fig.update_yaxes(title_text=y_title, title_font={"color": "black"})
    return fig


# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    df = pd.read_csv("data.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


data = load_data()
MIN_DATE = data["date"].min().date()
MAX_DATE = data["date"].max().date()
CONTINENTS = sorted(data["continent"].dropna().unique())
COUNTRIES = sorted(data["location"].dropna().unique())
DEFAULT_COUNTRIES = [country for country in DEFAULT_COUNTRIES if country in COUNTRIES] or COUNTRIES[:5]


def get_country_data():
    agg_dict = {
        "total_cases": "max",
        "total_deaths": "max",
        "total_cases_per_million": "max",
        "total_deaths_per_million": "max",
        "new_cases_per_million": "max",
        "new_deaths_per_million": "max",
        "new_cases": "max",
        "new_deaths": "max",
        "population_density": "max",
        "median_age": "max",
        "aged_65_older": "max",
        "gdp_per_capita": "max",
        "cardiovasc_death_rate": "max",
        "diabetes_prevalence": "max",
        "female_smokers": "max",
        "male_smokers": "max",
        "hospital_beds_per_thousand": "max",
        "life_expectancy": "max",
        "human_development_index": "max",
        "population": "max",
    }
    optional_cols = [
        "people_fully_vaccinated",
        "people_vaccinated_per_hundred",
        "people_fully_vaccinated_per_hundred",
        "total_boosters_per_hundred",
        "new_cases_smoothed",
        "case_fatality_rate",
        "male_deaths",
        "female_deaths",
        "male_cases",
        "female_cases",
    ]
    for col in optional_cols:
        if col in data.columns:
            agg_dict[col] = "max"

    country = (
        data.sort_values("date")
        .groupby(["location", "continent"])
        .agg(agg_dict)
        .reset_index()
    )
    return country


country_data = get_country_data()


def get_continent_data(cdf):
    agg_dict = {
        "total_cases": "sum",
        "total_deaths": "sum",
        "new_cases": "sum",
        "new_deaths": "sum",
        "population": "sum",
        "population_density": "mean",
        "median_age": "mean",
        "aged_65_older": "mean",
        "gdp_per_capita": "mean",
        "cardiovasc_death_rate": "mean",
        "diabetes_prevalence": "mean",
        "female_smokers": "mean",
        "male_smokers": "mean",
        "hospital_beds_per_thousand": "mean",
        "life_expectancy": "mean",
        "human_development_index": "mean",
    }
    optional_mean_cols = [
        "people_vaccinated_per_hundred",
        "people_fully_vaccinated_per_hundred",
        "total_boosters_per_hundred",
    ]
    for col in optional_mean_cols:
        if col in cdf.columns:
            agg_dict[col] = "mean"

    cf = cdf.groupby("continent").agg(agg_dict).reset_index()
    cf["total_cases_per_million"] = cf["total_cases"] / cf["population"] * 1e6
    cf["total_deaths_per_million"] = cf["total_deaths"] / cf["population"] * 1e6
    cf["new_cases_per_million"] = cf["new_cases"] / cf["population"] * 1e6
    cf["new_deaths_per_million"] = cf["new_deaths"] / cf["population"] * 1e6
    cf["survived"] = cf["total_cases"] - cf["total_deaths"]
    cf["non_aged_65"] = 100 - cf["aged_65_older"]
    return cf


continent_df = get_continent_data(country_data)


# ============================================================
# DASHBOARD HELPER FUNCTIONS
# ============================================================
def latest_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.sort_values("date").groupby("location", as_index=False).tail(1)


def filtered_data(continents, start_date, end_date) -> pd.DataFrame:
    df_local = data.copy()
    if continents:
        df_local = df_local[df_local["continent"].isin(continents)]
    start = pd.to_datetime(start_date) if start_date else data["date"].min()
    end = pd.to_datetime(end_date) if end_date else data["date"].max()
    df_local = df_local[(df_local["date"] >= start) & (df_local["date"] <= end)]
    return df_local


# ============================================================
# SIDEBAR / FILTER LAYOUT
# ============================================================
all_continents = sorted(data["continent"].dropna().unique().tolist())

sidebar = html.Div(
    [
        html.H4("⚙️ Dashboard Filters", className="mb-4", style={"fontWeight": "bold"}),

        html.Label("Continents", style={"fontWeight": "bold", "marginTop": "10px"}),
        dcc.Dropdown(
            id="continent-filter",
            options=[{"label": c, "value": c} for c in all_continents],
            value=all_continents,
            multi=True,
            placeholder="Select continents...",
            style={"marginBottom": "15px"},
        ),

        html.Label("Start date", style={"fontWeight": "bold", "marginTop": "10px"}),
        dcc.DatePickerSingle(
            id="start-date",
            min_date_allowed=MIN_DATE,
            max_date_allowed=MAX_DATE,
            initial_visible_month=MIN_DATE,
            date=MIN_DATE,
            style={"marginBottom": "15px"},
        ),

        html.Label("End date", style={"fontWeight": "bold", "marginTop": "10px"}),
        dcc.DatePickerSingle(
            id="end-date",
            min_date_allowed=MIN_DATE,
            max_date_allowed=MAX_DATE,
            initial_visible_month=MAX_DATE,
            date=MAX_DATE,
            style={"marginBottom": "15px"},
        ),

        html.Label("Top N countries", style={"fontWeight": "bold", "marginTop": "10px"}),
        html.Div(
            dcc.Slider(
                id="top-n-slider",
                min=5,
                max=20,
                step=1,
                value=10,
                marks={i: str(i) for i in range(5, 21, 5)},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            style={"marginBottom": "20px"},
        ),

        html.Label("Main metric", style={"fontWeight": "bold", "marginTop": "10px"}),
        dcc.RadioItems(
            id="metric-radio",
            options=[
                {"label": METRIC_LABELS[k], "value": k}
                for k in METRIC_LABELS.keys()
            ],
            value=list(METRIC_LABELS.keys())[0],
            labelStyle={"display": "block", "marginBottom": "5px"},
            style={"marginBottom": "15px"},
        ),

        html.Label("Countries (Time Series)", style={"fontWeight": "bold", "marginTop": "10px"}),
        dcc.Dropdown(
            id="country-dropdown",
            options=[{"label": c, "value": c} for c in COUNTRIES],
            value=DEFAULT_COUNTRIES,
            multi=True,
            placeholder="Select countries...",
            style={"marginBottom": "15px"},
        ),
    ],
    style={
        "padding": "20px",
        "backgroundColor": "#f8f9fa",
        "borderRight": "1px solid #dee2e6",
        "height": "100vh",
        "position": "sticky",
        "top": 0,
        "overflowY": "auto",
    },
)

# ============================================================
# MAIN CONTENT LAYOUT
# ============================================================
content = html.Div(
    [
        html.H1("COVID Data Visualization Data Analysis and Visualization", style={"marginTop": "20px"}),
        html.H5("Data Visualization Project", style={"color": "#666", "marginBottom": "20px"}),
        html.Hr(),

        # Chart 1: Column Chart — Top Countries by Selected Metric
        html.H4("1. Column Chart — Top Countries by Selected Metric", className="mt-4"),
        dcc.Graph(id="chart-1", style={"height": f"{CHART_HEIGHT}px"}),
        html.Hr(),

        # Chart 2: Bar Chart — Top Countries by Deaths per Million
        html.H4("2. Bar Chart — Top Countries by Deaths per Million", className="mt-4"),
        dcc.Graph(id="chart-2", style={"height": f"{CHART_HEIGHT}px"}),
        html.Hr(),

        # Chart 3: Stacked Bar Chart — Cases Composition by Country
        html.H4("3. Stacked Bar Chart — Cases Composition by Country", className="mt-4"),
        dcc.Graph(id="chart-3", style={"height": f"{CHART_HEIGHT}px"}),
        html.Hr(),

        # Chart 4: Stacked Column Chart — Male vs Female Outcomes by Continent
        html.H4("4. Stacked Column Chart — Male vs Female Outcomes by Continent", className="mt-4"),
        dcc.Graph(id="chart-4", style={"height": f"{CHART_HEIGHT}px"}),
        html.Hr(),

        # Chart 5: Clustered Column Chart — Male vs Female Health Burden by Continent
        html.H4("5. Clustered Column Chart — Male vs Female Health Burden by Continent", className="mt-4"),
        dcc.Graph(id="chart-5", style={"height": f"{CHART_HEIGHT}px"}),
        html.Hr(),

        # Chart 6: Clustered Bar Chart — Health Burden by Continent
        html.H4("6. Clustered Bar Chart — Health Burden by Continent", className="mt-4"),
        dcc.Graph(id="chart-6", style={"height": f"{CHART_HEIGHT}px"}),
        html.Hr(),

        # NEW Chart 6.5: Scatter Chart — Hospital Beds vs Cardiovascular Death Rate
        html.H4("7. Scatter Chart — Hospital Beds vs Cardiovascular Death Rate", className="mt-4"),
        dcc.Graph(id="chart-scatter", style={"height": f"{CHART_HEIGHT}px"}),
        html.Hr(),

        # Chart 7: Bubble Chart — GDP vs Life Expectancy (renumbered to 8)
        html.H4("8. Bubble Chart — GDP vs Life Expectancy", className="mt-4"),
        dcc.Graph(id="chart-7", style={"height": f"{CHART_HEIGHT}px"}),
        html.Hr(),

        # Chart 8: Histogram — Life Expectancy Distribution by Continent (renumbered to 9)
        html.H4("9. Histogram — Life Expectancy Distribution by Continent", className="mt-4"),
        dcc.Graph(id="chart-8", style={"height": f"{CHART_HEIGHT}px"}),
        html.Hr(),

        # Chart 9: Box Chart — Total Cases per Million by Continent (renumbered to 10)
        html.H4("10. Box Chart — Total Cases per Million by Continent", className="mt-4"),
        dcc.Graph(id="chart-9", style={"height": f"{CHART_HEIGHT}px"}),
        html.Hr(),

        # Chart 10: Violin Chart — Total Deaths per Million by Continent (renumbered to 11)
        html.H4("11. Violin Chart — Total Deaths per Million by Continent", className="mt-4"),
        dcc.Graph(id="chart-10", style={"height": f"{CHART_HEIGHT}px"}),
        html.Hr(),

        # Chart 11: Line Chart — Selected Metric over Time (renumbered to 12)
        html.H4("12. Line Chart — Selected Metric over Time", className="mt-4"),
        dcc.Graph(id="chart-11", style={"height": f"{CHART_HEIGHT}px"}),
        html.Hr(),

        # Chart 12: Area Chart — Total Deaths Over Time for Selected Countries (renumbered to 13)
        html.H4("13. Area Chart — Total Deaths Over Time", className="mt-4"),
        dcc.Graph(id="chart-12", style={"height": f"{CHART_HEIGHT}px"}),
        html.Hr(),
    ],
    style={"padding": "0 30px 30px 30px"},
)

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(sidebar, width=3),
                dbc.Col(content, width=9),
            ],
            className="g-0",
        )
    ],
    fluid=True,
    style={"maxWidth": "100%", "padding": 0},
)


# ============================================================
# CALLBACKS
# ============================================================

@app.callback(
    Output("country-dropdown", "options"),
    Input("continent-filter", "value"),
)
def update_country_options(selected_continents):
    if not selected_continents:
        countries_for_dropdown = data
    else:
        countries_for_dropdown = data[data["continent"].isin(selected_continents)]
    locations = sorted(countries_for_dropdown["location"].dropna().unique().tolist())
    return [{"label": c, "value": c} for c in locations]


@app.callback(
    Output("country-dropdown", "value"),
    Input("country-dropdown", "options"),
    State("country-dropdown", "value"),
)
def update_country_value(available_options, current_value):
    if not available_options:
        return []
    available_countries = [opt["value"] for opt in available_options]
    if current_value:
        valid = [c for c in current_value if c in available_countries]
        if valid:
            return valid
    # Default fallback
    default = [c for c in DEFAULT_COUNTRIES if c in available_countries][:5]
    return default or available_countries[:5]


@app.callback(
    [
        Output("chart-1", "figure"),
        Output("chart-2", "figure"),
        Output("chart-3", "figure"),
        Output("chart-4", "figure"),
        Output("chart-5", "figure"),
        Output("chart-6", "figure"),
        Output("chart-scatter", "figure"),
        Output("chart-7", "figure"),
        Output("chart-8", "figure"),
        Output("chart-9", "figure"),
        Output("chart-10", "figure"),
        Output("chart-11", "figure"),
        Output("chart-12", "figure"),
    ],
    [
        Input("continent-filter", "value"),
        Input("start-date", "date"),
        Input("end-date", "date"),
        Input("top-n-slider", "value"),
        Input("metric-radio", "value"),
        Input("country-dropdown", "value"),
    ],
)
def update_all_charts(continents, start_date, end_date, top_n, metric, selected_countries):
    if not continents:
        raise PreventUpdate

    # Prepare filtered data
    fd = filtered_data(continents, start_date, end_date)
    latest = latest_rows(fd)

    country_filt = country_data[country_data["continent"].isin(continents)].copy()
    continent_filt = continent_df[continent_df["continent"].isin(continents)].copy()

    metric_label = METRIC_LABELS.get(metric, metric)

    selected_countries = selected_countries or DEFAULT_COUNTRIES

    # ============================================================
    # GENDER DATA PREPARATION (for Charts 4 & 5)
    # ============================================================
    work_df = country_filt.copy()
    has_male = "male_deaths" in work_df.columns and "male_cases" in work_df.columns
    has_female = "female_deaths" in work_df.columns and "female_cases" in work_df.columns
    has_male_smokers = "male_smokers" in work_df.columns
    has_female_smokers = "female_smokers" in work_df.columns

    if has_male and has_female:
        work_df["male_deaths_pct"] = (work_df["male_deaths"] / work_df["male_cases"].replace(0, np.nan) * 100).fillna(0)
        work_df["male_survivals_pct"] = (100 - work_df["male_deaths_pct"])
        work_df["female_deaths_pct"] = (work_df["female_deaths"] / work_df["female_cases"].replace(0, np.nan) * 100).fillna(0)
        work_df["female_survivals_pct"] = (100 - work_df["female_deaths_pct"])
    elif has_male_smokers and has_female_smokers:
        work_df["male_deaths_pct"] = work_df["male_smokers"].fillna(0)
        work_df["male_survivals_pct"] = (100 - work_df["male_smokers"].fillna(0))
        work_df["female_deaths_pct"] = work_df["female_smokers"].fillna(0)
        work_df["female_survivals_pct"] = (100 - work_df["female_smokers"].fillna(0))
    else:
        if "case_fatality_rate" in work_df.columns:
            work_df["male_deaths_pct"] = work_df["case_fatality_rate"].fillna(0)
        else:
            work_df["male_deaths_pct"] = (work_df["total_deaths"] / work_df["total_cases"].replace(0, np.nan) * 100).fillna(0)
        work_df["male_survivals_pct"] = (100 - work_df["male_deaths_pct"])
        work_df["female_deaths_pct"] = work_df["male_deaths_pct"]
        work_df["female_survivals_pct"] = work_df["male_survivals_pct"]

    cont_gender = work_df.groupby("continent").agg({
        "male_deaths_pct": "mean",
        "male_survivals_pct": "mean",
        "female_deaths_pct": "mean",
        "female_survivals_pct": "mean",
    }).reset_index().dropna()

    # ============================================================
    # CHART 1: Column Chart — Top Countries by Selected Metric
    # FIX: Now uses top_n dynamically instead of hardcoded 5.
    # IMPROVED: Better scaling with bargap, text positioning, and bar width.
    # ============================================================
    column_df = latest.dropna(subset=[metric]).nlargest(top_n, metric).sort_values(metric, ascending=False)
    if column_df.empty:
        fig1 = empty_figure("Column Chart — Top Countries")
    else:
        top_row = column_df.iloc[0]
        others_df = column_df.iloc[1:].copy()

        fig1 = go.Figure()

        # Top 1 country — Light Green, legend item FIRST
        fig1.add_trace(go.Bar(
            name=top_row["location"],
            x=[top_row["location"]],
            y=[top_row[metric]],
            marker_color="#98FB98",  # Light Green
            text=[f"{top_row[metric]:,.2f}"],
            textposition="outside",
            textfont={"color": "black", "size": 12},
            hovertemplate="<b>%{x}</b><br>" + metric_label + ": %{y:,.2f}<extra>" + top_row["location"] + "</extra>",
        ))

        # Other countries — Light Blue, combined into a single legend item
        if not others_df.empty:
            fig1.add_trace(go.Bar(
                name="Other Countries",
                x=others_df["location"].tolist(),
                y=others_df[metric].tolist(),
                marker_color="#87CEEB",  # Light Blue
                text=[f"{v:,.2f}" for v in others_df[metric]],
                textposition="outside",
                textfont={"color": "black", "size": 11},
                hovertemplate="<b>%{x}</b><br>" + metric_label + ": %{y:,.2f}<extra>Other Countries</extra>",
            ))

        fig1.update_xaxes(
            categoryorder="array",
            categoryarray=column_df["location"].tolist(),
            tickangle=-30,
        )
        fig1.update_layout(
            barmode="group",
            legend_title_text="Country",
            # IMPROVED SCALING: Dynamic bargap based on number of bars for better proportions
            bargap=max(0.15, min(0.45, 0.05 * top_n)),
        )
        fig1 = polish(fig1, f"Column Chart — Top {top_n} Countries by {metric_label}", "Country", metric_label)

    # ============================================================
    # CHART 2: Bar Chart — Top Countries by Deaths per Million
    # ============================================================
    bar_df = latest.dropna(subset=["total_deaths_per_million"]).nlargest(top_n, "total_deaths_per_million")
    bar_df = bar_df.sort_values("total_deaths_per_million")
    if bar_df.empty:
        fig2 = empty_figure("Bar Chart — Deaths per Million")
    else:
        continents_in_bar = sorted(bar_df["continent"].dropna().unique().tolist())
        bar_color_map = {cont: PASTEL_PALETTE[i % len(PASTEL_PALETTE)] for i, cont in enumerate(continents_in_bar)}

        fig2 = go.Figure()
        for cont in continents_in_bar:
            subset = bar_df[bar_df["continent"] == cont]
            fig2.add_trace(go.Bar(
                name=cont,
                y=subset["location"],
                x=subset["total_deaths_per_million"],
                orientation="h",
                marker_color=bar_color_map[cont],
                hovertemplate="<b>%{y}</b><br>Deaths per million: %{x:,.2f}<extra>" + cont + "</extra>",
            ))

        fig2.update_layout(barmode="group", legend_title_text="Continent", bargap=0.25)
        fig2 = polish(fig2, f"Bar Chart — Top {top_n} Countries by Deaths per Million", "Deaths per million", "Country")

    # ============================================================
    # CHART 3: Stacked Bar Chart — Cases Composition by Country
    # ============================================================
    if latest.empty:
        fig3 = empty_figure("Stacked Bar Chart — Cases Composition by Country")
    else:
        comp_df = latest.dropna(subset=["total_cases", "total_deaths"]).nlargest(top_n, "total_cases").copy()
        comp_df["Survived"] = comp_df["total_cases"] - comp_df["total_deaths"]
        comp_df["Deaths"] = comp_df["total_deaths"]
        comp_df["pct_survived"] = (comp_df["Survived"] / comp_df["total_cases"] * 100).round(1)
        comp_df["pct_deaths"] = (comp_df["Deaths"] / comp_df["total_cases"] * 100).round(1)
        comp_df = comp_df.sort_values("total_cases", ascending=True)

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            name="Survived",
            y=comp_df["location"],
            x=comp_df["Survived"],
            orientation="h",
            marker_color="#C8E6C9",
            hovertemplate="<b>%{y}</b><br>Survived: %{x:,.0f}<extra></extra>",
            text=[f"{p}%" for p in comp_df["pct_survived"]],
            textposition="inside",
            insidetextanchor="middle",
            textfont={"color": "black", "size": 11},
        ))
        fig3.add_trace(go.Bar(
            name="Deaths",
            y=comp_df["location"],
            x=comp_df["Deaths"],
            orientation="h",
            marker_color="#388E3C",
            hovertemplate="<b>%{y}</b><br>Deaths: %{x:,.0f}<extra></extra>",
            text=[f"{p}%" for p in comp_df["pct_deaths"]],
            textposition="inside",
            insidetextanchor="middle",
            textfont={"color": "white", "size": 11},
        ))
        fig3.update_layout(barmode="stack", legend_title_text="Case Type", bargap=0.25)
        fig3 = polish(
            fig3,
            f"Stacked Bar Chart — Cases Composition in Top {top_n} Countries by Total Cases",
            "Case count",
            "Country",
        )

    # ============================================================
    # CHART 4: Stacked Column Chart — Male vs Female Outcomes by Continent
    # ============================================================
    if cont_gender.empty:
        fig4 = empty_figure("Stacked Column Chart — Male vs Female Outcomes by Continent")
    else:
        continents = cont_gender["continent"].tolist()

        fig4 = go.Figure()

        fig4.add_trace(go.Bar(
            name="Male — Survived",
            x=continents,
            y=cont_gender["male_survivals_pct"].round(1),
            offsetgroup=0,
            marker_color="#90CAF9",
            text=[f"{v:.1f}%" for v in cont_gender["male_survivals_pct"]],
            textposition="inside",
            insidetextanchor="middle",
            textfont={"color": "black", "size": 10},
            hovertemplate="<b>%{x}</b><br>Male Survived: %{y:.1f}%<extra></extra>",
        ))

        fig4.add_trace(go.Bar(
            name="Male — Died",
            x=continents,
            y=cont_gender["male_deaths_pct"].round(1),
            offsetgroup=0,
            marker_color="#1565C0",
            text=[f"{v:.1f}%" for v in cont_gender["male_deaths_pct"]],
            textposition="inside",
            insidetextanchor="middle",
            textfont={"color": "white", "size": 10},
            hovertemplate="<b>%{x}</b><br>Male Died: %{y:.1f}%<extra></extra>",
        ))

        fig4.add_trace(go.Bar(
            name="Female — Survived",
            x=continents,
            y=cont_gender["female_survivals_pct"].round(1),
            offsetgroup=1,
            marker_color="#F48FB1",
            text=[f"{v:.1f}%" for v in cont_gender["female_survivals_pct"]],
            textposition="inside",
            insidetextanchor="middle",
            textfont={"color": "black", "size": 10},
            hovertemplate="<b>%{x}</b><br>Female Survived: %{y:.1f}%<extra></extra>",
        ))

        fig4.add_trace(go.Bar(
            name="Female — Died",
            x=continents,
            y=cont_gender["female_deaths_pct"].round(1),
            offsetgroup=1,
            marker_color="#C2185B",
            text=[f"{v:.1f}%" for v in cont_gender["female_deaths_pct"]],
            textposition="inside",
            insidetextanchor="middle",
            textfont={"color": "white", "size": 10},
            hovertemplate="<b>%{x}</b><br>Female Died: %{y:.1f}%<extra></extra>",
        ))

        fig4.update_layout(
            barmode="relative",
            legend_title_text="Gender & Outcome",
            bargap=0.2,
            bargroupgap=0.15,
        )
        fig4 = polish(
            fig4,
            "Stacked Column Chart — Male vs Female Outcomes by Continent",
            "Continent",
            "Percentage (%)",
        )

    # ============================================================
    # CHART 5: Clustered Column Chart — Male vs Female Health Burden by Continent
    # ============================================================
    if cont_gender.empty:
        fig5 = empty_figure("Clustered Column Chart — Male vs Female Health Burden")
    else:
        continents = cont_gender["continent"].tolist()

        fig5 = go.Figure()

        fig5.add_trace(go.Bar(
            name="Male — Deaths",
            x=continents,
            y=cont_gender["male_deaths_pct"].round(2),
            marker_color="#81D4FA",
            hovertemplate="<b>%{x}</b><br>Male Deaths: %{y:.2f}<extra></extra>",
        ))

        fig5.add_trace(go.Bar(
            name="Male — Survivals",
            x=continents,
            y=cont_gender["male_survivals_pct"].round(2),
            marker_color="#4FC3F7",
            hovertemplate="<b>%{x}</b><br>Male Survivals: %{y:.2f}<extra></extra>",
        ))

        fig5.add_trace(go.Bar(
            name="Female — Deaths",
            x=continents,
            y=cont_gender["female_deaths_pct"].round(2),
            marker_color="#F48FB1",
            hovertemplate="<b>%{x}</b><br>Female Deaths: %{y:.2f}<extra></extra>",
        ))

        fig5.add_trace(go.Bar(
            name="Female — Survivals",
            x=continents,
            y=cont_gender["female_survivals_pct"].round(2),
            marker_color="#F06292",
            hovertemplate="<b>%{x}</b><br>Female Survivals: %{y:.2f}<extra></extra>",
        ))

        fig5.update_layout(barmode="group", legend_title_text="Gender & Metric", bargap=0.25)
        fig5 = polish(
            fig5,
            "Clustered Column Chart — Male vs Female Health Burden by Continent",
            "Continent",
            "Value",
        )

    # ============================================================
    # CHART 6: Clustered Bar Chart — Health Burden by Continent
    # ============================================================
    if continent_filt.empty:
        fig6 = empty_figure("Clustered Bar Chart — Health Burden by Continent")
    else:
        cluster_bar = continent_filt.dropna(
            subset=["cardiovasc_death_rate", "diabetes_prevalence"]
        )
        cluster_bar_long = cluster_bar.melt(
            id_vars=["continent"],
            value_vars=["cardiovasc_death_rate", "diabetes_prevalence"],
            var_name="Metric",
            value_name="Rate",
        ).dropna(subset=["Rate"])
        cluster_bar_long["Metric"] = cluster_bar_long["Metric"].replace(
            {
                "cardiovasc_death_rate": "Cardiovascular death rate",
                "diabetes_prevalence": "Diabetes prevalence",
            }
        )
        fig6 = px.bar(
            cluster_bar_long,
            y="continent",
            x="Rate",
            color="Metric",
            orientation="h",
            barmode="group",
            labels={"continent": "Continent", "Metric": "Metric"},
            color_discrete_sequence=PASTEL_PALETTE[:2],
        )
        fig6.update_layout(legend_title_text="Metric", bargap=0.25)
        fig6 = polish(
            fig6,
            "Clustered Bar Chart — Health Burden by Continent",
            "Rate",
            "Continent",
        )

    # ============================================================
    # NEW CHART: Scatter Chart — Hospital Beds vs Cardiovascular Death Rate
    # Added BEFORE the Bubble Chart. Uses meaningful health variables.
    # NO size attribute (unlike Bubble Chart).
    # ============================================================
    if country_filt.empty:
        fig_scatter = empty_figure("Scatter Chart — Hospital Beds vs Cardiovascular Death Rate")
    else:
        scatter_df = country_filt.dropna(
            subset=["hospital_beds_per_thousand", "cardiovasc_death_rate", "continent"]
        )
        if scatter_df.empty:
            fig_scatter = empty_figure("Scatter Chart — Hospital Beds vs Cardiovascular Death Rate")
        else:
            fig_scatter = px.scatter(
                scatter_df,
                x="hospital_beds_per_thousand",
                y="cardiovasc_death_rate",
                color="continent",
                hover_name="location",
                labels={
                    "hospital_beds_per_thousand": "Hospital beds per 1,000 people",
                    "cardiovasc_death_rate": "Cardiovascular death rate",
                    "continent": "Continent",
                },
                color_discrete_sequence=PASTEL_PALETTE,
            )
            fig_scatter.update_traces(
                marker=dict(size=10, opacity=0.85, line=dict(width=1, color="white")),
            )
            fig_scatter.update_layout(legend_title_text="Continent")
            fig_scatter = polish(
                fig_scatter,
                "Scatter Chart — Hospital Beds vs Cardiovascular Death Rate by Continent",
                "Hospital beds per 1,000 people",
                "Cardiovascular death rate",
            )

    # ============================================================
    # CHART 7: Bubble Chart — GDP vs Life Expectancy
    # ============================================================
    if country_filt.empty:
        fig7 = empty_figure("Bubble Chart — GDP vs Life Expectancy")
    else:
        bubble_df = country_filt.dropna(
            subset=["gdp_per_capita", "life_expectancy", "population"]
        )
        fig7 = px.scatter(
            bubble_df,
            x="gdp_per_capita",
            y="life_expectancy",
            size="population",
            color="continent",
            hover_name="location",
            size_max=44,
            labels={
                "gdp_per_capita": "GDP per capita",
                "life_expectancy": "Life expectancy",
                "continent": "Continent",
            },
            color_discrete_sequence=PASTEL_PALETTE,
        )
        fig7.update_layout(legend_title_text="Continent")
        fig7 = polish(
            fig7,
            "Bubble Chart — GDP vs Life Expectancy, Sized by Population",
            "GDP per capita",
            "Life expectancy",
        )

    # ============================================================
    # CHART 8: Histogram — Life Expectancy Distribution by Continent
    # ============================================================
    if country_filt.empty:
        fig8 = empty_figure("Histogram — Life Expectancy Distribution by Continent")
    else:
        hist_df = country_filt.dropna(subset=["life_expectancy"])
        continents_in_hist = sorted(hist_df["continent"].dropna().unique().tolist())

        hist_color_map = {
            cont: PASTEL_PALETTE[i % len(PASTEL_PALETTE)]
            for i, cont in enumerate(continents_in_hist)
        }

        fig8 = go.Figure()
        for cont in continents_in_hist:
            subset = hist_df[hist_df["continent"] == cont]
            fig8.add_trace(go.Histogram(
                x=subset["life_expectancy"],
                name=cont,
                nbinsx=30,
                marker_color=hist_color_map[cont],
                opacity=0.80,
                hovertemplate=f"<b>{cont}</b><br>Life expectancy: %{{x}}<br>Count: %{{y}}<extra></extra>",
            ))

        fig8.update_layout(barmode="overlay", legend_title_text="Continent")
        fig8 = polish(
            fig8,
            "Histogram — Distribution of Life Expectancy by Country",
            "Life expectancy",
            "Number of countries",
        )

    # ============================================================
    # CHART 9: Box Chart — Total Cases per Million by Continent
    # ============================================================
    if country_filt.empty:
        fig9 = empty_figure("Box Chart — Total Cases per Million by Continent")
    else:
        box_df = country_filt.dropna(subset=["continent", "total_cases_per_million"])
        fig9 = px.box(
            box_df,
            x="continent",
            y="total_cases_per_million",
            color="continent",
            points="outliers",
            labels={
                "continent": "Continent",
                "total_cases_per_million": "Cases per million",
            },
            color_discrete_sequence=PASTEL_PALETTE,
        )
        fig9.update_layout(legend_title_text="Continent")
        fig9 = polish(
            fig9,
            "Box Chart — Distribution of Cases per Million by Continent",
            "Continent",
            "Cases per million",
        )

    # ============================================================
    # CHART 10: Violin Chart — Total Deaths per Million by Continent
    # ============================================================
    if country_filt.empty:
        fig10 = empty_figure("Violin Chart — Total Deaths per Million by Continent")
    else:
        violin_df = country_filt.dropna(subset=["continent", "total_deaths_per_million"])
        continents_in_violin = sorted(violin_df["continent"].dropna().unique().tolist())

        violin_color_map = {
            cont: PASTEL_PALETTE[i % len(PASTEL_PALETTE)]
            for i, cont in enumerate(continents_in_violin)
        }

        fig10 = go.Figure()
        for cont in continents_in_violin:
            subset = violin_df[violin_df["continent"] == cont]
            fig10.add_trace(go.Violin(
                x=subset["continent"],
                y=subset["total_deaths_per_million"],
                name=cont,
                box_visible=True,
                points=False,
                fillcolor=violin_color_map[cont],
                line=dict(color="#555555", width=1.5),
                opacity=0.85,
                hovertemplate=f"<b>{cont}</b><br>Deaths per million: %{{y:,.2f}}<extra></extra>",
                meanline_visible=True,
                meanline=dict(color="#333333", width=1),
            ))

        fig10.update_layout(legend_title_text="Continent")
        fig10 = polish(
            fig10,
            "Violin Chart — Distribution of Deaths per Million by Continent",
            "Continent",
            "Deaths per million",
        )

    # ============================================================
    # CHART 11: Line Chart — Selected Metric over Time
    # ============================================================
    line_df = fd[fd["location"].isin(selected_countries)].dropna(subset=[metric])
    if line_df.empty:
        fig11 = empty_figure("Line Chart — Selected Countries over Time")
    else:
        fig11 = px.line(
            line_df,
            x="date",
            y=metric,
            color="location",
            labels={"date": "Date", metric: metric_label, "location": "Country"},
            color_discrete_sequence=PASTEL_PALETTE,
        )
        fig11.update_layout(legend_title_text="Country")
        fig11 = polish(fig11, f"Line Chart — {metric_label} over Time", "Date", metric_label)

    # ============================================================
    # CHART 12: Area Chart — Total Deaths Over Time for Selected Countries
    # ============================================================
    area_df = fd[fd["location"].isin(selected_countries)].copy()
    if area_df.empty:
        fig12 = empty_figure("Area Chart — Total Deaths Over Time")
    else:
        fig12 = go.Figure()
        for i, country in enumerate(selected_countries):
            country_df = area_df[area_df["location"] == country].sort_values("date")
            country_df = country_df.dropna(subset=["total_deaths"])
            if country_df.empty:
                continue
            fig12.add_trace(
                go.Scatter(
                    x=country_df["date"],
                    y=country_df["total_deaths"],
                    mode="lines",
                    fill="tozeroy",
                    name=country,
                    opacity=0.62,
                    line=dict(color=PASTEL_PALETTE[i % len(PASTEL_PALETTE)]),
                    hovertemplate="%{x|%Y-%m-%d}<br>Total deaths: %{y:,.0f}<extra>%{fullData.name}</extra>",
                )
            )
        fig12.update_layout(legend_title_text="Country")
        fig12 = polish(
            fig12,
            "Area Chart — Total Deaths Over Time for Selected Countries",
            "Date",
            "Total deaths",
        )

    return fig1, fig2, fig3, fig4, fig5, fig6, fig_scatter, fig7, fig8, fig9, fig10, fig11, fig12


if __name__ == "__main__":
    app.run(debug=True)
