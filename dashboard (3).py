from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html


# ── Paths & constants ─────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
DATA_CLEAN  = BASE_DIR / "data" / "cleaned_data.csv"
DATA_SAMPLE = BASE_DIR / "data" / "sample_cleaned_data.csv"

USECOLS = [
    "short_name", "age", "nationality_name", "overall",
    "potential", "club_name", "player_positions",
    "wage_eur", "value_eur", "position_group",
]
CHUNK_ROWS  = 200_000
MAX_ROWS    = int(os.environ.get("FIFA_DASH_MAX_ROWS", "500_000"))
SAMPLE_SIZE = 5_000
TEMPLATE    = "plotly_white"
H           = 440

# ── Monochromatic palettes ────────────────────────────────────────────────────
_TEALS = ["#004d40", "#00695c", "#009688", "#26a69a", "#80cbc4"]


# ── Data loading ──────────────────────────────────────────────────────────────
def _resolve_path() -> Path:
    if DATA_CLEAN.exists():
        return DATA_CLEAN
    if DATA_SAMPLE.exists():
        return DATA_SAMPLE
    raise FileNotFoundError(
        "Add data/cleaned_data.csv (or data/sample_cleaned_data.csv) under the project root."
    )


def _load(path: Path) -> pd.DataFrame:
    parts, n = [], 0
    for chunk in pd.read_csv(
        path,
        usecols=[c for c in USECOLS if c != "position_group"],
        chunksize=CHUNK_ROWS,
        low_memory=False,
    ):
        parts.append(chunk)
        n += len(chunk)
        if n >= MAX_ROWS:
            break
    df = pd.concat(parts, ignore_index=True)
    if len(df) > MAX_ROWS:
        df = df.iloc[:MAX_ROWS].copy()

    df = df.dropna(subset=["age", "overall", "club_name", "nationality_name"])
    df["age"]     = df["age"].astype(int)
    df["overall"] = df["overall"].astype(int)
    df["wage_eur"]  = pd.to_numeric(df["wage_eur"],  errors="coerce").fillna(0)
    df["value_eur"] = pd.to_numeric(df["value_eur"], errors="coerce").fillna(0)

    pos_map = {
        "ST": "Forward",  "CF": "Forward",  "LW": "Forward",  "RW": "Forward",
        "CAM": "Midfielder", "CM": "Midfielder", "LM": "Midfielder",
        "RM": "Midfielder",  "CDM": "Midfielder",
        "CB": "Defender",  "LB": "Defender",  "RB": "Defender",
        "LWB": "Defender", "RWB": "Defender",
        "GK": "Goalkeeper",
    }
    df["main_position"]  = df["player_positions"].str.split(",").str[0].str.strip()
    df["position_group"] = df["main_position"].map(pos_map).fillna("Other")
    return df


PATH      = _resolve_path()
DF        = _load(PATH)
DF_SAMPLE = DF.sample(min(SAMPLE_SIZE, len(DF)), random_state=42)

AGE_MIN    = int(DF["age"].min())
AGE_MAX    = int(DF["age"].max())
POS_GROUPS = sorted(DF["position_group"].unique())
SLIDER_MARK_STYLE           = {"color": "#f8fafc", "fontWeight": "800"}
DROPDOWN_OPTION_LABEL_STYLE = {"color": "#0b1626", "fontWeight": "700"}


def _dropdown_option(label: str, value: str) -> dict:
    return {"label": html.Span(label, style=DROPDOWN_OPTION_LABEL_STYLE), "value": value}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _empty(msg: str = "No data for this selection.") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        annotations=[dict(
            text=msg, xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=15, color="#888"),
        )],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template=TEMPLATE,
        height=H,
    )
    return fig


def _filter_df(df: pd.DataFrame, ftype: str, fval: str | None,
               age_lo: int, age_hi: int) -> pd.DataFrame:
    out = df[(df["age"] >= age_lo) & (df["age"] <= age_hi)].copy()
    if not fval or fval == "__ALL__":
        return out
    if ftype == "Club":
        return out[out["club_name"] == fval]
    return out[out["nationality_name"] == fval]


# ── Reusable UI components ────────────────────────────────────────────────────
def _filter_panel(pfx: str) -> html.Div:
    return html.Div(
        [
            html.Div([
                html.Label("Filter by", className="ctrl-label"),
                dcc.Dropdown(
                    id=f"{pfx}-ftype",
                    options=[_dropdown_option("Club", "Club"),
                             _dropdown_option("Country", "Country")],
                    value="Club", clearable=False,
                    style={"minWidth": "170px"},
                ),
            ], style={"flex": "0 0 auto"}),

            html.Div([
                html.Label("Club / Country", className="ctrl-label"),
                dcc.Dropdown(
                    id=f"{pfx}-fval",
                    options=[], value=None, clearable=False,
                    style={"minWidth": "260px"},
                ),
            ], style={"flex": "1 1 280px"}),

            html.Div([
                html.Label("Age range", className="ctrl-label"),
                dcc.RangeSlider(
                    id=f"{pfx}-age",
                    min=AGE_MIN, max=AGE_MAX, step=1,
                    value=[AGE_MIN, AGE_MAX],
                    marks={
                        i: {"label": str(i), "style": SLIDER_MARK_STYLE}
                        for i in range(AGE_MIN, AGE_MAX + 1, 5)
                    },
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], style={"flex": "1 1 360px", "paddingTop": "8px"}),
        ],
        className="ctrl-bar",
    )


def _chart_card(graph_id: str, title: str, accent: str = "green") -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Span("\u26bd", className="card-icon"),
                    html.H3(title, className="card-title"),
                ],
                className="card-heading",
            ),
            dcc.Graph(id=graph_id, className="chart-graph"),
        ],
        className=f"chart-card accent-{accent}",
    )


def _kpi_card(label: str, value: str, note: str, accent: str = "green") -> html.Div:
    return html.Div(
        [
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.Div(note, className="kpi-note"),
            html.Div(className="kpi-meter"),
        ],
        className=f"kpi-card accent-{accent}",
    )


# ── Shared chart layout helper ────────────────────────────────────────────────
def _apply_chart_rules(fig: go.Figure, title: str, xlab: str, ylab: str,
                       height: int = H, xangle: int = 0,
                       legend_x: float = 0.98, legend_y: float = 0.98,
                       legend_xanchor: str = "right", legend_yanchor: str = "top") -> go.Figure:
    """Apply the shared Rules (border, zero baseline, horizontal labels,
    black titles, gridlines, legend inside top-right)."""
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(color="black", size=15)),
        xaxis=dict(
            title=dict(text=xlab, font=dict(color="black")),
            tickfont=dict(color="black"),
            tickangle=xangle,
            showgrid=False,
            showline=True, linecolor="black",
        ),
        yaxis=dict(
            title=dict(text=ylab, font=dict(color="black")),
            tickfont=dict(color="black"),
            showgrid=True, gridcolor="lightgray",
            showline=True, linecolor="black",
            rangemode="tozero",          # Rule 2: Y starts at 0
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        shapes=[dict(                    # Rule 1: black border
            type="rect", xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color="black", width=2),
        )],
        legend=dict(                     # Rule 5: legend inside, top-right
            x=legend_x, y=legend_y,
            xanchor=legend_xanchor, yanchor=legend_yanchor,
            bordercolor="black", borderwidth=1, bgcolor="white",
            font=dict(color="black"),
        ),
        height=height,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def column_chart(data: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Column chart — Top N clubs by average overall rating.
    Rules: border, zero baseline, horizontal labels, magnitudes outside,
    legend inside top-right, winner on far LEFT (highest leftmost)."""
    df = (
        data.groupby("club_name")["overall"]
        .mean()
        .sort_values(ascending=False)   # winner first = far left
        .head(top_n)
        .reset_index()
    )
    # Rule 10 (column): winner on far left → descending order
    df = df.sort_values("overall", ascending=False)

    colors = ["lightblue"] * len(df)
    colors[0] = "lightgreen"           # highest (leftmost) highlighted

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["club_name"],
        y=df["overall"],
        text=df["overall"].round(1),
        textposition="outside",
        textfont=dict(color="black"),
        marker_color=colors,
        marker_line_color="black", marker_line_width=0.5,
        name="Avg Rating",
    ))
    _apply_chart_rules(fig, "Average Overall Rating by Club", "Club", "Average Rating", height=550)
    fig.update_traces(texttemplate="%{text:.1f}")
    return fig


def bar_chart(data: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart — Top N players by overall rating.
    Rules: border, zero baseline, horizontal labels, magnitudes outside,
    legend inside bottom-right (to avoid bars), winner at TOP."""
    df = (
        data.nlargest(top_n * 3, "overall")
        .drop_duplicates("short_name")
        .head(top_n)[["short_name", "overall", "club_name"]]
    )
    # Rule 9 (bar): order top → bottom; ascending so plotly places highest at top
    df = df.sort_values("overall", ascending=True)

    normal = df.iloc[:-1]
    best   = df.iloc[-1:]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=normal["overall"], y=normal["short_name"],
        orientation="h",
        marker_color="lightblue",
        marker_line_color="black", marker_line_width=0.5,
        name="Players",
        text=normal["overall"].round(1),
        textposition="outside",
        textfont=dict(color="black"),
    ))
    fig.add_trace(go.Bar(
        x=best["overall"], y=best["short_name"],
        orientation="h",
        marker_color="lightgreen",
        marker_line_color="black", marker_line_width=0.5,
        name="Top Player",
        text=best["overall"].round(1),
        textposition="outside",
        textfont=dict(color="black"),
    ))
    _apply_chart_rules(
        fig, "Top Players in FIFA", "Rating", "Player", height=550,
        legend_x=0.98, legend_y=0.02, legend_xanchor="right", legend_yanchor="bottom",
    )
    # For horizontal bar: gridlines on x-axis, not y
    fig.update_layout(
        xaxis=dict(
            title=dict(text="Rating", font=dict(color="black")),
            tickfont=dict(color="black"),
            showgrid=True, gridcolor="lightgray",
            showline=True, linecolor="black",
            rangemode="tozero",
        ),
        yaxis=dict(
            title=dict(text="Player", font=dict(color="black")),
            tickfont=dict(color="black"),
            showgrid=False, showline=True, linecolor="black",
        ),
    )
    fig.update_traces(texttemplate="%{x:.1f}")
    return fig


# ── Tab layouts ───────────────────────────────────────────────────────────────

tab_comparison = html.Div([
    html.Div([
        html.Div([
            html.Label("Top N clubs / players", className="ctrl-label"),
            dcc.Slider(
                id="cmp-topn", min=5, max=20, step=1, value=10,
                marks={i: {"label": str(i), "style": SLIDER_MARK_STYLE} for i in [5, 10, 15, 20]},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ], style={"flex": "1 1 320px", "paddingTop": "8px"}),

        html.Div([
            html.Label("Clustered Bar metric", className="ctrl-label"),
            dcc.RadioItems(
                id="cmp-metric",
                options=[
                    {"label": "Wage (€)", "value": "wage"},
                    {"label": "Overall",  "value": "overall"},
                    {"label": "Both (normalized)", "value": "both"},
                ],
                value="both", inline=True,
                inputStyle={"marginRight": "4px"},
                style={"marginTop": "6px"},
            ),
        ], style={"flex": "1 1 360px"}),
    ], className="ctrl-bar"),

    html.Div([
        _chart_card("cmp-col", "Top Clubs", "blue"),
        _chart_card("cmp-bar", "Elite Players", "green"),
    ], className="chart-row"),

    html.Div([
        _chart_card("cmp-stk-col", "Position Depth", "purple"),
        _chart_card("cmp-stk-bar", "Nationality Mix", "blue"),
    ], className="chart-row"),

    html.Div([
        _chart_card("cmp-clust-col", "Rating Bands", "green"),
        _chart_card("cmp-clust-bar", "Wage vs Rating", "purple"),
    ], className="chart-row"),
])

tab_relationship = html.Div([
    _filter_panel("rel"),
    html.Div([
        _chart_card("rel-scatter", "Age vs Overall", "blue"),
        _chart_card("rel-bubble",  "Potential Market", "green"),
    ], className="chart-row"),
])

tab_distribution = html.Div([
    html.Div([
        html.Div([
            html.Label("Position group", className="ctrl-label"),
            dcc.Dropdown(
                id="dist-pos",
                options=[_dropdown_option("All positions", "__ALL__")]
                        + [_dropdown_option(p, p) for p in POS_GROUPS],
                value="__ALL__", clearable=False,
                style={"minWidth": "200px"},
            ),
        ], style={"flex": "0 0 auto"}),

        html.Div([
            html.Label("Box & Violin metric", className="ctrl-label"),
            dcc.RadioItems(
                id="dist-metric",
                options=[
                    {"label": "Overall Rating", "value": "overall"},
                    {"label": "Wage (€)",        "value": "wage_eur"},
                ],
                value="overall", inline=True,
                inputStyle={"marginRight": "4px"},
                style={"marginTop": "6px"},
            ),
        ], style={"flex": "1 1 260px"}),
    ], className="ctrl-bar"),

    html.Div([
        _chart_card("dist-hist", "Age Distribution", "blue"),
    ], className="chart-row chart-row-single"),

    html.Div([
        _chart_card("dist-box",    "Position Range",  "purple"),
        _chart_card("dist-violin", "Rating Density",  "green"),
    ], className="chart-row"),
])

tab_timeseries = html.Div([
    _filter_panel("ts"),
    html.Div([
        html.Label("Time Series Options", className="ctrl-label"),
        dcc.RadioItems(
            id="ts-ma-type",
            options=[
                {"label": "Raw Data Only",               "value": "raw"},
                {"label": "With 5-Age Moving Average",   "value": "ma5"},
                {"label": "With 10-Age Moving Average",  "value": "ma10"},
                {"label": "Both MA (5 & 10)",            "value": "both_ma"},
            ],
            value="ma5", inline=True,
            inputStyle={"marginRight": "4px"},
            style={"marginTop": "6px"},
        ),
    ], style={"flex": "1 1 360px", "paddingTop": "8px"}, className="ctrl-bar"),
    html.Div([
        _chart_card("ts-line", "Rating Trend Over Age (Ordered X-axis)", "blue"),
        _chart_card("ts-area", "Player Distribution Over Age (Volume)", "green"),
    ], className="chart-row"),
    html.Div([
        _chart_card("ts-stacked-area", "Multi-Position Volume Trend", "purple"),
    ], className="chart-row chart-row-single"),
])


# ── App + layout ──────────────────────────────────────────────────────────────
app = Dash(__name__)
app.title = "FIFA Data Visualization Dashboard"

app.layout = html.Div([
    html.Div(className="stadium-hologram"),

    html.Div([
        html.Div([
            html.Div("\u26bd GLOBAL FOOTBALL ANALYTICS HUB", className="eyebrow"),
            html.H1("Match Day Performance Insights", className="app-title"),
            html.P(
                f"FIFA Player Data | {PATH.name} | {len(DF):,} players analyzed | "
                "all 13 visualization modules",
                className="app-subtitle",
            ),
        ], className="hero-copy"),
        html.Div([
            html.Div(className="pitch-line center"),
            html.Div(className="pitch-line box-left"),
            html.Div(className="pitch-line box-right"),
            html.Div(className="radar-ring ring-1"),
            html.Div(className="radar-ring ring-2"),
            html.Div(className="radar-sweep"),
            html.Div("\u26bd", className="animated-ball"),
        ], className="hero-visual"),
        html.Div([
            _kpi_card("Players",    f"{len(DF):,}",         "loaded dataset",  "green"),
            _kpi_card("Age Window", f"{AGE_MIN}-{AGE_MAX}", "available range", "blue"),
            _kpi_card("Positions",  f"{len(POS_GROUPS)}",   "player groups",   "purple"),
        ], className="kpi-grid"),
    ], className="hero"),

    dcc.Tabs(
        id="main-tabs",
        value="tab-cmp",
        children=[
            dcc.Tab(label="Comparison",   value="tab-cmp",  children=tab_comparison,
                    selected_style={"borderTop": "3px solid #24c8ff",
                                    "fontWeight": "700", "color": "#07111c"}),
            dcc.Tab(label="Relationship", value="tab-rel",  children=tab_relationship,
                    selected_style={"borderTop": "3px solid #2cff8f",
                                    "fontWeight": "700", "color": "#07111c"}),
            dcc.Tab(label="Distribution", value="tab-dist", children=tab_distribution,
                    selected_style={"borderTop": "3px solid #a855f7",
                                    "fontWeight": "700", "color": "#07111c"}),
            dcc.Tab(label="Time Series",  value="tab-ts",   children=tab_timeseries,
                    selected_style={"borderTop": "3px solid #a855f7",
                                    "fontWeight": "700", "color": "#07111c"}),
        ],
        className="tabs-shell",
    ),
], style={
    "fontFamily": "Poppins, Montserrat, system-ui, -apple-system, sans-serif",
    "width": "100%", "maxWidth": "none",
    "margin": "0",
    "padding": "1.25rem clamp(1rem, 2.2vw, 2.75rem) 2.5rem",
})


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

# ── Tab 1 — Comparison ────────────────────────────────────────────────────────
@callback(
    Output("cmp-col",       "figure"),
    Output("cmp-bar",       "figure"),
    Output("cmp-stk-col",   "figure"),
    Output("cmp-stk-bar",   "figure"),
    Output("cmp-clust-col", "figure"),
    Output("cmp-clust-bar", "figure"),
    Input("cmp-topn",   "value"),
    Input("cmp-metric", "value"),
)
def update_comparison(top_n: int, metric: str):
    top_n = top_n or 10

    # 1. Column chart
    col_fig = column_chart(DF, top_n)

    # 2. Bar chart
    bar_fig = bar_chart(DF, top_n)

    top_clubs = DF["club_name"].value_counts().head(top_n).index.tolist()
    stk_df    = DF[DF["club_name"].isin(top_clubs)]

    # ── 3. Stacked Column ────────────────────────────────────────────────────
    # Color rule: light→dark same hue per segment; highest-total club gets greens
    stk_grp      = stk_df.groupby(["club_name", "position_group"]).size().reset_index(name="count")
    club_totals  = stk_grp.groupby("club_name")["count"].sum()
    highest_club = club_totals.idxmax()
    pos_groups   = sorted(stk_grp["position_group"].unique())

    blue_shades  = ["lightblue", "steelblue", "royalblue", "blue", "darkblue"]
    green_shades = ["lightgreen", "#66bb6a", "seagreen", "green", "darkgreen"]

    stk_col_fig = go.Figure()
    for i, pos in enumerate(pos_groups):
        pos_data = stk_grp[stk_grp["position_group"] == pos]
        colors = [
            green_shades[i % len(green_shades)] if c == highest_club else blue_shades[i % len(blue_shades)]
            for c in pos_data["club_name"]
        ]
        stk_col_fig.add_trace(go.Bar(
            x=pos_data["club_name"], y=pos_data["count"],
            name=pos,
            marker_color=colors,
            marker_line_color="black", marker_line_width=0.5,
            text=pos_data["count"],
            textposition="inside",
            textfont=dict(color="black", size=9),
        ))

    stk_col_fig.update_layout(barmode="stack")
    _apply_chart_rules(
        stk_col_fig,
        f"Stacked Column — Player Count by Position (Top {top_n} Clubs)",
        "Club", "Number of Players",
    )
    stk_col_fig.update_layout(
        legend=dict(title=dict(text="Position", font=dict(color="black")),
                    x=0.98, y=0.98, xanchor="right", yanchor="top",
                    bordercolor="black", borderwidth=1, bgcolor="white",
                    font=dict(color="black")),
    )

    # ── 4. Stacked Bar ───────────────────────────────────────────────────────
    # Color rule: light→dark oranges; highest-total club gets greens
    top_nat = DF["nationality_name"].value_counts().head(8).index.tolist()
    nat_df  = stk_df.copy()
    nat_df["nationality_label"] = nat_df["nationality_name"].apply(
        lambda x: x if x in top_nat else "Other"
    )
    nat_grp = nat_df.groupby(["club_name", "nationality_label"]).size().reset_index(name="count")

    # Order: top → bottom = highest total at top (ascending for plotly horizontal)
    club_order       = (nat_grp.groupby("club_name")["count"].sum()
                        .sort_values(ascending=True).index.tolist())
    highest_nat_club = club_order[-1]
    nat_groups       = sorted(nat_grp["nationality_label"].unique())

    orange_shades = ["#ffe0b2", "#ffb74d", "#fb8c00", "#e65100", "#bf360c",
                     "#ff8a65", "#ff7043", "#f4511e", "#d84315"]
    green_shades2 = ["lightgreen", "#81c784", "#66bb6a", "seagreen", "green",
                     "darkgreen", "#a5d6a7", "#43a047", "#2e7d32", "#1b5e20"]

    stk_bar_fig = go.Figure()
    for i, nat in enumerate(nat_groups):
        nat_data = nat_grp[nat_grp["nationality_label"] == nat]
        colors = [
            green_shades2[i % len(green_shades2)] if c == highest_nat_club else orange_shades[i % len(orange_shades)]
            for c in nat_data["club_name"]
        ]
        stk_bar_fig.add_trace(go.Bar(
            x=nat_data["count"], y=nat_data["club_name"],
            name=nat, orientation="h",
            marker_color=colors,
            marker_line_color="black", marker_line_width=0.5,
            text=nat_data["count"],
            textposition="inside",
            textfont=dict(color="black", size=9),
        ))

    stk_bar_fig.update_layout(barmode="stack")
    _apply_chart_rules(
        stk_bar_fig,
        f"Stacked Bar — Nationality Distribution (Top {top_n} Clubs)",
        "Number of Players", "Club",
    )
    stk_bar_fig.update_layout(
        yaxis=dict(categoryorder="array", categoryarray=club_order,
                   title=dict(text="Club", font=dict(color="black")),
                   tickfont=dict(color="black"), showgrid=False,
                   showline=True, linecolor="black"),
        xaxis=dict(title=dict(text="Number of Players", font=dict(color="black")),
                   tickfont=dict(color="black"),
                   showgrid=True, gridcolor="lightgray",
                   showline=True, linecolor="black", rangemode="tozero"),
        legend=dict(title=dict(text="Nationality", font=dict(color="black")),
                    x=0.98, y=0.98, xanchor="right", yanchor="top",
                    bordercolor="black", borderwidth=1, bgcolor="white",
                    font=dict(color="black")),
    )

    # ── 5. Clustered Column ──────────────────────────────────────────────────
    # Color rule: light→dark purples per rating band; highest-total position gets greens
    bins   = [40, 60, 70, 80, 90, 100]
    labels = ["40–60", "60–70", "70–80", "80–90", "90+"]
    rated  = DF.copy()
    rated["rating_group"] = pd.cut(rated["overall"], bins=bins, labels=labels, right=False)
    clust_grp   = (rated.groupby(["position_group", "rating_group"], observed=True)
                   .size().reset_index(name="count"))
    pos_totals  = clust_grp.groupby("position_group")["count"].sum()
    highest_pos = pos_totals.idxmax()

    purple_shades = ["#e1bee7", "#ce93d8", "#ab47bc", "#7b1fa2", "#4a148c"]
    green_shades3 = ["lightgreen", "#81c784", "seagreen", "green", "darkgreen"]

    clust_col_fig = go.Figure()
    for i, rg in enumerate(labels):
        rg_data = clust_grp[clust_grp["rating_group"] == rg]
        colors = [
            green_shades3[i % len(green_shades3)] if p == highest_pos else purple_shades[i % len(purple_shades)]
            for p in rg_data["position_group"]
        ]
        clust_col_fig.add_trace(go.Bar(
            x=rg_data["position_group"], y=rg_data["count"],
            name=rg,
            marker_color=colors,
            marker_line_color="black", marker_line_width=0.5,
            text=rg_data["count"],
            textposition="outside",
            textfont=dict(color="black", size=9),
        ))

    clust_col_fig.update_layout(barmode="group")
    _apply_chart_rules(
        clust_col_fig,
        "Clustered Column — Players by Position and Rating Group",
        "Position", "Number of Players",
    )
    clust_col_fig.update_layout(
        legend=dict(title=dict(text="Rating Range", font=dict(color="black")),
                    x=0.98, y=0.98, xanchor="right", yanchor="top",
                    bordercolor="black", borderwidth=1, bgcolor="white",
                    font=dict(color="black")),
    )

    # ── 6. Clustered Bar ─────────────────────────────────────────────────────
    # Color rule: light→dark blues/greens; highest club gets greens
    wage_rating = (
        DF[DF["club_name"].isin(top_clubs)]
        .groupby("club_name")
        .agg(avg_wage=("wage_eur", "mean"), avg_overall=("overall", "mean"))
        .reset_index()
    )

    if metric == "both":
        wage_rating["wage_score"]    = (wage_rating["avg_wage"]    / wage_rating["avg_wage"].max())    * 100
        wage_rating["overall_score"] = (wage_rating["avg_overall"] / wage_rating["avg_overall"].max()) * 100
        wage_rating = wage_rating.sort_values("wage_score", ascending=True)
        highest_club_wr = wage_rating.iloc[-1]["club_name"]

        clust_bar_fig = go.Figure()
        clust_bar_fig.add_trace(go.Bar(
            name="Avg Wage (norm. 0–100)",
            y=wage_rating["club_name"], x=wage_rating["wage_score"],
            orientation="h",
            marker_color=["lightgreen" if c == highest_club_wr else "lightblue"
                          for c in wage_rating["club_name"]],
            marker_line_color="black", marker_line_width=0.5,
            text=wage_rating["avg_wage"].apply(lambda v: f"€{v:,.0f}"),
            textposition="outside", textfont=dict(color="black", size=9),
        ))
        clust_bar_fig.add_trace(go.Bar(
            name="Avg Overall (norm. 0–100)",
            y=wage_rating["club_name"], x=wage_rating["overall_score"],
            orientation="h",
            marker_color=["seagreen" if c == highest_club_wr else "steelblue"
                          for c in wage_rating["club_name"]],
            marker_line_color="black", marker_line_width=0.5,
            text=wage_rating["avg_overall"].apply(lambda v: f"{v:.1f}"),
            textposition="outside", textfont=dict(color="black", size=9),
        ))
        x_label = "Normalized Score (0–100)"

    elif metric == "wage":
        wage_rating = wage_rating.sort_values("avg_wage", ascending=True)
        highest_club_wr = wage_rating.iloc[-1]["club_name"]
        clust_bar_fig = go.Figure()
        clust_bar_fig.add_trace(go.Bar(
            name="Avg Wage (€)",
            y=wage_rating["club_name"], x=wage_rating["avg_wage"],
            orientation="h",
            marker_color=["lightgreen" if c == highest_club_wr else "lightblue"
                          for c in wage_rating["club_name"]],
            marker_line_color="black", marker_line_width=0.5,
            text=wage_rating["avg_wage"].apply(lambda v: f"€{v:,.0f}"),
            textposition="outside", textfont=dict(color="black", size=9),
        ))
        x_label = "Average Wage (€)"

    else:  # overall
        wage_rating = wage_rating.sort_values("avg_overall", ascending=True)
        highest_club_wr = wage_rating.iloc[-1]["club_name"]
        clust_bar_fig = go.Figure()
        clust_bar_fig.add_trace(go.Bar(
            name="Avg Overall Rating",
            y=wage_rating["club_name"], x=wage_rating["avg_overall"],
            orientation="h",
            marker_color=["lightgreen" if c == highest_club_wr else "lightblue"
                          for c in wage_rating["club_name"]],
            marker_line_color="black", marker_line_width=0.5,
            text=wage_rating["avg_overall"].apply(lambda v: f"{v:.1f}"),
            textposition="outside", textfont=dict(color="black", size=9),
        ))
        x_label = "Average Overall Rating"

    clust_bar_fig.update_layout(barmode="group")
    _apply_chart_rules(
        clust_bar_fig,
        f"Clustered Bar — Avg Wage vs Avg Rating  (Top {top_n} Clubs)",
        x_label, "Club",
    )
    clust_bar_fig.update_layout(
        xaxis=dict(
            title=dict(text=x_label, font=dict(color="black")),
            tickfont=dict(color="black"),
            showgrid=True, gridcolor="lightgray",
            showline=True, linecolor="black", rangemode="tozero",
        ),
        yaxis=dict(
            title=dict(text="Club", font=dict(color="black")),
            tickfont=dict(color="black"),
            showgrid=False, showline=True, linecolor="black",
        ),
        legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top",
                    bordercolor="black", borderwidth=1, bgcolor="white",
                    font=dict(color="black")),
    )

    return col_fig, bar_fig, stk_col_fig, stk_bar_fig, clust_col_fig, clust_bar_fig


# ── Tab 2 — Relationship ──────────────────────────────────────────────────────
@callback(
    Output("rel-fval", "options"),
    Output("rel-fval", "value"),
    Input("rel-ftype", "value"),
)
def rel_options(ftype):
    all_opt = _dropdown_option("All", "__ALL__")
    if ftype == "Club":
        vals = sorted(DF["club_name"].dropna().astype(str).unique())
    else:
        vals = sorted(DF["nationality_name"].dropna().astype(str).unique())
    return [all_opt] + [_dropdown_option(v, v) for v in vals], "__ALL__"


@callback(
    Output("rel-scatter", "figure"),
    Output("rel-bubble",  "figure"),
    Input("rel-ftype", "value"),
    Input("rel-fval",  "value"),
    Input("rel-age",   "value"),
)
def update_relationship(ftype, fval, age_range):
    age_lo, age_hi = (AGE_MIN, AGE_MAX) if not age_range else (int(age_range[0]), int(age_range[1]))
    sub = _filter_df(DF_SAMPLE, ftype, fval, age_lo, age_hi)

    if sub.empty:
        msg = "No data for this filter — widen the age range or select All."
        return _empty(msg), _empty(msg)

    sub = sub.copy()
    threshold = sub["overall"].quantile(0.95)
    sub["Player Type"] = sub["overall"].apply(
        lambda x: "Top Rated (Outlier)" if x >= threshold else "Player"
    )
    top10_scatter = set(sub.nlargest(10, "overall")["short_name"])
    sub["label"] = sub["short_name"].apply(lambda n: n if n in top10_scatter else "")
    scatter_fig = px.scatter(
        sub, x="age", y="overall",
        color="Player Type",
        color_discrete_map={"Player": "lightblue", "Top Rated (Outlier)": "lightcoral"},
        text="label", hover_name="short_name",
        title="<b>Age vs Overall Rating</b>",
        labels={"age": "Age", "overall": "Overall Rating", "Player Type": "Player Type"},
        template=TEMPLATE, height=H,
    )
    scatter_fig.update_traces(
        marker=dict(line=dict(width=1, color="black")),
        textposition="top center", textfont=dict(size=9),
    )
    scatter_fig.update_layout(
        title_x=0.5,
        legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top",
                    bgcolor="white", bordercolor="black", borderwidth=1),
        xaxis=dict(showgrid=True, gridcolor="lightgrey",
                   showline=True, linecolor="black", mirror=True),
        yaxis=dict(range=[0, sub["overall"].max() + 5],
                   showgrid=True, gridcolor="lightgrey",
                   showline=True, linecolor="black", mirror=True),
        shapes=[dict(type="rect", xref="paper", yref="paper",
                     x0=0, y0=0, x1=1, y1=1, line=dict(color="black", width=2))],
        plot_bgcolor="white", paper_bgcolor="white",
    )

    bub = sub.dropna(subset=["value_eur", "potential"])
    bub = bub[bub["value_eur"] > 0].copy()
    if bub.empty:
        bubble_fig = _empty("No market-value data for this selection.")
    else:
        bp_thresh = bub["potential"].quantile(0.95)
        bub["Player Type"] = bub["potential"].apply(
            lambda x: "High Potential (Outlier)" if x >= bp_thresh else "Player"
        )
        top10_bubble = set(bub.nlargest(10, "potential")["short_name"])
        bub["label"] = bub["short_name"].apply(lambda n: n if n in top10_bubble else "")
        bubble_fig = px.scatter(
            bub, x="age", y="potential",
            size="value_eur", size_max=50,
            color="Player Type",
            color_discrete_map={"Player": "lightblue", "High Potential (Outlier)": "lightgreen"},
            text="label", hover_name="short_name",
            title="<b>Age vs Potential</b>  (Bubble Size = Market Value €)",
            labels={"age": "Age", "potential": "Potential",
                    "value_eur": "Value (€)", "Player Type": "Player Type"},
            template=TEMPLATE, height=H,
        )
        bubble_fig.update_traces(
            marker=dict(line=dict(width=1, color="black")),
            textposition="top center", textfont=dict(size=9),
        )
        bubble_fig.update_layout(
            title_x=0.5,
            legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top",
                        bgcolor="white", bordercolor="black", borderwidth=1),
            xaxis=dict(showgrid=True, gridcolor="lightgrey",
                       showline=True, linecolor="black", mirror=True),
            yaxis=dict(range=[0, bub["potential"].max() + 5],
                       showgrid=True, gridcolor="lightgrey",
                       showline=True, linecolor="black", mirror=True),
            shapes=[dict(type="rect", xref="paper", yref="paper",
                         x0=0, y0=0, x1=1, y1=1, line=dict(color="black", width=2))],
            plot_bgcolor="white", paper_bgcolor="white",
        )

    return scatter_fig, bubble_fig


# ── Tab 3 — Distribution ──────────────────────────────────────────────────────
@callback(
    Output("dist-hist",   "figure"),
    Output("dist-box",    "figure"),
    Output("dist-violin", "figure"),
    Input("dist-pos",    "value"),
    Input("dist-metric", "value"),
)
def update_distribution(position, metric):
    sub = DF_SAMPLE.copy()
    if position != "__ALL__":
        sub = sub[sub["position_group"] == position]
    if sub.empty:
        e = _empty()
        return e, e, e

    mlabel    = "Overall Rating" if metric == "overall" else "Wage (€)"
    pos_label = position if position != "__ALL__" else "All Positions"

    # Histogram
    hist_fig = px.histogram(
        sub, x="age", nbins=20,
        title=f"Histogram — Age Distribution  ({pos_label})",
        labels={"age": "Age", "count": "Number of Players"},
        color_discrete_sequence=["lightblue"],
        template=TEMPLATE, height=H,
    )
    hist_fig.update_layout(bargap=0.05, xaxis_title="Age", yaxis_title="Number of Players")

    # Box chart
    box_sub = DF_SAMPLE.copy()
    box_fig = px.box(
        box_sub, x="main_position", y="wage_eur",
        color="main_position", points="outliers",
        color_discrete_sequence=_TEALS,
        title="Salary Distribution by Player Position",
        labels={"main_position": "Position", "wage_eur": "Wage (EUR)"},
        template=TEMPLATE, height=H,
    )
    box_fig.update_layout(
        showlegend=False, xaxis_tickangle=0,
        xaxis_title="Position", yaxis_title="Wage (EUR)",
        yaxis_range=[0, 250000],
        plot_bgcolor="white", paper_bgcolor="white",
    )

    # ── Violin chart ─────────────────────────────────────────────────────────
    # Rule 1  — black border frame
    # Rule 2  — Y-axis starts at 0
    # Rule 3  — all labels horizontal (x ticks, legend text, titles); Y-title vertical (default)
    # Rule 4  — median value annotated on the white dot
    # Rule 5  — legend top-right inside border
    # Rule 6  — Main Title + X-axis Title + Y-axis Title
    # Rule 7  — lightblue silhouette; white median dot; black IQR box, whiskers, titles/labels
    # Rule 8  — horizontal gridlines on Y-axis
    pos_order = sorted(box_sub["position_group"].unique())

    violin_fig = go.Figure()
    for pos in pos_order:
        pos_data = box_sub[box_sub["position_group"] == pos][metric].dropna()
        if pos_data.empty:
            continue
        median_val = round(float(pos_data.median()), 1)
        violin_fig.add_trace(go.Violin(
            x=[pos] * len(pos_data),
            y=pos_data,
            name=pos,
            # Rule 7: lightblue silhouette fill, black outline/whiskers
            fillcolor="lightblue",
            line_color="black",          # black IQR box outline, whiskers, and violin border
            opacity=0.75,
            # Rule 7: white-filled IQR box, black outline & whiskers
            box_visible=True,
            box=dict(fillcolor="white", line=dict(color="black", width=1.5)),
            meanline_visible=False,
            points=False,
        ))
        # Rule 4: median value directly on the white dot
        violin_fig.add_annotation(
            x=pos, y=median_val,
            text=f"<b>{median_val:,.1f}</b>",
            showarrow=False,
            font=dict(color="black", size=10),
            yshift=12,
            xanchor="center",
        )

    violin_fig.update_layout(
        # Rule 6: three titles
        title=dict(
            text=f"Violin — {mlabel} Distribution by Position Group",
            x=0.5, font=dict(color="black", size=15),
        ),
        xaxis=dict(
            title=dict(text="Position Group", font=dict(color="black")),
            tickfont=dict(color="black"),
            tickangle=0,             # Rule 3: horizontal x-axis labels
            showgrid=False,
            showline=True, linecolor="black",
        ),
        yaxis=dict(
            title=dict(text=mlabel, font=dict(color="black")),
            tickfont=dict(color="black"),
            rangemode="tozero",      # Rule 2: Y starts at 0
            showgrid=True, gridcolor="lightgray",   # Rule 8
            showline=True, linecolor="black",
            zeroline=True, zerolinecolor="lightgray", zerolinewidth=1,
        ),
        violingap=0.3,
        violinmode="group",
        plot_bgcolor="white", paper_bgcolor="white",
        # Rule 1: black border
        shapes=[dict(type="rect", xref="paper", yref="paper",
                     x0=0, y0=0, x1=1, y1=1,
                     line=dict(color="black", width=2))],
        # Rule 5: legend top-right inside border, in the free space right of violins
        legend=dict(
            title=dict(text="Position", font=dict(color="black")),
            x=0.98, y=0.98, xanchor="right", yanchor="top",
            bordercolor="black", borderwidth=1, bgcolor="white",
            font=dict(color="black"),
        ),
        margin=dict(r=160),
        showlegend=True,
        height=H,
    )

    return hist_fig, box_fig, violin_fig


# ── Tab 4 — Time Series ───────────────────────────────────────────────────────
@callback(
    Output("ts-fval", "options"),
    Output("ts-fval", "value"),
    Input("ts-ftype", "value"),
)
def ts_options(ftype):
    all_opt = _dropdown_option("All", "__ALL__")
    if ftype == "Club":
        vals = sorted(DF["club_name"].dropna().astype(str).unique())
    else:
        vals = sorted(DF["nationality_name"].dropna().astype(str).unique())
    return [all_opt] + [_dropdown_option(v, v) for v in vals], "__ALL__"


@callback(
    Output("ts-line",         "figure"),
    Output("ts-area",         "figure"),
    Output("ts-stacked-area", "figure"),
    Input("ts-ftype",   "value"),
    Input("ts-fval",    "value"),
    Input("ts-age",     "value"),
    Input("ts-ma-type", "value"),
)
def update_timeseries(ftype, fval, age_range, ma_type):
    age_lo, age_hi = (AGE_MIN, AGE_MAX) if not age_range else (int(age_range[0]), int(age_range[1]))
    sub = _filter_df(DF, ftype, fval, age_lo, age_hi)

    if sub.empty:
        e = _empty("No data — widen the age range or select All.")
        return e, e, e

    trend = sub.groupby("age", as_index=False)["overall"].mean().sort_values("age")
    trend["ma_5"]  = trend["overall"].rolling(window=5,  center=True, min_periods=1).mean()
    trend["ma_10"] = trend["overall"].rolling(window=10, center=True, min_periods=1).mean()

    all_ages = pd.Series(range(int(trend["age"].min()), int(trend["age"].max()) + 1), name="age")
    trend = all_ages.to_frame().merge(trend, on="age", how="left")

    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(
        x=trend["age"], y=trend["overall"],
        mode="lines+markers", name="Raw Data",
        line=dict(color="#636efa", width=1.5),
        marker=dict(size=4, color="#636efa"),
        opacity=0.4,
        hovertemplate="<b>Age %{x}</b><br>Mean Rating: %{y:.2f}<extra></extra>",
    ))
    if ma_type in ["ma5", "both_ma"]:
        line_fig.add_trace(go.Scatter(
            x=trend["age"], y=trend["ma_5"],
            mode="lines", name="5-Age MA",
            line=dict(color="#f97316", width=3),
            hovertemplate="<b>Age %{x}</b><br>5-Age MA: %{y:.2f}<extra></extra>",
        ))
    if ma_type in ["ma10", "both_ma"]:
        line_fig.add_trace(go.Scatter(
            x=trend["age"], y=trend["ma_10"],
            mode="lines", name="10-Age MA",
            line=dict(color="#ef4444", width=3),
            hovertemplate="<b>Age %{x}</b><br>10-Age MA: %{y:.2f}<extra></extra>",
        ))
    line_fig.update_layout(
        title="<b>Line Chart — Mean Overall Rating by Age</b><br>"
              "<sub>Ordered continuous x-axis | Trend focus | Moving averages reveal pattern</sub>",
        xaxis_title="Age (Ordered Sequence)",
        yaxis_title="Mean Overall Rating (0–100)",
        template=TEMPLATE, height=H, hovermode="x unified",
        legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1),
        plot_bgcolor="white", paper_bgcolor="white",
    )

    counts = sub.groupby("age").size().reset_index(name="players").sort_values("age")
    counts = all_ages.to_frame().merge(counts, on="age", how="left")

    area_fig = go.Figure()
    area_fig.add_trace(go.Scatter(
        x=counts["age"], y=counts["players"],
        fill="tozeroy", name="Players per Age",
        line=dict(color="#16b34a", width=2.5),
        fillcolor="rgba(22, 179, 74, 0.4)",
        hovertemplate="<b>Age %{x}</b><br>Players: %{y}<extra></extra>",
    ))
    area_fig.update_layout(
        title="<b>Area Chart — Player Count Distribution by Age</b><br>"
              "<sub>Y-axis starts at 0 | Filled area = total volume/magnitude</sub>",
        xaxis_title="Age (Ordered Sequence)",
        yaxis_title="Number of Players",
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="lightgray", range=[0, None]),
        template=TEMPLATE, height=H, hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
    )

    position_age = sub.groupby(["age", "position_group"]).size().reset_index(name="count")
    position_age = position_age.sort_values(["age", "position_group"])

    stacked_fig = go.Figure()
    pos_colors = {
        "Goalkeeper": "#1f77b4",
        "Defender":   "#ff7f0e",
        "Midfielder": "#2ca02c",
        "Forward":    "#d62728",
    }
    for pos in sorted(position_age["position_group"].unique()):
        pos_data = position_age[position_age["position_group"] == pos]
        pos_data = all_ages.to_frame().merge(pos_data, on="age", how="left")
        stacked_fig.add_trace(go.Scatter(
            x=pos_data["age"], y=pos_data["count"],
            stackgroup="one", name=pos,
            line=dict(width=0.5, color=pos_colors.get(pos, "#000")),
            fillcolor=pos_colors.get(pos, "#000"),
            hovertemplate="<b>Age %{x}</b><br>" + pos + ": %{y} players<extra></extra>",
        ))
    stacked_fig.update_layout(
        title="<b>Stacked Area Chart — Position Composition by Age</b><br>"
              "<sub>Shows category contribution over time | Y-axis starts at 0</sub>",
        xaxis_title="Age (Ordered Sequence)",
        yaxis_title="Number of Players by Position",
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="lightgray", range=[0, None]),
        template=TEMPLATE, height=H, hovermode="x unified",
        legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1),
        plot_bgcolor="white", paper_bgcolor="white",
    )

    return line_fig, area_fig, stacked_fig


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)