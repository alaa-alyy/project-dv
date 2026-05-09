from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html

# ── Paths & constants ─────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_CLEAN  = BASE_DIR / "data" / "cleaned_data.csv"
DATA_SAMPLE = BASE_DIR / "data" / "sample_cleaned_data.csv"

USECOLS = [
    "short_name", "age", "nationality_name", "overall",
    "potential", "club_name", "player_positions",
    "wage_eur", "value_eur", "position_group",
]
CHUNK_ROWS  = 200_000
MAX_ROWS    = int(os.environ.get("FIFA_DASH_MAX_ROWS", "500_000"))
SAMPLE_SIZE = 5_000   # used for scatter / bubble / box / violin to keep the UI fast
TEMPLATE    = "plotly_white"
H           = 440     # default chart height

# ── Monochromatic palettes (dark → light shades per chart) ───────────────────
_BLUES   = ["#0d47a1", "#1565c0", "#1976d2", "#2196f3", "#64b5f6"]   # stacked col (positions)
_ORANGES = ["#bf360c", "#d84315", "#e64a19", "#f4511e", "#ff7043",
            "#ff8a65", "#ffab91", "#ffccbc", "#fde0d0"]              # stacked bar (nationalities)
_PURPLES = ["#4a148c", "#6a1b9a", "#7b1fa2", "#9c27b0", "#ba68c8"]   # clustered col (rating groups)
_TEALS   = ["#004d40", "#00695c", "#009688", "#26a69a", "#80cbc4"]   # box + violin (positions)


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

    # position_group may already exist in the CSV; rebuild it to be safe
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

AGE_MIN = int(DF["age"].min())
AGE_MAX = int(DF["age"].max())
POS_GROUPS = sorted(DF["position_group"].unique())
SLIDER_MARK_STYLE = {"color": "#f8fafc", "fontWeight": "800"}
DROPDOWN_OPTION_LABEL_STYLE = {"color": "#0b1626", "fontWeight": "700"}


def _dropdown_option(label: str, value: str) -> dict:
    return {"label": html.Span(label, style=DROPDOWN_OPTION_LABEL_STYLE), "value": value}


# ── Helper ────────────────────────────────────────────────────────────────────
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


# ── Reusable filter panel ─────────────────────────────────────────────────────
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


# ── Tab layouts ───────────────────────────────────────────────────────────────

# ── Tab 1 — Comparison ────────────────────────────────────────────────────────
tab_comparison = html.Div([
    # Controls
    html.Div([
        html.Div([
            html.Label("Top N clubs / players", className="ctrl-label"),
            dcc.Slider(
                id="cmp-topn", min=5, max=20, step=1, value=10,
                marks={i: {"label": str(i), "style": SLIDER_MARK_STYLE} for i in [5, 10, 15, 20]},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ], style={"flex": "1 1 320px", "paddingTop": "8px"}),

    ], className="ctrl-bar"),

    # Row 1 — Column + Bar
    html.Div([
        _chart_card("cmp-col", "Top Clubs", "blue"),
        _chart_card("cmp-bar", "Elite Players", "green"),
    ], className="chart-row"),

    # Row 2 — Stacked Column + Stacked Bar
    html.Div([
        _chart_card("cmp-stk-col", "Position Depth", "purple"),
        _chart_card("cmp-stk-bar", "Nationality Mix", "blue"),
    ], className="chart-row"),

    # Row 3 — Clustered Column + Clustered Bar (Person 4)
    html.Div([
        _chart_card("cmp-clust-col", "Rating Bands", "green"),
        _chart_card("cmp-clust-bar", "Wage vs Rating", "purple"),
    ], className="chart-row"),
])

# ── Tab 2 — Relationship ──────────────────────────────────────────────────────
tab_relationship = html.Div([
    _filter_panel("rel"),
    html.Div([
        _chart_card("rel-scatter", "Age vs Overall", "blue"),
        _chart_card("rel-bubble", "Potential Market", "green"),
    ], className="chart-row"),
])

# ── Tab 3 — Distribution ──────────────────────────────────────────────────────
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

    # Histogram (full width)
    html.Div([
        _chart_card("dist-hist", "Age Distribution", "blue"),
    ], className="chart-row chart-row-single"),

    # Box + Violin
    html.Div([
        _chart_card("dist-box", "Position Range", "purple"),
        _chart_card("dist-violin", "Rating Density", "green"),
    ], className="chart-row"),
])

# ── Tab 4 — Time Series ───────────────────────────────────────────────────────
tab_timeseries = html.Div([
    _filter_panel("ts"),
    html.Div([
        _chart_card("ts-line", "Age Rating Trend", "blue"),
        _chart_card("ts-area", "Player Distribution", "green"),
    ], className="chart-row"),
])


# ── App + layout ──────────────────────────────────────────────────────────────
app = Dash(__name__)
app.title = "FIFA Data Visualization Dashboard"

app.layout = html.Div([
    html.Div(className="stadium-hologram"),

    # ── Header
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
            _kpi_card("Players", f"{len(DF):,}", "loaded dataset", "green"),
            _kpi_card("Age Window", f"{AGE_MIN}-{AGE_MAX}", "available range", "blue"),
            _kpi_card("Positions", f"{len(POS_GROUPS)}", "player groups", "purple"),
        ], className="kpi-grid"),
    ], className="hero"),

    # ── Tabs
    dcc.Tabs(
        id="main-tabs",
        value="tab-cmp",
        children=[
            dcc.Tab(label="Comparison", value="tab-cmp",
                    children=tab_comparison,
                    selected_style={"borderTop": "3px solid #24c8ff",
                                    "fontWeight": "700", "color": "#07111c"}),
            dcc.Tab(label="Relationship", value="tab-rel",
                    children=tab_relationship,
                    selected_style={"borderTop": "3px solid #2cff8f",
                                    "fontWeight": "700", "color": "#07111c"}),
            dcc.Tab(label="Distribution", value="tab-dist",
                    children=tab_distribution,
                    selected_style={"borderTop": "3px solid #a855f7",
                                    "fontWeight": "700", "color": "#07111c"}),
            dcc.Tab(label="Time Series", value="tab-ts",
                    children=tab_timeseries,
                    selected_style={"borderTop": "3px solid #2cff8f",
                                    "fontWeight": "700", "color": "#07111c"}),
        ],
        className="tabs-shell",
    ),

], style={
    "fontFamily": "Poppins, Montserrat, system-ui, -apple-system, sans-serif",
    "width": "100%",
    "maxWidth": "none",
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
    Input("cmp-topn", "value"),
)
def update_comparison(top_n: int):
    top_n = top_n or 10

    # ── 1. Column — Top N clubs by avg overall ────────────────────────────────
    club_avg = (
        DF.groupby("club_name")["overall"].mean()
        .sort_values(ascending=False).head(top_n).reset_index()
    )
    # Dark → light blues: rank 1 (highest) gets darkest shade
    bar_colors = ["#0f4ea3"] + ["#38a8f4"] * max(len(club_avg) - 1, 0)
    col_fig = go.Figure(go.Bar(
        x=club_avg["club_name"],
        y=club_avg["overall"],
        text=club_avg["overall"].round(1),
        textposition="outside",
        marker_color=bar_colors,
    ))
    col_fig.update_layout(
        title=f"Column Chart — Top {top_n} Clubs by Average Overall Rating",
        xaxis_title="Club", yaxis_title="Avg Overall Rating",
        xaxis_tickangle=-35,
        template=TEMPLATE, height=H,
        plot_bgcolor="white", paper_bgcolor="white",
    )

    # ── 2. Bar — Top N players by overall (vertical, like Person 2's original) ──
    top_players = (
        DF.nlargest(top_n * 3, "overall")
        .drop_duplicates("short_name")
        .head(top_n)[["short_name", "overall", "club_name"]]
    )
    bar_fig = px.bar(
        top_players, x="short_name", y="overall",
        text="overall",
        color="overall", color_continuous_scale="viridis",
        title=f"Top {top_n} Players in FIFA",
        labels={"short_name": "Player", "overall": "Rating"},
        template=TEMPLATE, height=H,
    )
    bar_fig.update_traces(texttemplate="%{text}", textposition="outside")
    bar_fig.update_layout(
        xaxis_tickangle=-45,
        coloraxis_showscale=False,
        xaxis_title="Player", yaxis_title="Rating",
        plot_bgcolor="white", paper_bgcolor="white",
    )

    # ── 3. Stacked Column — player count by position per top N clubs ──────────
    top_clubs = DF["club_name"].value_counts().head(top_n).index.tolist()
    stk_df  = DF[DF["club_name"].isin(top_clubs)]
    stk_grp = stk_df.groupby(["club_name", "position_group"]).size().reset_index(name="count")
    stk_col_fig = px.bar(
        stk_grp, x="club_name", y="count",
        color="position_group", barmode="stack",
        color_discrete_sequence=_BLUES,
        title=f"Stacked Column — Player Count by Position (Top {top_n} Clubs)",
        labels={"club_name": "Club", "count": "Players", "position_group": "Position"},
        template=TEMPLATE, height=H,
    )
    stk_col_fig.update_layout(
        xaxis_tickangle=-35,
        xaxis_title="Club", yaxis_title="Number of Players",
        legend_title="Position",
    )

    # ── 4. Stacked Bar — nationality distribution per top N clubs (horizontal) ─
    top_nat = DF["nationality_name"].value_counts().head(8).index.tolist()
    nat_df  = stk_df.copy()
    nat_df["nationality_label"] = nat_df["nationality_name"].apply(
        lambda x: x if x in top_nat else "Other"
    )
    nat_grp = nat_df.groupby(["club_name", "nationality_label"]).size().reset_index(name="count")
    stk_bar_fig = px.bar(
        nat_grp, x="count", y="club_name",
        color="nationality_label", barmode="stack", orientation="h",
        color_discrete_sequence=_ORANGES,
        title=f"Stacked Bar — Nationality Distribution (Top {top_n} Clubs)",
        labels={"club_name": "Club", "count": "Players", "nationality_label": "Nationality"},
        template=TEMPLATE, height=H,
    )
    stk_bar_fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Number of Players", yaxis_title="Club",
        legend_title="Nationality",
    )

    # ── 5. Clustered Column — rating groups by position ───────────────────────
    bins   = [40, 60, 70, 80, 90, 100]
    labels = ["40–60", "60–70", "70–80", "80–90", "90+"]
    rated  = DF.copy()
    rated["rating_group"] = pd.cut(
        rated["overall"], bins=bins, labels=labels, right=False
    )
    clust_grp = (
        rated.groupby(["position_group", "rating_group"], observed=True)
        .size().reset_index(name="count")
    )
    clust_col_fig = px.bar(
        clust_grp, x="position_group", y="count",
        color="rating_group", barmode="group",
        color_discrete_sequence=_PURPLES,
        title="Clustered Column — Players by Position and Rating Group",
        labels={"position_group": "Position", "count": "Players", "rating_group": "Rating Range"},
        template=TEMPLATE, height=H,
    )
    clust_col_fig.update_layout(
        xaxis_title="Position", yaxis_title="Number of Players",
        legend_title="Rating Range",
    )

    # ── 6. Clustered Bar — Avg Wage vs Avg Rating per club (Person 4) ─────────
    wage_rating = (
        DF[DF["club_name"].isin(top_clubs)]
        .groupby("club_name")
        .agg(avg_wage=("wage_eur", "mean"), avg_overall=("overall", "mean"))
        .reset_index()
    )

    # Normalize both to [0, 100] — sort highest at top (ascending for horizontal)
    wage_rating["wage_score"]    = wage_rating["avg_wage"]    / wage_rating["avg_wage"].max()    * 100
    wage_rating["overall_score"] = wage_rating["avg_overall"] / wage_rating["avg_overall"].max() * 100
    wage_rating = wage_rating.sort_values("wage_score", ascending=True)

    # Highest-wage club gets a different (green) color — rest get blue shades
    highest_club = wage_rating["club_name"].iloc[-1]
    wage_colors    = ["lightgreen" if c == highest_club else "lightblue"  for c in wage_rating["club_name"]]
    overall_colors = ["seagreen"   if c == highest_club else "steelblue"  for c in wage_rating["club_name"]]

    clust_bar_fig = go.Figure()
    clust_bar_fig.add_trace(go.Bar(
        name="Avg Wage",
        y=wage_rating["club_name"],
        x=wage_rating["wage_score"],
        orientation="h",
        marker_color=wage_colors,
        text=wage_rating["avg_wage"].apply(lambda v: f"€{v:,.0f}"),
        textposition="outside",
        textfont=dict(color="black"),
        hovertemplate="<b>%{y}</b><br>Avg Wage: %{text}<extra></extra>",
    ))
    clust_bar_fig.add_trace(go.Bar(
        name="Avg Overall",
        y=wage_rating["club_name"],
        x=wage_rating["overall_score"],
        orientation="h",
        marker_color=overall_colors,
        text=wage_rating["avg_overall"].apply(lambda v: f"{v:.1f}"),
        textposition="outside",
        textfont=dict(color="black"),
        hovertemplate="<b>%{y}</b><br>Avg Overall: %{text}<extra></extra>",
    ))

    clust_bar_fig.update_layout(
        barmode="group",
        title=dict(
            text=f"Avg Wage vs Avg Rating — Top {top_n} Clubs Side-by-Side",
            font=dict(color="black"),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(text="Normalized Score (0 = min, 100 = max)", font=dict(color="black")),
            range=[0, 145],
            showgrid=True, gridcolor="lightgrey",
            showline=True, linecolor="black", mirror=True,
            tickfont=dict(color="black"),
        ),
        yaxis=dict(
            title=dict(text="Club", font=dict(color="black")),
            showline=True, linecolor="black", mirror=True,
            tickfont=dict(color="black"),
        ),
        shapes=[dict(type="rect", xref="paper", yref="paper",
                     x0=0, y0=0, x1=1, y1=1,
                     line=dict(color="black", width=2))],
        legend=dict(
            x=0.99, y=0.99, xanchor="right", yanchor="top",
            bgcolor="white", bordercolor="black", borderwidth=1,
            font=dict(color="black"),
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        template=TEMPLATE,
        height=H,
    )

    return col_fig, bar_fig, stk_col_fig, stk_bar_fig, clust_col_fig, clust_bar_fig


# ── Tab 2 — Relationship ──────────────────────────────────────────────────────
@callback(
    Output("rel-fval",     "options"),
    Output("rel-fval",     "value"),
    Input("rel-ftype",     "value"),
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

    # Scatter — Age vs Overall Rating  (Person 5 original)
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
        text="label",
        hover_name="short_name",
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
                     x0=0, y0=0, x1=1, y1=1,
                     line=dict(color="black", width=2))],
        plot_bgcolor="white", paper_bgcolor="white",
    )

    # Bubble — Age vs Potential  (Person 5 original)
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
            text="label",
            hover_name="short_name",
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
                         x0=0, y0=0, x1=1, y1=1,
                         line=dict(color="black", width=2))],
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

    mlabel = "Overall Rating" if metric == "overall" else "Wage (€)"
    pos_label = position if position != "__ALL__" else "All Positions"

    # Histogram — Age distribution
    hist_fig = px.histogram(
        sub, x="age", nbins=20,
        title=f"Histogram — Age Distribution  ({pos_label})",
        labels={"age": "Age", "count": "Number of Players"},
        color_discrete_sequence=["#1f77b4"],
        template=TEMPLATE, height=H,
    )
    hist_fig.update_layout(
        bargap=0.05,
        xaxis_title="Age", yaxis_title="Number of Players",
    )

    # Box — uses main_position (raw: ST, CB, GK…) like Person 2's original
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
        showlegend=False,
        xaxis_tickangle=-45,
        xaxis_title="Position", yaxis_title="Wage (EUR)",
        yaxis_range=[0, 250000],
        plot_bgcolor="white", paper_bgcolor="white",
    )

    # Violin — keeps position_group + dynamic metric
    violin_fig = px.violin(
        box_sub, x="position_group", y=metric,
        color="position_group", box=True,
        color_discrete_sequence=_TEALS,
        title=f"Violin — {mlabel} Distribution by Position Group",
        labels={"position_group": "Position", metric: mlabel},
        template=TEMPLATE, height=H,
    )
    violin_fig.update_layout(
        showlegend=False,
        xaxis_title="Position", yaxis_title=mlabel,
    )

    return hist_fig, box_fig, violin_fig


# ── Tab 4 — Time Series ───────────────────────────────────────────────────────
@callback(
    Output("ts-fval",  "options"),
    Output("ts-fval",  "value"),
    Input("ts-ftype",  "value"),
)
def ts_options(ftype):
    all_opt = _dropdown_option("All", "__ALL__")
    if ftype == "Club":
        vals = sorted(DF["club_name"].dropna().astype(str).unique())
    else:
        vals = sorted(DF["nationality_name"].dropna().astype(str).unique())
    return [all_opt] + [_dropdown_option(v, v) for v in vals], "__ALL__"


@callback(
    Output("ts-line", "figure"),
    Output("ts-area", "figure"),
    Input("ts-ftype", "value"),
    Input("ts-fval",  "value"),
    Input("ts-age",   "value"),
)
def update_timeseries(ftype, fval, age_range):
    age_lo, age_hi = (AGE_MIN, AGE_MAX) if not age_range else (int(age_range[0]), int(age_range[1]))
    sub = _filter_df(DF, ftype, fval, age_lo, age_hi)

    if sub.empty:
        e = _empty("No data — widen the age range or select All.")
        return e, e

    trend = sub.groupby("age", as_index=False)["overall"].mean().sort_values("age")
    line_fig = px.line(
        trend, x="age", y="overall", markers=True,
        title="Line — Mean Overall Rating by Age",
        labels={"age": "Age", "overall": "Mean Overall Rating"},
        template=TEMPLATE, height=H,
    )
    line_fig.update_traces(line=dict(width=3))
    line_fig.update_layout(
        hovermode="x unified",
        xaxis_title="Age", yaxis_title="Mean Overall Rating",
    )

    counts = sub.groupby("age").size().reset_index(name="players").sort_values("age")
    area_fig = px.area(
        counts, x="age", y="players",
        title="Area — Player Count Distribution by Age",
        labels={"age": "Age", "players": "Player Count"},
        template=TEMPLATE, height=H,
    )
    area_fig.update_traces(line_color="#636efa", fillcolor="rgba(99,110,250,0.35)")
    area_fig.update_layout(
        xaxis_title="Age", yaxis_title="Player Count",
    )

    return line_fig, area_fig


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
