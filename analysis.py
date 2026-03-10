import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.express as px


# -------------------------
# Data Prep (matches ipynb)
# -------------------------

def load_master(file_path="Data/OMT_MDCR_RY25_P04_V10_Y23_GEO.csv"):
    """Loads raw CSV exactly like the notebook Cell 3."""
    return pd.read_csv(file_path)


def build_filtered_df(df_master: pd.DataFrame) -> pd.DataFrame:
    """Renames columns like notebook Cell 7."""
    filtered_df = df_master.rename(columns={
        "Year": "year",
        "Prscrbr_Geo_Lvl": "geo_level",
        "Prscrbr_Geo_Desc": "geography",
        "Tot_Prscrbrs": "total_prescribers",
        "Tot_Opioid_Prscrbrs": "opioid_prescribers",
        "Tot_Opioid_Clms": "opioid_claims",
        "Tot_Clms": "total_claims",
        "Opioid_Prscrbng_Rate": "opioid_rate",
        "Opioid_Prscrbng_Rate_1Y_Chg": "opioid_rate_change_1y",
        "Opioid_Prscrbng_Rate_5Y_Chg": "opioid_rate_change_5y"
    })
    return filtered_df


def build_df_clean(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Selects final columns + dropna like notebook Cells 9–10."""
    columns_to_keep = [
        "year",
        "geo_level",
        "geography",
        "total_prescribers",
        "opioid_prescribers",
        "opioid_claims",
        "total_claims",
        "opioid_rate"
    ]
    df_clean = filtered_df[columns_to_keep].copy()
    df_clean = df_clean.dropna()
    return df_clean


def build_df_state(df_clean: pd.DataFrame) -> pd.DataFrame:
    """State filter + derived cols like notebook Cells 12, 14, 17."""
    df_state = df_clean.loc[df_clean["geo_level"] == "State"].copy()

    # Cell 14: opioid_rate_category
    bins = [0, 3, 5, 7, 10]
    labels = ["Low", "Moderate", "High", "Very High"]
    df_state["opioid_rate_category"] = pd.cut(
        df_state["opioid_rate"], bins=bins, labels=labels
    )

    # Cell 17: opioid_prescriber_ratio + prescriber_category
    df_state["opioid_prescriber_ratio"] = (
        df_state["opioid_prescribers"] / df_state["total_prescribers"]
    )

    bins = [0, 0.25, 0.5, 0.75, 1]
    labels = ["Low Participation", "Moderate Participation",
              "High Participation", "Very High Participation"]
    df_state["prescriber_category"] = pd.cut(
        df_state["opioid_prescriber_ratio"], bins=bins, labels=labels
    )

    return df_state


def ensure_charts_dir(charts_dir="Data/Charts"):
    os.makedirs(charts_dir, exist_ok=True)
    return charts_dir


# -------------------------
# Charts (matches ipynb)
# Each function returns fig
# -------------------------

def chart_states_by_opioid_category_pie(df_state: pd.DataFrame, save_pdf=False):
    """Notebook Cell 15 pie chart."""
    df_state_counts = df_state["opioid_rate_category"].value_counts(ascending=True)

    color_map = {
        "Low": "#55A868",
        "Moderate": "#FFB90F",
        "High": "#DD8452",
        "Very High": "#EE3B3B"
    }
    colors = [color_map[label] for label in df_state_counts.index]

    fig, ax = plt.subplots(figsize=(8, 8))
    df_state_counts.plot(
        kind="pie",
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        explode=[0.03] * len(df_state_counts),
        wedgeprops={"edgecolor": "black"},
        shadow=True,
        ax=ax
    )

    ax.set_title("States by Opioid Prescribing Category", fontsize=16, weight="bold")
    ax.set_ylabel("")

    if save_pdf:
        charts_dir = ensure_charts_dir()
        fig.savefig(os.path.join(charts_dir, "States_by_Opioid_Perscribing_Category.pdf"))

    return fig


def chart_prescriber_participation(df_state: pd.DataFrame, save_pdf=False):
    """Notebook Cell 17 bar chart with labels."""
    df_state_presc_category = (
        df_state["prescriber_category"].value_counts().sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax = df_state_presc_category.plot(
        kind="bar",
        color=["#C44E52", "#55A868", "#4C72B0", "#9400D3"],
        ax=ax
    )

    ax.set_title("Prescriber Participation", fontsize=14, weight="bold")
    ax.set_xlabel("Participation Category", fontsize=12, fontstyle="italic")
    ax.set_ylabel("Number of Records", fontsize=12, fontstyle="italic")

    for bar in ax.patches:
        ax.annotate(
            f"{int(bar.get_height())}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.xticks(rotation=20)
    plt.tight_layout()

    if save_pdf:
        charts_dir = ensure_charts_dir()
        fig.savefig(os.path.join(charts_dir, "Perscriber_Participation.pdf"))

    return fig


def compute_top_bottom_claims_and_rates(df_state: pd.DataFrame):
    """Notebook Cell 18 helper: computes top/bottom by claims and by rate."""
    # claims totals
    state_claims = df_state.groupby("geography")["opioid_claims"].sum()
    top_states_claims = state_claims.sort_values(ascending=False).head(10)
    bottom_states_claims = state_claims.sort_values(ascending=True).head(10)

    # mean opioid rate
    state_rates = df_state.groupby("geography")["opioid_rate"].mean()
    top_states_rates = state_rates.sort_values(ascending=False).head(10)
    bottom_states_rates = state_rates.sort_values(ascending=True).head(10)

    return top_states_claims, bottom_states_claims, top_states_rates, bottom_states_rates


def _bar_with_labels(series, title, ylabel, save_name=None, save_pdf=False):
    """Shared bar styling used in Cells 19–22."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax = series.plot(kind="bar", color="#4C72B0", edgecolor="black", ax=ax)
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("State", fontsize=12, fontstyle="italic")
    ax.set_ylabel(ylabel, fontsize=12, fontstyle="italic")

    # y formatting (your notebook uses commas)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=30)

    # add labels
    for bar in ax.patches:
        h = bar.get_height()
        ax.annotate(
            f"{int(h):,}" if abs(h) >= 1 else f"{h:.2f}",
            (bar.get_x() + bar.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()

    if save_pdf and save_name:
        charts_dir = ensure_charts_dir()
        fig.savefig(os.path.join(charts_dir, save_name))

    return fig


def chart_top_10_states_by_opioid_claims(df_state: pd.DataFrame, save_pdf=False):
    """Notebook Cell 19."""
    top_claims, _, _, _ = compute_top_bottom_claims_and_rates(df_state)
    return _bar_with_labels(
        top_claims,
        title="Top 10 States by Opioid Claims",
        ylabel="Total Opioid Claims",
        save_name="Top_10_States_by_Opioid_Claims.pdf",
        save_pdf=save_pdf
    )


def chart_bottom_10_states_by_opioid_claims(df_state: pd.DataFrame, save_pdf=False):
    """Notebook Cell 20."""
    _, bottom_claims, _, _ = compute_top_bottom_claims_and_rates(df_state)
    return _bar_with_labels(
        bottom_claims,
        title="Bottom 10 States by Opioid Claims",
        ylabel="Total Opioid Claims",
        save_name="Bottom_10_States_by_Opioid_Claims.pdf",
        save_pdf=save_pdf
    )


def chart_top_10_states_by_opioid_rate(df_state: pd.DataFrame, save_pdf=False):
    """Notebook Cell 21 (named 'claims rates' in your title, but it’s opioid_rate mean)."""
    _, _, top_rates, _ = compute_top_bottom_claims_and_rates(df_state)

    # Keep your exact title text for consistency with notebook
    fig = _bar_with_labels(
        top_rates,
        title="Top 10 States by Opioid Claims Rates",
        ylabel="Percent of Claims that are Opioids",
        save_name="Top_10_States_by_Opioid_Claims_Rates.pdf",
        save_pdf=save_pdf
    )

    # Adjust bar label formatting (your notebook shows 1 decimal)
    ax = fig.axes[0]
    for txt in ax.texts:
        pass  # keep existing
    return fig


def chart_bottom_10_states_by_opioid_rate(df_state: pd.DataFrame, save_pdf=False):
    """Notebook Cell 22 (named 'claims rates' in your title, but it’s opioid_rate mean)."""
    _, _, _, bottom_rates = compute_top_bottom_claims_and_rates(df_state)

    fig = _bar_with_labels(
        bottom_rates,
        title="Bottom 10 States by Opioid Claims Rates",
        ylabel="Percent of Claims that are Opioids",
        save_name="Bottom_10_States_by_Opioid_Claims_Rates.pdf",
        save_pdf=save_pdf
    )

    return fig


def chart_national_rural_vs_urban_trend(df_master: pd.DataFrame, save_pdf=False):
    """Notebook Cell 23 line chart: National Rural vs Urban trend."""
    df_nat_ru = df_master[
        (df_master["Prscrbr_Geo_Lvl"] == "National") &
        (df_master["Breakout"].isin(["Rural", "Urban"]))
    ].dropna(subset=["Opioid_Prscrbng_Rate"]).sort_values("Year")

    rural_data = df_nat_ru[df_nat_ru["Breakout"] == "Rural"]
    urban_data = df_nat_ru[df_nat_ru["Breakout"] == "Urban"]

    fig, ax = plt.subplots(figsize=(13, 7))

    ax.plot(
        rural_data["Year"], rural_data["Opioid_Prscrbng_Rate"],
        color="#2E86AB", linewidth=3, marker="o", markersize=8,
        label="Rural", zorder=4, markerfacecolor="white", markeredgewidth=2.5
    )
    ax.plot(
        urban_data["Year"], urban_data["Opioid_Prscrbng_Rate"],
        color="#E8553D", linewidth=3, marker="s", markersize=8,
        label="Urban", zorder=4, markerfacecolor="white", markeredgewidth=2.5
    )

    ax.fill_between(
        rural_data["Year"].values,
        rural_data["Opioid_Prscrbng_Rate"].values,
        urban_data["Opioid_Prscrbng_Rate"].values,
        alpha=0.12, color="#8E44AD", zorder=1, label="Rural-Urban Gap"
    )

    for _, row in rural_data.iterrows():
        ax.text(row["Year"], row["Opioid_Prscrbng_Rate"] + 0.08,
                f'{row["Opioid_Prscrbng_Rate"]:.2f}',
                ha="center", va="bottom", fontsize=8.5,
                color="#2E86AB", fontweight="bold")

    for _, row in urban_data.iterrows():
        ax.text(row["Year"], row["Opioid_Prscrbng_Rate"] - 0.15,
                f'{row["Opioid_Prscrbng_Rate"]:.2f}',
                ha="center", va="top", fontsize=8.5,
                color="#E8553D", fontweight="bold")

    gap_2013 = rural_data[rural_data["Year"] == 2013]["Opioid_Prscrbng_Rate"].values[0] - \
               urban_data[urban_data["Year"] == 2013]["Opioid_Prscrbng_Rate"].values[0]
    gap_2023 = rural_data[rural_data["Year"] == 2023]["Opioid_Prscrbng_Rate"].values[0] - \
               urban_data[urban_data["Year"] == 2023]["Opioid_Prscrbng_Rate"].values[0]

    ax.set_title(
        "National Opioid Prescribing Rate: Rural vs. Urban (2013-2023)",
        fontsize=16, fontweight="bold", pad=20, color="#2C3E50"
    )
    ax.set_xlabel("Year", fontsize=13, labelpad=10, color="#555")
    ax.set_ylabel("Opioid Prescribing Rate (%)", fontsize=13, labelpad=10, color="#555")
    ax.set_xticks(range(2013, 2024))
    ax.set_xticklabels(range(2013, 2024), rotation=45, fontsize=10)

    ax.legend(fontsize=11, loc="upper right", framealpha=0.95,
              edgecolor="#ccc", fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#ccc")
    ax.spines["bottom"].set_color("#ccc")

    ax.annotate(
        f"Rural consistently exceeds Urban.\n"
        f"Gap in 2013: {gap_2013:.2f}%  →  2023: {gap_2023:.2f}%\n"
        f"The gap has narrowed slightly over the decade.",
        xy=(0.02, 0.35), xycoords="axes fraction",
        fontsize=10, color="#666", style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#ddd", alpha=0.9)
    )

    plt.tight_layout()

    if save_pdf:
        charts_dir = ensure_charts_dir()
        fig.savefig(os.path.join(charts_dir, "Rural_vs_Urban_Opioid_Rate.pdf"))

    return fig

# For MAP
US_STATE_ABBREV = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
}


def add_state_abbrev_from_name(df_state: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 2-letter state abbreviations using the full state name in `geography`.
    Filters out rows that are not mappable (e.g., 'United States', territories).
    """
    df = df_state.copy()

    df["state_code"] = df["geography"].astype(str).str.strip().map(US_STATE_ABBREV)

    # Keep only rows that successfully mapped to a 2-letter code
    df = df[df["state_code"].notna()].copy()

    return df


def map_prescriber_ratio_choropleth(df_state_with_codes, year=None):
    df = df_state_with_codes.copy()
    if year is None:
        year = int(df["year"].max())

    df = df.loc[df["year"] == year].copy()

    df = df.groupby(["state_code", "geography"], as_index=False).agg(
        opioid_prescriber_ratio=("opioid_prescriber_ratio", "mean"),
        opioid_rate=("opioid_rate", "mean"),
        opioid_claims=("opioid_claims", "sum"),
    )

    # Create 6 quantile buckets (equal count per bucket)
    df["ratio_bin"] = pd.qcut(df["opioid_prescriber_ratio"], q=6, duplicates="drop")

    fig = px.choropleth(
        df,
        locations="state_code",
        locationmode="USA-states",
        color="ratio_bin",                 # <-- bins drive color
        scope="usa",
        color_discrete_sequence=px.colors.sequential.Blues,  # or Viridis, Plasma, etc.
        hover_name="geography",
        hover_data={
            "state_code": True,
            "opioid_prescriber_ratio": ":.2%",
            "opioid_rate": ":.2f",
            "opioid_claims": ":,",
        },
        title=f"US Map: Prescriber Participation Intensity ({year})",
    )

    fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))
    return fig
