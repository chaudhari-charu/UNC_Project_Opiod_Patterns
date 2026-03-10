import streamlit as st
import analysis
import plotly.express as px 

st.set_page_config(page_title="Opioid Dashboard", layout="wide")
st.title("US Opioid Prescribing Dashboard")

# ---- Load + prep data (same pipeline as ipynb) ----
df_master = analysis.load_master()
filtered_df = analysis.build_filtered_df(df_master)
df_clean = analysis.build_df_clean(filtered_df)
df_state = analysis.build_df_state(df_clean)

# ---- Build charts (all from functions) ----
charts = [

    # ("Prescriber Participation (Bar)", analysis.chart_prescriber_participation(df_state)),
    ("Top 10 States by Opioid Claims", analysis.chart_top_10_states_by_opioid_claims(df_state)),
    ("Top 10 States by Opioid Rate", analysis.chart_top_10_states_by_opioid_rate(df_state)),
    ("Bottom 10 States by Opioid Claims", analysis.chart_bottom_10_states_by_opioid_claims(df_state)),
    ("Bottom 10 States by Opioid Rate", analysis.chart_bottom_10_states_by_opioid_rate(df_state)),
    ("National Rural vs Urban Trend (2013–2023)", analysis.chart_national_rural_vs_urban_trend(df_master)),
    # ("States by Opioid Prescribing Category (Pie)", analysis.chart_states_by_opioid_category_pie(df_state))
]

# ---- Render 2 per row ----
for i in range(0, len(charts), 2):
    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(charts[i][1], clear_figure=True)

    with col2:
        if i + 1 < len(charts):
            st.pyplot(charts[i + 1][1], clear_figure=True)
        else:
            st.empty()

st.divider()

# ---- Map placeholder ----
# st.subheader("Map (Coming Next)")
# st.info("We will add a US state choropleth map here next.")
df_state_codes = analysis.add_state_abbrev_from_name(df_state)

map_fig = analysis.map_prescriber_ratio_choropleth(
    df_state_codes,
    year=int(df_state_codes["year"].max())  # or 2023
)

st.subheader("Map: Prescriber Participation Intensity")
st.plotly_chart(map_fig, use_container_width=True)