import streamlit as st

st.set_page_config(
    page_title="South Barnegat Bay Onshore Wind Model Prediction",
    page_icon="👋",
)
PRIMARY_BG = "#486e8d"
PANEL_BG = "#4d6070"
ACCENT = "#FFC94A"
TEXT_LIGHT = "#F5F5F5"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {PRIMARY_BG};
        color: {TEXT_LIGHT};
    }}
    .stSidebar {{
        background-color: {PANEL_BG} !important;
    }}
    <style>
    [data-testid="stDecoration"] {{
        background-color: {PANEL_BG};
        background-image: none;
    }}
    .stButton>button {{
        background-color: {ACCENT};
        color: #333333;
        border-radius: 8px;
        border: none;
    }}
    .stButton>button:hover {{
        opacity: 0.9;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.write("# Welcome to South Barnegat Bay Onshore Wind Model Prediction!")

st.sidebar.success("Select a Prediction Model.")

st.markdown(
    """
    South Barnegat Bay frequently experiences sudden, powerful onshore wind gusts during summer, often connected to local upwelling events that bring cold water from deeper parts of the bay or nearby ocean to the surface. These gusts can rapidly create steep, choppy waves that are dangerous for kayaks, canoes, and personal watercraft.
    Traditional weather models and public forecasts operate at regional scales and often miss small‑scale, bay‑specific changes, leaving local boaters with little warning.
    This project explores whether a local LSTM model trained on detailed hourly observations can provide more accurate, location‑specific, one‑hour‑ahead warnings.
    
    **<== Select a model from the sidebar** to predict wind speed and direction for the next hour!
    ### For Southern Barnegat Bay, you can get the current temperatures from these locations:

    - [Stafford Weather Station](https://www.wunderground.com/dashboard/pws/KNJMANAH7)  
       - [Back-up Station](https://www.wunderground.com/dashboard/pws/KNJMANAH87)
    - [Ship Bottom NJDEP MB_01 Buoy](https://njdep.rutgers.edu/continuous/station/NJBuoy759/)  
       - Or use the water temperature and salinity of nearby bay water.
    - [Surf City Yacht Club Weather Station](https://www.wunderground.com/dashboard/pws/KNJSURFC12)
       -[Back-up Station](https://www.wunderground.com/dashboard/pws/KNJSURFC7)
    - [LBI Ocean NDBC 44091 Buoy](https://www.ndbc.noaa.gov/station_page.php?station=44091)
    
    
    ### Want to learn more?
    - Check out the github documentation[github.com/Krupamc/Research-2026-LSTM](https://github.com/Krupamc/Research-2026-LSTM)
    
"""
)