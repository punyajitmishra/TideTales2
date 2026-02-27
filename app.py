import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import anthropic
import requests
import time
import random

# ============================================================
# 1. CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Tide Tales",
    layout="wide",
    page_icon="🌊",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. SESSION STATE INITIALIZATION
# ============================================================
defaults = {
    'user_location': "Bhubaneswar, India",
    'data_mapped': None,
    'sci_metadata': None,
    'last_story_output': "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def get_server_location():
    """Attempts IP-based location detection."""
    try:
        res = requests.get('https://ipapi.co/json/', timeout=5).json()
        city = res.get('city', 'Bhubaneswar')
        country = res.get('country_name', 'India')
        return f"{city}, {country}"
    except:
        return "Bhubaneswar, India"


def get_science_mood(label):
    """Returns visual and metaphorical theme based on science type."""
    lbl = str(label).lower()
    if any(k in lbl for k in ['temp', 'warm', 'anomaly', 'j-d']):
        return {"element": "Fire", "metaphor": "the earth's fever", "color": "#00D4FF"}
    if any(k in lbl for k in ['air', 'aqi', 'pm2', 'pm10', 'smog']):
        return {"element": "Breath", "metaphor": "the choking haze", "color": "#FF5733"}
    if any(k in lbl for k in ['sea', 'level', 'tide', 'ocean', 'water']):
        return {"element": "Water", "metaphor": "the hungry ocean", "color": "#2ECC71"}
    if any(k in lbl for k in ['co2', 'carbon', 'ppm']):
        return {"element": "Weight", "metaphor": "the heavy sky", "color": "#8E44AD"}
    return {"element": "Change", "metaphor": "the changing pulse", "color": "#FFFFFF"}


@st.cache_data(show_spinner=False)
def fetch_nasa_gistemp():
    """Fetches live NASA GISTEMP global temperature data."""
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    try:
        df = pd.read_csv(url, skiprows=1, na_values="***")
        if 'Year' in df.columns and 'J-D' in df.columns:
            df_clean = df[['Year', 'J-D']].copy()
            df_clean.columns = ['year', 'val']
            df_clean = df_clean.apply(pd.to_numeric, errors='coerce').dropna()
            return df_clean, {"YearCol": "Year", "DataCol": "J-D", "Unit": "°C Anomaly", "Label": "Global Warming"}
    except Exception as e:
        st.warning(f"NASA fetch failed ({e}). Using fallback data.")
    # Fallback synthetic data
    years = list(range(1880, 2024))
    vals = np.linspace(-0.3, 1.2, len(years)) + np.random.normal(0, 0.05, len(years))
    return pd.DataFrame({'year': years, 'val': vals}), {"YearCol": "Year", "DataCol": "J-D", "Unit": "°C Anomaly", "Label": "Global Warming"}


def ai_interpret_data(df, api_key):
    """Uses Claude to sniff CSV column structure and science type."""
    client = anthropic.Anthropic(api_key=api_key)
    sample = df.head(10).to_string()
    headers = df.columns.tolist()

    prompt = f"""
    Analyze this climate dataset and identify its structure.
    HEADERS: {headers}
    SAMPLE:
    {sample}

    IDENTIFY and return ONLY in this exact format:
    YearCol: [column name]
    DataCol: [column name]
    Unit: [unit of measurement]
    Label: [science label, e.g. Air Quality, Sea Level Rise, Global Warming]
    """
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    mapping = {}
    for line in response.content[0].text.strip().split('\n'):
        if ':' in line:
            key, val = line.split(':', 1)
            mapping[key.strip()] = val.strip()
    return mapping


def build_demo_narrative(loc, sci, selected_range, net_shift, slope, val_end, trough):
    """Procedural narrative engine for demo/offline mode."""
    intensity = (
        "a frantic gallop" if slope > 0.015
        else "a steady, relentless climb" if slope > 0.005
        else "a subtle, whispering shift"
    )
    impact = (
        "the world has broken its ancient promises"
        if abs(net_shift) > 1.0
        else "the balance is beginning to fray at the edges"
    )

    ch1_opts = [
        f"In the ancient memory of **{loc}**, the wind once spoke a language of predictable seasons. But since **{selected_range[0]}**, a new dialect has emerged—one written in the language of {sci['metaphor']}.",
        f"The soil of **{loc}** has its own way of keeping time. Long before we had records starting in **{selected_range[0]}**, the ancestors knew the rhythm of the {sci['element']}. Now, that rhythm has faltered."
    ]
    ch2_opts = [
        f"Science confirms what our hearts already suspected. This is no random flicker. Our trendline moves at **{intensity}**—a rate of **{round(slope, 4)} units per year**.",
        f"The math does not lie, even when it is hard to hear. Moving at **{round(slope, 4)} per year**, the {sci['element']} is undergoing **{intensity}**."
    ]
    ch3_opts = [
        f"There is a legend in **{loc}** about a mirror that reflects the health of the earth. Today, that mirror is clouded. The trough of **{round(trough, 2)}** is a ghost—a remnant of a more stable past.",
        f"The measurement stands today at **{round(val_end, 2)}**, far from the stability of the past. The trough of **{round(trough, 2)}** is a milestone we are leaving behind."
    ]

    chapters = [
        f"### Chapter 1: The Omens\n{random.choice(ch1_opts)} The data reveals a shift of **{round(net_shift, 2)}**, but to the people here, it is {impact}.",
        f"### Chapter 2: The Quickening\n{random.choice(ch2_opts)} This is no longer a fluctuation — it is a transformation of our physical reality, lived by every soul in {loc}.",
        f"### Chapter 3: The Ghost in the Mirror\n{random.choice(ch3_opts)} We realize the balance has shifted. The space between the data and our lives is where the fear lives — and where hope must grow.",
        f"### Chapter 4: The Convergence\nAs we stand at the end of this record in **{selected_range[1]}**, the narrative of {sci['metaphor']} is an epic still being written. In **{loc}**, the convergence of scientific truth and local song is our only map home."
    ]
    return "\n\n".join(chapters)


# ============================================================
# 4. SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🌊 Tide Tales")
    st.divider()

    api_key = st.text_input(
        "Anthropic API Key", type="password", key="api_key",
        help="Get your key from console.anthropic.com. Required for AI narrative and CSV sniffing."
    )

    st.markdown("**📍 Location Context**")
    st.session_state['user_location'] = st.text_input(
        "Your Location",
        value=st.session_state['user_location'],
        key="loc_input",
        help="Anchors the story in local folklore (e.g., Bhubaneswar, New Delhi, Sundarbans)."
    )
    col_loc1, col_loc2 = st.columns(2)
    if col_loc1.button("🔄 Auto-detect", key="loc_retry"):
        st.session_state['user_location'] = get_server_location()
        st.rerun()

    st.divider()

    # Data source
    source_choice = st.radio(
        "Data Source",
        ["🛰️ NASA GISTEMP (Auto)", "📁 Upload My Own CSV"],
        key="side_src"
    )

    user_file = None
    if source_choice == "📁 Upload My Own CSV":
        user_file = st.file_uploader(
            "Upload CSV", type="csv", key="side_uploader",
            help="Upload any climate data CSV. AI will auto-detect columns."
        )

    st.divider()

    # Narrative style controls
    st.subheader("🎨 Narrative Style")
    narr_tone = st.select_slider(
        "Tone",
        options=["Scientific", "Balanced", "Mythic"],
        value="Balanced",
        key="narr_tone",
        help="Scientific = data-focused. Mythic = folklore-led. Balanced = weaves both."
    )
    narr_length = st.radio(
        "Story Length",
        ["Short (~600 words)", "Epic (~1500 words)"],
        index=1,
        key="narr_length"
    )

    st.divider()
    demo_mode = st.toggle(
        "Demo Mode (no API key needed)",
        value=not bool(api_key),
        key="side_demo"
    )


# ============================================================
# 5. DATA LOADING
# ============================================================
if source_choice == "🛰️ NASA GISTEMP (Auto)":
    with st.spinner("Fetching NASA data..."):
        nasa_df, nasa_meta = fetch_nasa_gistemp()
    st.session_state['data_mapped'] = nasa_df
    st.session_state['sci_metadata'] = nasa_meta

elif source_choice == "📁 Upload My Own CSV" and user_file:
    try:
        raw_df = pd.read_csv(user_file, na_values="***")
        st.info(f"File loaded: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns.")

        if st.button("🔍 AI: Interpret Data Structure", key="sniff_btn"):
            if not api_key:
                st.error("API Key required to use the AI column sniffer.")
            else:
                with st.spinner("AI is sniffing the columns..."):
                    try:
                        mapping = ai_interpret_data(raw_df, api_key)
                        y_col, d_col = mapping['YearCol'], mapping['DataCol']
                        clean_df = raw_df[[y_col, d_col]].copy()
                        clean_df.columns = ['year', 'val']
                        clean_df = clean_df.apply(pd.to_numeric, errors='coerce').dropna()
                        st.session_state['data_mapped'] = clean_df
                        st.session_state['sci_metadata'] = mapping
                        st.success(f"✅ Identified as **{mapping['Label']}** ({mapping['Unit']})")
                    except Exception as e:
                        st.error(f"Column mapping failed: {e}")
    except Exception as e:
        st.error(f"File load error: {e}")


# ============================================================
# 6. MAIN UI
# ============================================================
st.title("🌊 Tide Tales")
st.caption("Where scientific data becomes living story.")

# Orientation Center
col_how, col_gal = st.columns(2)
with col_how:
    with st.expander("❓ How to use Tide Tales"):
        st.markdown("""
        1. **Set Location** — Confirm your city in the sidebar for localized folklore.
        2. **Choose Data** — Use NASA for instant results, or upload your own climate CSV.
        3. **Refine** — Drag the timeline slider to the period you want to explore.
        4. **Style** — Choose Tone (Scientific → Mythic) and Story Length in the sidebar.
        5. **Weave** — Hit **Weave Narrative** to generate your dual-language epic.
        """)
with col_gal:
    with st.expander("📚 Example Use Cases"):
        st.caption("• **Global Warming** — 1.2°C rise over 140 years (NASA GISTEMP)")
        st.caption("• **Sea Level Rise** — 4mm/year in the Sundarbans delta")
        st.caption("• **Delhi Smog** — AQI 450+ winter haze vs. the Smog-Demon's Shroud")
        st.caption("• **CO₂ Rise** — Atmospheric carbon since industrialization")


# ============================================================
# 7. DASHBOARD & EVIDENCE PANEL
# ============================================================
if st.session_state['data_mapped'] is not None:
    data = st.session_state['data_mapped'].copy()
    meta = st.session_state['sci_metadata']
    mood = get_science_mood(meta.get('Label', ''))

    data = data.apply(pd.to_numeric, errors='coerce').dropna()

    min_y, max_y = int(data['year'].min()), int(data['year'].max())
    if min_y >= max_y:
        max_y = min_y + 1

    selected_range = st.slider(
        "📅 Select Analysis Timeframe",
        min_y, max_y, (min_y, max_y),
        key="main_slider"
    )

    f_df = data[(data['year'] >= selected_range[0]) & (data['year'] <= selected_range[1])]

    if f_df.empty or len(f_df) < 2:
        st.warning("Not enough data in selected range. Adjust the slider.")
        st.stop()

    # Math (Fact Pack)
    slope, intercept = np.polyfit(f_df['year'].values, f_df['val'].values, 1)
    net_shift = f_df['val'].iloc[-1] - f_df['val'].iloc[0]
    peak = f_df['val'].max()
    trough = f_df['val'].min()
    val_end = f_df['val'].iloc[-1]

    # Evidence Panel
    st.header(f"📊 Evidence: {meta.get('Label', 'Climate Data')}")
    st.write(f"Observing **{meta.get('Label', 'change')}** as experienced in **{st.session_state['user_location']}**.")

    fig = px.line(
        f_df, x='year', y='val',
        template="plotly_dark",
        labels={'year': 'Year', 'val': meta.get('Unit', 'Value')},
        title=f"{mood['metaphor'].capitalize()} — {st.session_state['user_location']}"
    )
    fig.add_scatter(
        x=f_df['year'],
        y=slope * f_df['year'] + intercept,
        name="Trend Line",
        line=dict(color='red', dash='dot', width=2)
    )
    fig.update_traces(selector=dict(mode='lines'), line_color=mood['color'], line_width=2.5)
    fig.update_layout(margin=dict(t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Fact Pack
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Shift", f"{round(net_shift, 2)} {meta.get('Unit', '')}")
    c2.metric("Trend Rate", f"{round(slope, 4)} / yr")
    c3.metric("Highest Peak", f"{round(peak, 2)}")
    c4.metric("Lowest Trough", f"{round(trough, 2)}")

    # ============================================================
    # 8. NARRATIVE ENGINE
    # ============================================================
    st.divider()
    st.header(f"📖 The Tale of {st.session_state['user_location']}")

    if st.button("✨ Weave Narrative", key="weave_btn"):
        loc = st.session_state['user_location']

        if demo_mode or not api_key:
            st.info("Demo Mode: Generating procedural narrative (no API key needed)...")
            story = build_demo_narrative(loc, mood, selected_range, net_shift, slope, val_end, trough)
            for chapter in story.split("\n\n"):
                st.markdown(chapter)
                time.sleep(0.5)
            st.session_state['last_story_output'] = story
            st.balloons()

        else:
            st.info("AI Mode: Engaging Claude for a unique narrative epic...")
            prompt = f"""
            Identify as a Cultural Data Sentinel and a master literary novelist.
            Write a **{narr_length}** immersive story in a **{narr_tone}** tone.

            LOCATION: {loc}
            SCIENCE: {meta.get('Label', 'Climate change')} — the element of {mood['element']}.
            DATA FACTS:
            - Timeframe: {selected_range[0]} to {selected_range[1]}
            - Net shift: {round(net_shift, 2)} {meta.get('Unit', '')}
            - Trend rate: {round(slope, 4)} units/year
            - Peak value: {round(peak, 2)}
            - Trough value: {round(trough, 2)}

            TONE INSTRUCTIONS:
            - If Scientific: Focus on precise physical observation of change. Let numbers breathe.
            - If Mythic: Let folklore spirits and metaphors lead. Data is subtext.
            - If Balanced: Weave metrics and legends into a single tapestry.

            NARRATIVE PHYSICS:
            1. FULL CREATIVE FREEDOM — no fixed chapter scaffold.
            2. The scientific data is the inescapable atmosphere. Integrate numbers naturally.
            3. Use the specific folklore, mythology, and local metaphors of {loc}.
            4. The trend velocity sets the narrative pace: high slope = frantic, low = creeping dread.
            5. Treat {mood['metaphor']} as the central metaphor of the story.

            FORMAT: Write the FULL story in English first, then the FULL story in the local vernacular of {loc} (Odia if Bhubaneswar). Use [ENGLISH] and [LOCAL] markers.
            """

            client = anthropic.Anthropic(api_key=api_key)
            col_e, col_l = st.columns(2)
            col_e.subheader("🇬🇧 English")
            col_l.subheader("🏠 Local")
            e_placeholder = col_e.empty()
            l_placeholder = col_l.empty()
            full_resp = ""

            try:
                with client.messages.stream(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=8192,
                    messages=[{"role": "user", "content": prompt}]
                ) as stream:
                    for text in stream.text_stream:
                        full_resp += text
                        if "[LOCAL]" in full_resp:
                            parts = full_resp.split("[LOCAL]")
                            e_text = parts[0].replace("[ENGLISH]", "").strip()
                            l_text = parts[1].strip()
                            e_placeholder.markdown(e_text)
                            l_placeholder.markdown(l_text + " ▌")
                        else:
                            e_placeholder.markdown(full_resp.replace("[ENGLISH]", "").strip() + " ▌")

                st.session_state['last_story_output'] = full_resp
                st.balloons()

            except Exception as e:
                st.error(f"Story generation failed: {e}")

else:
    st.info("👈 Select a data source in the sidebar to begin.")


# ============================================================
# 9. EXPORT SECTION
# ============================================================
if st.session_state.get('last_story_output'):
    st.divider()
    st.subheader("💾 Save for the Field")
    fname = f"Tide_Tale_{st.session_state['user_location'].replace(' ', '_').replace(',', '')}.txt"
    st.download_button(
        label="📥 Download Story as Text File",
        data=st.session_state['last_story_output'],
        file_name=fname,
        mime="text/plain",
        key="download_btn"
    )
