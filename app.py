import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import anthropic
import requests

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Tide Tales", layout="wide")

if 'user_location' not in st.session_state:
    st.session_state['user_location'] = "Bhubaneswar, India"
if 'data_mapped' not in st.session_state:
    st.session_state['data_mapped'] = None

# --- 2. SIDEBAR (The Inputs) ---
with st.sidebar:
    st.title("🌊 Tide Tales Settings")
    api_key = st.text_input("Anthropic API Key", type="password", key="api_key")
    
    # Accurate Location Control
    st.session_state['user_location'] = st.text_input(
        "📍 Location Context", 
        value=st.session_state['user_location'], 
        key="loc_input"
    )
    
    st.divider()
    uploaded_file = st.file_uploader("Step 1: Upload Scientific CSV", type="csv")

# --- 3. AI DATA READER (Reading and Interpreting any form/type) ---
def ai_interpret_data(df, api_key):
    """Sends sample rows to AI to identify Time, Measurement, and Science Type."""
    client = anthropic.Anthropic(api_key=api_key)
    sample = df.head(5).to_string()
    headers = df.columns.tolist()
    
    prompt = f"""
    Analyze this dataset sample:
    {sample}
    
    Headers: {headers}
    
    Identify:
    1. Which column is the Time/Year?
    2. Which column is the primary Measurement/Data?
    3. What is the unit/type of science (e.g., Temperature in C, AQI, Sea Level in mm)?
    
    Return ONLY in this format:
    YearCol: [name]
    DataCol: [name]
    Unit: [unit]
    Label: [label]
    """
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )
    res = response.content[0].text
    
    # Parsing the AI's response
    mapping = {}
    for line in res.strip().split('\n'):
        if ':' in line:
            key, val = line.split(':', 1)
            mapping[key.strip()] = val.strip()
    return mapping

# --- 4. MAIN APP LOGIC ---
st.title("🌊 Tide Tales")

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file, na_values="***")
    
    # Step 2: AI Sniffing the Data
    if st.button("🔍 AI: Interpret Data Structure"):
        if not api_key:
            st.error("Please enter an API Key to use the AI Reader.")
        else:
            with st.spinner("AI is reading your data columns..."):
                mapping = ai_interpret_data(raw_df, api_key)
                
                # Create a cleaned, standardized dataframe
                y_col, d_col = mapping['YearCol'], mapping['DataCol']
                clean_df = raw_df[[y_col, d_col]].copy()
                clean_df.columns = ['year', 'val']
                clean_df['year'] = pd.to_numeric(clean_df['year'], errors='coerce')
                clean_df['val'] = pd.to_numeric(clean_df['val'], errors='coerce')
                
                # Save results to session state
                st.session_state['data_mapped'] = clean_df.dropna()
                st.session_state['sci_metadata'] = mapping
                st.success(f"AI identified this as {mapping['Label']} data.")

# --- 5. NUMPY PLOTTER & MATH ---
if st.session_state['data_mapped'] is not None:
    data = st.session_state['data_mapped']
    meta = st.session_state['sci_metadata']
    
    # Time Slider
    min_y, max_y = int(data['year'].min()), int(data['year'].max())
    selected_range = st.slider("Select Time Range", min_y, max_y, (min_y, max_y))
    f_df = data[(data['year'] >= selected_range[0]) & (data['year'] <= selected_range[1])]

    # NUMPY MATH: Calculate Trendline
    slope, intercept = np.polyfit(f_df['year'], f_df['val'], 1)
    
    # PLOTTING correctly based on interpreted type
    st.header(f"📊 Evidence: {meta['Label']}")
    fig = px.line(f_df, x='year', y='val', template="plotly_dark", 
                  labels={'year': 'Year', 'val': meta['Unit']},
                  title=f"Observing {meta['Label']} trends in {st.session_state['user_location']}")
    
    # Add Trendline Scatter
    fig.add_scatter(x=f_df['year'], y=slope*f_df['year'] + intercept, 
                    name="Mathematical Trend", line=dict(color='red', dash='dot'))
    st.plotly_chart(fig, use_container_width=True)

    # --- 6. AI STORY WRITER ---
    st.divider()
    if st.button("✨ Weave 1500-Word Narrative"):
        if not api_key:
            st.error("API Key required.")
        else:
            # Construct the prompt with the "Locked Facts" from our math
            net_change = f_df['val'].iloc[-1] - f_df['val'].iloc[0]
            
            prompt = f"""
            You are a master storyteller. 
            LOCATION: {st.session_state['user_location']}
            DATA FACTS: {meta['Label']} change in {meta['Unit']}. 
            Period: {selected_range[0]} to {selected_range[1]}.
            Total Change: {round(net_change, 2)}. 
            Rate of Change (Slope): {round(slope, 4)} per year.

            TASK:
            Write a 1,500-word immersive story grounded in this data. 
            1. Use local folklore and metaphors of {st.session_state['user_location']}.
            2. The data is the environment, not a villain.
            3. Chaptered structure.
            4. Output FULL story in English, then FULL story in the local vernacular of {st.session_state['user_location']}.
            
            FORMAT: [ENGLISH] ... [LOCAL]
            """
            
            client = anthropic.Anthropic(api_key=api_key)
            col_e, col_l = st.columns(2)
            e_p, l_p = col_e.empty(), col_l.empty()
            full_resp = ""
            
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
                        e_p.markdown(e_text); l_p.markdown(l_text + " ▌")
                    else:
                        e_p.markdown(full_resp.replace("[ENGLISH]", "").strip() + " ▌")
            st.balloons()
else:
    st.info("Please upload a file and click 'Interpret Data Structure' to begin.")
