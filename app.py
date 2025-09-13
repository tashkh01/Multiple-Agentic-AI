import streamlit as st
from openai import OpenAI
import anthropic
from google import genai
from google.genai.types import GenerateContentConfig


st.set_page_config(page_title="Peer Response Orchestrator (BYOK)", layout="wide")

# --- UI ---
st.title("Peer Response Orchestrator — BYOK (Friends & Testing)")
st.caption("Pipeline: ChatGPT 5 Auto → Claude Sonnet 4 → Gemini 2.5 Flash (combine) → optional critique/revise")

with st.sidebar:
    st.subheader("Bring Your Own Keys (not stored)")
    openai_key = st.text_input("OpenAI API Key", type="password")
    anthropic_key = st.text_input("Anthropic API Key", type="password")
    google_key = st.text_input("Google (Gemini) API Key", type="password")
    
debug = st.sidebar.checkbox("Debug mode", value=False)

st.markdown("---")
st.subheader("Models (override if needed)")
    openai_model = st.text_input("OpenAI model", value="gpt-5-auto")
    anthropic_model = st.text_input("Anthropic model", value="claude-sonnet-4")
    gemini_model = st.text_input("Gemini model", value="gemini-2.5-flash")

st.markdown("---")
    st.subheader("Guardrails")
    attest = st.checkbox("I will use this for drafting/learning; I’ll follow my institution’s rules.", value=False)
    min_words = st.number_input("Min words", value=150, step=10)
    max_words = st.number_input("Max words", value=250, step=10)

st.write("### Original Post")
original_post = st.text_area("Paste the discussion prompt/post here:", height=240)

# Buttons
colA, colB = st.columns(2)
run = colA.button("Generate")
clear = colB.button("Clear")

# Divider line (optional)
st.markdown("---")

# Test keys button
test = st.button("Test keys (ping providers)")
if test:
    # --- OpenAI ping ---
    try:
        client = OpenAI(api_key=openai_key)
        _ = client.chat.completions.create(
            model=openai_model or "gpt-4o-mini",
            messages=[{"role": "user", "content": "Ping"}],
            temperature=0.0,
            max_tokens=5,
        )
        st.success("OpenAI: OK")
    except Exception as e:
        st.error("OpenAI ping failed.")
        if debug: st.exception(e)

    # --- Anthropic ping ---
    try:
        if anthropic_key:
            aclient = anthropic.Anthropic(api_key=anthropic_key)
            _ = aclient.messages.create(
                model=anthropic_model or "claude-3-5-sonnet-latest",
                max_tokens=5,
                temperature=0.0,
                messages=[{"role": "user", "content": "Ping"}],
            )
            st.success("Anthropic: OK")
        else:
            st.info("Anthropic: no key provided")
    except Exception as e:
        st.error("Anthropic ping failed.")
        if debug: st.exception(e)

    # --- Gemini ping ---
    try:
        if google_key:
            g = genai.Client(api_key=google_key)
            cfg = GenerateContentConfig(temperature=0.0, max_output_tokens=5)
            _ = g.models.generate_content(
                model=gemini_model or "gemini-1.5-flash",
                contents="Ping",
                config=cfg,
            )
            st.success("Gemini: OK")
        else:
            st.info("Gemini: no key provided")
    except Exception as e:
        st.error("Gemini ping failed.")
        if debug: st.exception(e)

# Continue with rest of your UI
st.subheader("Models (override if needed)")
