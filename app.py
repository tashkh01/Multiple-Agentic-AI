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

colA, colB = st.columns(2)
run = colA.button("Generate")
clear = colB.button("Clear")
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

if clear:
    st.experimental_rerun()

# --- Prompts ---
def prompt_peer(original_post: str) -> str:
    return f"Please provide a peer response to the following:\n\n{original_post}"

def prompt_combine(original_post: str, a: str, b: str) -> str:
    return (
        "Please combine what I have pasted and rewrite in a polished way:\n\n"
        f"ORIGINAL POST:\n{original_post}\n\n"
        f"PEER RESPONSE A (ChatGPT):\n{a}\n\n"
        f"PEER RESPONSE B (Claude):\n{b}\n"
    )

def prompt_critic(text: str, min_w: int, max_w: int) -> str:
    return (
        "You are a strict reviewer for a graduate discussion reply. "
        f"Check the text against 4 criteria:\n"
        f"1) Word count between {min_w}-{max_w}\n"
        "2) Collegial, constructive tone\n"
        "3) No fabricated citations; only reference what the user provided\n"
        "4) Clear takeaway + one practical suggestion\n"
        "Respond with a short verdict line: PASS or FAIL, then a single paragraph of fixes if FAIL.\n\n"
        f"TEXT:\n{text}"
    )

def prompt_revise(text: str, feedback: str) -> str:
    return (
        "Revise the following to satisfy the reviewer’s notes. Keep it concise and polished.\n\n"
        f"REVIEWER NOTES:\n{feedback}\n\nTEXT TO REVISE:\n{text}"
    )

# --- Provider calls (synchronous for Streamlit simplicity) ---
def call_openai(api_key: str, model: str, post: str) -> str:
    client = OpenAI(api_key=api_key)
    res = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt_peer(post)}],
        temperature=0.4,
    )
    return res.choices[0].message.content.strip()

def call_anthropic(api_key: str, model: str, post: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    res = client.messages.create(
        model=model,
        max_tokens=800,
        temperature=0.4,
        messages=[{"role":"user","content":prompt_peer(post)}],
    )
    return "".join(getattr(b, "text", "") for b in res.content).strip()

def call_gemini(api_key: str, model: str, original: str, a: str, b: str) -> str:
    g = genai.Client(api_key=api_key)
    cfg = GenerateContentConfig(temperature=0.4, max_output_tokens=900)
    out = g.models.generate_content(
        model=model,
        contents=prompt_combine(original, a, b),
        config=cfg
    )
    return out.text.strip()

def critic_gpt(api_key: str, model: str, text: str, min_w: int, max_w: int) -> str:
    client = OpenAI(api_key=api_key)
    res = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt_critic(text, min_w, max_w)}],
        temperature=0.0,
    )
    return res.choices[0].message.content.strip()

def revise_gemini(api_key: str, model: str, text: str, notes: str) -> str:
    g = genai.Client(api_key=api_key)
    cfg = GenerateContentConfig(temperature=0.3, max_output_tokens=900)
    out = g.models.generate_content(
        model=model,
        contents=prompt_revise(text, notes),
        config=cfg
    )
    return out.text.strip()

# --- Run pipeline ---
if run:
    if not attest:
        st.error("Please check the attestation box to proceed.")
    elif len(original_post.strip()) < 20:
        st.error("Please paste a longer original post (≥ 20 characters).")
    else:
        peer_a, peer_b, combined = "", "", ""
        review, final, revised = "", "", False

        # --- OpenAI (GPT) ---
        try:
            with st.spinner("ChatGPT drafting…"):
                if not openai_key:
                    raise RuntimeError("Missing OpenAI key.")
                peer_a = call_openai(openai_key, openai_model, original_post)
            st.success("OpenAI OK")
        except Exception as e:
            st.error("OpenAI error. Check key/model (try model: gpt-4o-mini).")
            if debug: st.exception(e)

        # --- Anthropic (Claude) ---
        try:
            if anthropic_key:
                with st.spinner("Claude drafting…"):
                    peer_b = call_anthropic(anthropic_key, anthropic_model, original_post)
                st.success("Anthropic OK")
            else:
                st.info("Skipping Claude (no key).")
        except Exception as e:
            st.error("Anthropic error. Common causes: $5 prepay needed, invalid key, or model id. Try: claude-3-5-sonnet-latest.")
            if debug: st.exception(e)

        # --- Gemini (Combine) ---
        try:
            if google_key and peer_a and peer_b:
                with st.spinner("Gemini combining…"):
                    combined = call_gemini(google_key, gemini_model, original_post, peer_a, peer_b)
                st.success("Gemini (combine) OK")
            elif not google_key:
                st.info("Skipping Gemini combine (no Google key).")
            elif not (peer_a and peer_b):
                st.info("Skipping Gemini combine (need both GPT and Claude drafts).")
        except Exception as e:
            st.error("Gemini error. Check key/model (try model: gemini-1.5-flash) or quota.")
            if debug: st.exception(e)

        # Show drafts if present
        if peer_a:
            st.write("#### Peer Response A — ChatGPT")
            st.write(peer_a)
        if peer_b:
            st.write("#### Peer Response B — Claude")
            st.write(peer_b)
        if combined:
            st.write("#### Combined (Gemini initial)")
            st.write(combined)

        # --- Critique & Revise (only if we have a combine) ---
        try:
            if combined and openai_key:
                with st.spinner("Critiquing & (if needed) revising…"):
                    review = critic_gpt(openai_key, openai_model, combined, int(min_words), int(max_words))
                    verdict = review.splitlines()[0].strip().upper()
                    if verdict.startswith("PASS"):
                        final = combined
                        revised = False
                    else:
                        final = revise_gemini(google_key, gemini_model, combined, review) if google_key else combined
                        revised = not verdict.startswith("PASS")

                st.write("#### Reviewer Notes")
                st.code(review)
                st.write("#### Final Polished")
                st.write(final)
                st.info(f"Closed-loop status: {'Revised' if revised else 'Passed on first try'}")
        except Exception as e:
            st.error("Critique/revise step failed.")
            if debug: st.exception(e)
