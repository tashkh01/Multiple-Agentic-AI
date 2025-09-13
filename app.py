import streamlit as st
from openai import OpenAI
import anthropic
from google import genai
from google.genai.types import GenerateContentConfig

# ---------- Page ----------
st.set_page_config(page_title="Peer Response Orchestrator (BYOK)", layout="wide")
st.title("Peer Response Orchestrator — BYOK")
st.caption("Pipeline: OpenAI → Anthropic → Gemini (combine) → optional critique/revise | No data is stored.")

# ---------- Sidebar: Keys ----------
with st.sidebar:
    st.subheader("Bring Your Own Keys (required)")
    openai_key = st.text_input("OpenAI API Key (sk-...)", type="password")
    anthropic_key = st.text_input("Anthropic API Key (sk-ant-...)", type="password")
    google_key = st.text_input("Google (Gemini) API Key (AIza...)", type="password")

    st.markdown("---")
    st.subheader("Models (override if needed)")
    # Use widely available defaults
    openai_model = st.text_input("OpenAI model", value="gpt-4o-mini")
    anthropic_model = st.text_input("Anthropic model", value="claude-3-5-sonnet-latest")
    gemini_model = st.text_input("Gemini model", value="gemini-1.5-flash")

    st.markdown("---")
    st.subheader("Guardrails")
    attest = st.checkbox("I will use this for drafting/learning and follow my institution’s rules.", value=False)
    min_words = st.number_input("Min words", value=150, step=10)
    max_words = st.number_input("Max words", value=250, step=10)

    st.markdown("---")
    debug = st.checkbox("Debug mode (show detailed errors)", value=False)

# ---------- Prompts ----------
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
        "You are a strict reviewer for a graduate discussion reply.\n"
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

# ---------- Provider Calls ----------
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
    # concatenate any text blocks
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

# ---------- Main UI ----------
st.write("### Original Post")
original_post = st.text_area("Paste the discussion prompt/post here:", height=240)

# Buttons row
colA, colB = st.columns(2)
run = colA.button("Generate")
clear = colB.button("Clear")
if clear:
    st.rerun()

# Ping providers (tiny calls) to verify keys/models quickly
st.markdown("---")
test = st.button("Test keys (ping providers)")
if test:
    # OpenAI ping
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
        st.error("OpenAI ping failed. Check key or model (try gpt-4o-mini).")
        if debug: st.exception(e)

    # Anthropic ping
    try:
        if anthropic_key:
            aclient = anthropic.Anthropic(api_key=anthropic_key)
            _ = aclient.messages.create(
                model=anthropic_model or "claude-3-5-sonnet-latest",
                max_tokens=5,
                temperature=0.0,
                messages=[{"role":"user","content":"Ping"}],
            )
            st.success("Anthropic: OK")
        else:
            st.info("Anthropic: no key provided")
    except Exception as e:
        st.error("Anthropic ping failed. Check key, $5 prepay, or model (try claude-3-5-sonnet-latest).")
        if debug: st.exception(e)

    # Gemini ping
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
        st.error("Gemini ping failed. Check key/model (try gemini-1.5-flash) or quota.")
        if debug: st.exception(e)

# ---------- Pipeline (agentic: critique → (optional) revise) ----------
if run:
    if not attest:
        st.error("Please check the attestation box to proceed.")
    elif len(original_post.strip()) < 20:
        st.error("Please paste a longer original post (≥ 20 characters).")
    elif not (openai_key and anthropic_key and google_key):
        st.error("Please enter all three API keys in the sidebar.")
    else:
        peer_a, peer_b, combined = "", "", ""
        review, final, revised = "", "", False

        # OpenAI (draft A)
        try:
            with st.spinner("ChatGPT drafting…"):
                peer_a = call_openai(openai_key, openai_model, original_post)
            st.success("OpenAI OK")
        except Exception as e:
            st.error("OpenAI error. Check key/model (try gpt-4o-mini).")
            if debug: st.exception(e)

        # Anthropic (draft B)
        try:
            with st.spinner("Claude drafting…"):
                peer_b = call_anthropic(anthropic_key, anthropic_model, original_post)
            st.success("Anthropic OK")
        except Exception as e:
            st.error("Anthropic error. Check key, $5 prepay, model (try claude-3-5-sonnet-latest).")
            if debug: st.exception(e)

        # Show drafts if present
        if peer_a:
            st.write("#### Peer Response A — ChatGPT")
            st.write(peer_a)
        if peer_b:
            st.write("#### Peer Response B — Claude")
            st.write(peer_b)

        # Gemini (combine)
        try:
            if peer_a and peer_b:
                with st.spinner("Gemini combining…"):
                    combined = call_gemini(google_key, gemini_model, original_post, peer_a, peer_b)
                st.success("Gemini (combine) OK")
                st.write("#### Combined (Gemini initial)")
                st.write(combined)
            else:
                st.info("Skipping combine (need both GPT and Claude drafts).")
        except Exception as e:
            st.error("Gemini error. Check key/model (try gemini-1.5-flash) or quota.")
            if debug: st.exception(e)

        # Critique & (optional) revise via Gemini if combine exists
        try:
            if combined:
                with st.spinner("Critiquing & (if needed) revising…"):
                    review = critic_gpt(openai_key, openai_model, combined, int(min_words), int(max_words))
                    verdict = review.splitlines()[0].strip().upper()
                    if verdict.startswith("PASS"):
                        final = combined
                        revised = False
                    else:
                        final = revise_gemini(google_key, gemini_model, combined, review)
                        revised = True

                st.write("#### Reviewer Notes")
                st.code(review)
                st.write("#### Final Polished")
                st.write(final)
                st.info(f"Closed-loop status: {'Revised' if revised else 'Passed on first try'}")
        except Exception as e:
            st.error("Critique/revise step failed.")
            if debug: st.exception(e)

# ---------- Footer ----------
st.markdown("---")
st.caption("BYOK • No storage • For drafting/learning only • Use within your institution’s rules.")
