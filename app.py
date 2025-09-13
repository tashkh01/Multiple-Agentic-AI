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
    elif not (openai_key and anthropic_key and google_key):
        st.error("Please enter all three API keys in the sidebar.")
    elif len(original_post.strip()) < 20:
        st.error("Please paste a longer original post (≥ 20 characters).")
    else:
        try:
            with st.spinner("ChatGPT 5 Auto drafting…"):
                peer_a = call_openai(openai_key, openai_model, original_post)
            with st.spinner("Claude Sonnet 4 drafting…"):
                peer_b = call_anthropic(anthropic_key, anthropic_model, original_post)

            st.success("Drafts ready.")
            col1, col2 = st.columns(2)
            with col1:
                st.write("#### Peer Response A — ChatGPT 5 Auto")
                st.write(peer_a)
            with col2:
                st.write("#### Peer Response B — Claude Sonnet 4")
                st.write(peer_b)

            with st.spinner("Gemini 2.5 Flash combining…"):
                combined = call_gemini(google_key, gemini_model, original_post, peer_a, peer_b)
            st.write("#### Combined (Gemini initial)")
            st.write(combined)

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
            # Avoid exposing sensitive details
            st.error("Generation failed. Check your keys/models and try again.")
