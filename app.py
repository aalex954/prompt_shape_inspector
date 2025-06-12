#!/usr/bin/env python3
# ------------------------------------------------------------
#  LLM Prompt Shape Inspector  ▸  Edge-Finder • Polysemy • Contractor
# ------------------------------------------------------------
import os, json, re, functools, math, html
import numpy as np
import streamlit as st
import tiktoken, pyperclip
from openai import OpenAI
from dotenv import load_dotenv; load_dotenv()


# ------------------ CONFIG ----------------------------------
EMBED_MODEL         = "text-embedding-3-small"
POLY_SENSES         = 4          # how many “imagined” senses per word (WordNet proxy)
POLY_STRESS_TAU     = 0.25       # what counts as “high” polysemy
EDGE_TAU            = 0.30       # default edge-score threshold
LLM_DRIFT_TAU       = 0.07       # occlusion drift threshold
#openai.api_key      = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ENC                 = tiktoken.encoding_for_model(EMBED_MODEL)
# ------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _lazy_wordnet():
    import nltk
    try:
        from nltk.corpus import wordnet as wn
        # Touch a synset to verify the corpus is really present
        _ = wn.synset('dog.n.01')
    except (LookupError, OSError):
        nltk.download('wordnet', quiet=True)   # ← everything happens automatically
        from nltk.corpus import wordnet as wn
    return wn


def embed(texts):
    """
    Accepts str or list[str], returns a NumPy vector or list thereof
    using the new openai-python ≥1.0.0 interface.
    """
    if isinstance(texts, str):
        texts = [texts]

    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [np.asarray(item.embedding, dtype=np.float32) for item in resp.data]
    vecs = [v / np.linalg.norm(v) for v in vecs]       # unit-length
    return vecs if len(vecs) > 1 else vecs[0]


# ---------------- 4.1 Edge-Finder ---------------------------
def edge_scores(prompt_tokens, constraint_vecs):
    """Return list of (edge_score ∈ [0,1])."""
    tok_vecs = embed(prompt_tokens)
    if isinstance(tok_vecs, np.ndarray):      # len==1 corner case
        tok_vecs = [tok_vecs]
    scores = []
    for v in tok_vecs:
        score = max(abs(float(v @ c)) for c in constraint_vecs)
        scores.append(score)
    return scores

# ---------------- 4.2 Polysemy Stress -----------------------
def poly_stress(tokens):
    wn = _lazy_wordnet()
    out = []
    for t in tokens:
        synsets = wn.synsets(t.strip())[:POLY_SENSES]
        if len(synsets) < 2:
            out.append(0.0)
            continue
        glosses = [s.definition() for s in synsets]
        vecs = embed(glosses)
        if isinstance(vecs, np.ndarray):
            vecs = [vecs]
        centroid = np.mean(vecs, axis=0)
        var = float(np.mean([np.linalg.norm(v - centroid) for v in vecs]))
        out.append(var)
    return out

# ---------------- Occlusion Drift ---------------------------
def occlusion_drift(prompt_tokens, full_vec):
    drift = []
    for i in range(len(prompt_tokens)):
        sub = prompt_tokens[:i] + prompt_tokens[i+1:]
        if len(sub) == 0:
            drift.append(0.0)
            continue
        vec_sub = embed("".join(sub))
        drift.append(float(np.linalg.norm(full_vec - vec_sub)))
    return drift

# ---------------- Contractor (4.3) --------------------------
def contractor(prompt_tokens, edge_mask, poly_mask):
    """Drop everything that is *not* edge & not poly if occlusion drift < τ."""
    to_keep = []
    full_vec = embed("".join(prompt_tokens))
    drift   = occlusion_drift(prompt_tokens, full_vec)
    for tok, drift_val, is_edge, is_poly in zip(prompt_tokens, drift, edge_mask, poly_mask):
        if is_edge or is_poly or drift_val > LLM_DRIFT_TAU:
            to_keep.append(tok)
    # Remove leading/trailing spaces introduced by token boundaries
    contracted = re.sub(r'\s+', ' ', "".join(to_keep)).strip()
    return contracted

# -------------------- UI Helpers ----------------------------
def style_token(tok, edge, poly, edge_on, poly_on):
    """Return HTML <span> w/ colour & tooltip."""
    e_score = f"{edge:.2f}"
    p_score = f"{poly:.2f}"
    title = f"edge score: {e_score} | poly stress: {p_score}"
    alpha = 0.0
    if edge_on:
        alpha += np.clip(edge / 0.8, 0, 1)
    if poly_on:
        alpha += np.clip(poly / 0.5, 0, 1)
    alpha = min(alpha, 1)
    rgb = f"rgba(255,87,51,{alpha:.2f})"   # heat-color
    safe = html.escape(tok)
    return f"<span title='{title}' style='background:{rgb}; padding:2px'>{safe}</span>"

def copy_to_clipboard(text):
    try:
        pyperclip.copy(text)
        return True
    except pyperclip.PyperclipException:
        return False

# ============================================================
#                     Streamlit GUI
# ============================================================
st.set_page_config(page_title="LLM Prompt Shape Inspector", page_icon="🌐", layout="wide")

col_prompt, col_opts = st.columns([3,1])

with col_prompt:
    st.markdown("## ✏️ Prompt")
    user_prompt = st.text_area("Enter your prompt …", height=200, key="prompt")

with col_opts:
    st.markdown("### 🔧 Constraint Phrases\n*(one per line)*")
    raw_constraints = st.text_area("constraints", value="context: cybersecurity\nformat: JSON\nstyle: strict, formal", height=150)
    constraints = [c.strip() for c in raw_constraints.splitlines() if c.strip()]

run_btn = st.button("▶ Analyse")

if run_btn and user_prompt.strip():
    # --- tokenise exactly like the embedding model ---
    token_ids   = ENC.encode(user_prompt, disallowed_special=())
    tokens      = [ENC.decode([tid]) for tid in token_ids]

    st.session_state["tokens"] = tokens
    st.session_state["constraint_vecs"] = embed(constraints)
    st.session_state["edge"]   = edge_scores(tokens, st.session_state["constraint_vecs"])
    st.session_state["poly"]   = poly_stress(tokens)
    st.session_state["full_vec"] = embed(user_prompt)

if "tokens" in st.session_state:
    tokens  = st.session_state["tokens"]
    edge_v  = st.session_state["edge"]
    poly_v  = st.session_state["poly"]

    edge_mask = [s >= EDGE_TAU for s in edge_v]
    poly_mask = [s >= POLY_STRESS_TAU for s in poly_v]

    st.sidebar.markdown("## Heat-Map Toggles")
    show_edge = st.sidebar.checkbox("Edge-Finder", value=True)
    show_poly = st.sidebar.checkbox("Polysemy Stress", value=False)

    # ---------------- contractor preview ----------------
    contracted = contractor(tokens, edge_mask, poly_mask)
    st.sidebar.markdown("### Contractor Output")
    st.sidebar.code(contracted, language="markdown")
    if st.sidebar.button("📋 Copy contracted prompt"):
        ok = copy_to_clipboard(contracted)
        st.sidebar.success("Copied!" if ok else "Clipboard failed.")

    # ---------------- budget read-out -------------------
    tot_poly = sum(poly_v)
    st.sidebar.markdown(f"### Polysemy Budget\nTotal σ = **{tot_poly:.2f}**  (τ={POLY_STRESS_TAU})")

    # ---------------- main visualisation ----------------
    st.markdown("### 🖼️  Heat-Map")
    html_tokens = [
        style_token(tok, e, p, show_edge, show_poly)
        for tok, e, p in zip(tokens, edge_v, poly_v)
    ]
    st.markdown(
        "<div style='font-family:monospace; line-height:2em'>" +
        "".join(html_tokens) +
        "</div>",
        unsafe_allow_html=True
    )

    # ---------------- guidance panel -------------------
    st.markdown("---")
    st.markdown("### 📌 Guidance")
    if any(poly_mask):
        st.markdown(
            "* **Sense-Locking** – consider adding clarifiers right after "
            f"{sum(poly_mask)} polysemous high-σ word(s)."
        )
    if not any(edge_mask):
        st.markdown(
            "* **Edge Reinforcement** – no edge tokens detected above τ; "
            "your prompt may drift."
        )
    low_info = [t for t,e,p in zip(tokens,edge_mask,poly_mask) if not e and not p and len(t.strip()) <= 3]
    if len(low_info) > 4:
        st.markdown(
            "* **Dimensional Drop-Out** – consider dropping fillers like "
            f"`{' '.join(low_info[:6])} …`"
        )
    if tot_poly > 4:
        st.markdown(
            "* **Polysemy Budget** exceeded – tighten wording or add examples."
        )

else:
    st.info("Enter a prompt and press **Analyse**.")
