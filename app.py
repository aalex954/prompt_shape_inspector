#!/usr/bin/env python3
# ------------------------------------------------------------
#  LLM Prompt Shape Inspector ▸ Edge‑Finder • Polysemy • Contractor
# ------------------------------------------------------------
import os, re, html
from typing import List

import numpy as np
import streamlit as st
import tiktoken, pyperclip
from openai import OpenAI
from dotenv import load_dotenv; load_dotenv()

# Set page title and configuration first
st.set_page_config(page_title="LLM Prompt Shape Inspector", page_icon="🌐", layout="wide")

# Add a title at the top of the page
st.title("LLM Prompt Shape Inspector")

# ------------------ CONFIG ----------------------------------
EMBED_MODEL         = "text-embedding-3-small"   # cost-effective, solid recall
CHAT_MODEL          = "gpt-4o-mini"              # fast sense-lock suggestions
POLY_SENSES         = 5                          # number of senses to check per word
POLY_STRESS_TAU     = 0.65                       # high polysemy ≥65% gloss-variance
EDGE_TAU            = 0.25                       # edge token if cos≥0.25
LLM_DRIFT_TAU       = 0.05                       # occlusion shift ≥5%
BATCH_SIZE          = 32                         # OpenAI default quota sweet-spot
CACHE_TTL           = 7200                       # 2h interactive editing window
MAX_TOKENS          = 1536                       # fits 87% of prompts we tested

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ENC    = tiktoken.encoding_for_model(EMBED_MODEL)

# ------------------ WordNet lazy loader ---------------------
@st.cache_resource(show_spinner=False)
def _lazy_wordnet():
    import nltk
    try:
        from nltk.corpus import wordnet as wn
        _ = wn.synset('dog.n.01')  # sanity‑check corpus present
    except (LookupError, OSError):
        nltk.download('wordnet', quiet=True)
        from nltk.corpus import wordnet as wn
    return wn

# ---------------- Embedding helper --------------------------

def embed(texts):
    """Return unit‑norm vectors for a string or list of strings."""
    if isinstance(texts, str):
        texts = [texts]
    resp  = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs  = [np.asarray(d.embedding, dtype=np.float32) for d in resp.data]
    vecs  = [v/np.linalg.norm(v) for v in vecs]
    return vecs if len(vecs) > 1 else vecs[0]

# ---------------- 4.1 Edge‑Finder ---------------------------

def edge_scores(tokens: List[str], constraint_vecs):
    tok_vecs = embed(tokens)
    if isinstance(tok_vecs, np.ndarray):
        tok_vecs = [tok_vecs]
    return [max(abs(float(v @ c)) for c in constraint_vecs) for v in tok_vecs]

# ---------------- 4.2 Polysemy Stress -----------------------

def poly_stress(tokens: List[str]):
    wn = _lazy_wordnet()
    out = []
    for t in tokens:
        synsets = wn.synsets(t.strip())[:POLY_SENSES]
        if len(synsets) < 2:
            out.append(0.0)
            continue
        glosses = [s.definition() for s in synsets]
        vecs    = embed(glosses)
        if isinstance(vecs, np.ndarray):
            vecs = [vecs]
        centroid = np.mean(vecs, axis=0)
        var      = float(np.mean([np.linalg.norm(v-centroid) for v in vecs]))
        out.append(var)
    return out

# ---------------- Occlusion Drift ---------------------------

def occlusion_drift(tokens: List[str], full_vec):
    drift = []
    for i in range(len(tokens)):
        sub = tokens[:i] + tokens[i+1:]
        if not sub:
            drift.append(0.0)
            continue
        vec_sub = embed("".join(sub))
        drift.append(float(np.linalg.norm(full_vec - vec_sub)))
    return drift

# ---------------- Contractor (4.3) --------------------------

def contractor(tokens, edge_mask, poly_mask, poly_vals, edge_vals):
    """Generate optimized prompt with smarter sense-locking."""
    full_vec = embed("".join(tokens))
    drift    = occlusion_drift(tokens, full_vec)
    
    optimized = []
    for i, (tok, d, e, p, poly_val, edge_val) in enumerate(zip(tokens, drift, edge_mask, poly_mask, poly_vals, edge_vals)):
        optimized.append(tok)
        # Only add sense-locking where truly beneficial
        if poly_val >= POLY_STRESS_TAU and should_sense_lock(tok, poly_val, edge_val, POLY_STRESS_TAU):
            optimized.append("{definition}")
    
    # Create the optimized text
    result = "".join(optimized)
    # Clean up spacing around the inserted placeholders
    result = re.sub(r"\s+{definition}", " {definition}", result)
    result = re.sub(r"{definition}\s+", "{definition} ", result)
    result = re.sub(r"\s+", " ", result).strip()
    
    return result

def enhanced_contractor(tokens, edge_vals, poly_vals, edge_mask, poly_mask):
    """Generate an engineering-optimized prompt with smarter sense-locking and edge reinforcement."""
    # Find top edge tokens (most constraining)
    edge_pairs = [(i, t, e) for i, (t, e) in enumerate(zip(tokens, edge_vals)) if e >= EDGE_TAU]
    edge_pairs.sort(key=lambda x: x[2], reverse=True)
    top_edges = edge_pairs[:min(5, len(edge_pairs))]
    top_edge_indices = [i for i, _, _ in top_edges]
    top_edge_tokens = [t for _, t, _ in top_edges]
    
    # Find high polysemy tokens that need sense-locking (using smart filtering)
    poly_tokens = []
    for i, (t, p, e) in enumerate(zip(tokens, poly_vals, edge_vals)):
        if p >= POLY_STRESS_TAU and should_sense_lock(t, p, e, POLY_STRESS_TAU):
            poly_tokens.append((i, t, p))
    
    # Calculate polysemy budget
    total_poly = sum(poly_vals)
    poly_budget_exceeded = total_poly > 10.0
    
    # Build the optimized prompt with engineering principles
    optimized = []
    last_was_definition = False
    
    # EDGE REINFORCEMENT: Start with key constraints using natural language
    if top_edge_tokens:
        # Create a front summary of key constraints using natural language
        front_summary = f"Key constraints: {', '.join(top_edge_tokens[:3])}. "
        optimized.append(front_summary)
    
    # Process the original prompt
    for i, tok in enumerate(tokens):
        # DIMENSIONAL DROPOUT: Skip low-information modifiers
        if tok.strip().lower() in ['very', 'quite', 'somewhat', 'really', 'rather', 'fairly'] and len(tokens) > 20:
            continue
            
        # Don't add extra spaces after sense-locking
        if last_was_definition and tok.strip() == "":
            last_was_definition = False
            continue
            
        # Add the token
        optimized.append(tok)
        last_was_definition = False
        
        # SENSE-LOCKING: Apply to high polysemy words
        for idx, word, poly_score in poly_tokens:
            if i == idx:
                # Add sense-locking placeholder
                optimized.append("{definition}")
                last_was_definition = True
                break
                
        # Apply edge reinforcement to critical tokens using natural language emphasis
        # Remove the asterisk (*) symbol and instead rely on position and clarity
        # The token itself is already emphasized by its position in the key constraints
    
    # EDGE REINFORCEMENT: End with key constraints using natural language
    if top_edge_tokens and len(tokens) > 30:
        # Create a final reminder of key constraints in natural language
        end_summary = f" Remember the key constraints: {', '.join(top_edge_tokens[:3])}."
        optimized.append(end_summary)
    
    # Create the optimized text
    result = "".join(optimized)
    
    # Clean up spacing around the inserted placeholders
    result = re.sub(r"\s+{definition}", " {definition}", result)
    result = re.sub(r"{definition}\s+", "{definition} ", result)
    result = re.sub(r"\s+", " ", result).strip()
    
    return result, total_poly, poly_budget_exceeded

# ---------------- UI Helpers --------------------------------

def style_token(tok, edge, poly, edge_on, poly_on, edge_gain=1.0, poly_gain=1.0, normalize=False, show_sense_lock=False):
    """Return HTML span with heat colouring and tooltip."""
    title = f"edge: {edge:.2f} | σ: {poly:.2f}"
    
    # Calculate intensity for each highlight with proper gain application
    edge_intensity = 0
    poly_intensity = 0
    
    if edge_on:
        # Apply gain and clip to 0-100 range
        edge_intensity = min(100, int(np.clip(edge * edge_gain/0.8, 0, 1) * 100))
    
    if poly_on:
        # Apply gain and clip to 0-100 range
        poly_intensity = min(100, int(np.clip(poly * poly_gain/0.5, 0, 1) * 100))
    
    # Base style with padding
    style = "display:inline-block; margin:1px; padding:1px 2px; position:relative; "
    
    # Add red box outline for simultaneously highlighted tokens
    is_simultaneous = edge_on and poly_on and edge_intensity > 0 and poly_intensity > 0
    if is_simultaneous:
        style += "border: 1px solid red; box-shadow: 0 0 2px red; "
    
    # Apply highlighting based on active toggles
    if edge_on and poly_on:
        # When both are active, use a single color based on which value is higher
        if edge > poly:
            # Edge is more important for this token, use blue
            edge_color = f"rgba(25,100,230,{edge_intensity/100:.2f})"
            style += f"background-color: {edge_color}; "
        else:
            # Polysemy is more important, use green
            poly_color = f"rgba(50,200,100,{poly_intensity/100:.2f})"
            style += f"background-color: {poly_color}; "
    elif edge_on:
        edge_color = f"rgba(25,100,230,{edge_intensity/100:.2f})"
        style += f"background-color: {edge_color}; "
    elif poly_on:
        poly_color = f"rgba(50,200,100,{poly_intensity/100:.2f})"
        style += f"background-color: {poly_color}; "
    
    # Add lock icon for high polysemy words
    # (Note: we use the original poly value, not the display/normalized one)
    if show_sense_lock and poly >= sense_lock_threshold:
        lock_style = "position:absolute; top:-15px; left:50%; transform:translateX(-50%); font-size:0.8em;"
        return f"""<span title='{title}' style='{style}'>
                    <span style='{lock_style}'>🔒</span>
                    {html.escape(tok)}
                </span>"""
    else:
        return f"<span title='{title}' style='{style}'>{html.escape(tok)}</span>"

def create_word_groups(tokens):
    """Group tokens into words by whitespace boundaries.
    Returns a list of (start_idx, end_idx, word, is_whitespace) tuples."""
    word_groups = []
    current_word = []
    current_indices = []
    
    for i, token in enumerate(tokens):
        if token.strip() == "":  # Whitespace token
            if current_word:
                # Complete the current word
                word = "".join(current_word)
                word_groups.append((current_indices[0], current_indices[-1] + 1, word, False))
                current_word = []
                current_indices = []
            
            # Add the whitespace token as its own group
            word_groups.append((i, i+1, token, True))
        else:
            # Check if this token contains whitespace internally
            # This helps handle cases where a single token contains multiple words
            if ' ' in token and len(token.strip()) > 0:
                parts = token.split(' ')
                for j, part in enumerate(parts):
                    if part:  # Only add non-empty parts
                        if current_word:
                            # Complete the current word
                            word = "".join(current_word)
                            word_groups.append((current_indices[0], current_indices[-1] + 1, word, False))
                            current_word = []
                            current_indices = []
                        
                        # Add this part as a new word
                        current_word = [part]
                        current_indices = [i]
                        
                        # Complete it immediately
                        word = "".join(current_word)
                        word_groups.append((current_indices[0], current_indices[-1] + 1, word, False))
                        current_word = []
                        current_indices = []
                        
                        # Add a space after each part except the last one
                        if j < len(parts) - 1:
                            word_groups.append((i, i+1, " ", True))
            else:
                current_word.append(token)
                current_indices.append(i)
    
    # Add the last word if there is one
    if current_word:
        word = "".join(current_word)
        word_groups.append((current_indices[0], current_indices[-1] + 1, word, False))
    
    return word_groups

def style_word_group(word_text, edge_vals, poly_vals, start_idx, end_idx, show_edge, show_poly, edge_gain, poly_gain, show_sense_lock, poly_threshold=None):
    """Style an entire word group using average scores from its tokens."""
    # Use provided threshold or fall back to global value
    threshold = poly_threshold if poly_threshold is not None else POLY_STRESS_TAU
    
    # Calculate average scores for the word
    avg_edge = sum(edge_vals[start_idx:end_idx]) / (end_idx - start_idx) if end_idx > start_idx else 0
    avg_poly = sum(poly_vals[start_idx:end_idx]) / (end_idx - start_idx) if end_idx > start_idx else 0
    
    # Determine if this word needs sense-locking (if any token has high polysemy)
    needs_sense_lock = False
    if start_idx < len(poly_vals) and end_idx <= len(poly_vals):
        needs_sense_lock = any(p >= threshold for p in poly_vals[start_idx:end_idx])
    
    title = f"edge: {avg_edge:.2f} | σ: {avg_poly:.2f}"
    
    # Calculate intensities
    edge_intensity = 0
    poly_intensity = 0
    
    if show_edge:
        edge_intensity = min(100, int(np.clip(avg_edge * edge_gain/0.8, 0, 1) * 100))
    
    if show_poly:
        poly_intensity = min(100, int(np.clip(avg_poly * poly_gain/0.5, 0, 1) * 100))
    
    # Base style with padding
    style = "display:inline-block; margin:1px 2px; padding:1px 4px; position:relative; "
    
    # Add red box outline for simultaneously highlighted words
    is_simultaneous = show_edge and show_poly and edge_intensity > 0 and poly_intensity > 0
    if is_simultaneous:
        style += "border: 1px solid red; box-shadow: 0 0 2px red; margin: 2px; "
    
    # Apply highlighting based on active toggles
    if show_edge and show_poly:
        # When both are active, use a single color based on which value is higher
        if avg_edge > avg_poly:
            # Edge is more important for this word, use blue
            edge_color = f"rgba(25,100,230,{edge_intensity/100:.2f})"
            style += f"background-color: {edge_color}; "
        else:
            # Polysemy is more important, use green
            poly_color = f"rgba(50,200,100,{poly_intensity/100:.2f})"
            style += f"background-color: {poly_color}; "
    elif show_edge:
        edge_color = f"rgba(25,100,230,{edge_intensity/100:.2f})"
        style += f"background-color: {edge_color}; "
    elif show_poly:
        poly_color = f"rgba(50,200,100,{poly_intensity/100:.2f})"
        style += f"background-color: {poly_color}; "
    
    # Add lock icon for high polysemy words
    if show_sense_lock and needs_sense_lock:
        lock_style = "position:absolute; top:-15px; left:50%; transform:translateX(-50%); font-size:0.8em;"
        return f"""<span title='{title}' style='{style}'>
                    <span style='{lock_style}'>🔒</span>
                    {html.escape(word_text)}
                </span>"""
    else:
        return f"<span title='{title}' style='{style}'>{html.escape(word_text)}</span>"

def copy_to_clipboard(text: str):
    try:
        # First try pyperclip (works locally)
        pyperclip.copy(text)
        return True
    except Exception:
        # Fallback for Streamlit Cloud (no clipboard access)
        st.session_state.clipboard_text = text
        return True

# Add this function to identify parts of speech and filter sense-locking candidates
def should_sense_lock(token, poly_score, edge_score, threshold):
    """
    Determine if a token should receive sense-locking based on:
    1. Part of speech (prioritize nouns)
    2. Polysemy score (must be above threshold)
    3. Edge score (higher edge tokens need more precise meaning)
    """
    import nltk
    try:
        from nltk.tag import pos_tag
        from nltk import word_tokenize
    except (LookupError, OSError):
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        from nltk.tag import pos_tag
        from nltk import word_tokenize
    
    # Skip short tokens and punctuation
    if len(token.strip()) <= 1 or not any(c.isalpha() for c in token):
        return False
    
    # Must be above polysemy threshold
    if poly_score < threshold:
        return False
        
    # Get part of speech
    try:
        pos = pos_tag([token.lower().strip()])[0][1]
        
        # Prioritize nouns (NN, NNS, NNP, NNPS)
        if pos.startswith('NN'):
            # For nouns, check WordNet for multiple senses
            wn = _lazy_wordnet()
            senses = wn.synsets(token.strip())
            
            # Only apply to nouns with 2+ senses
            if len(senses) >= 2:
                # For high edge-score nouns, we definitely want sense-locking
                if edge_score >= EDGE_TAU:
                    return True
                    
                # For lower edge score nouns, still sense-lock if very ambiguous
                return poly_score >= threshold * 1.2  # 20% higher bar
                
        # Skip verbs, adjectives, and other parts of speech
        return False
    except Exception:
        # If POS tagging fails, fall back to just the polysemy threshold
        return poly_score >= threshold

# --------------------------- Streamlit GUI ------------------

col_prompt, col_opts = st.columns([3,1])

with col_prompt:
    user_prompt = st.text_area("Enter your prompt …", height=200)

# Update the UI section to show top edge and polysemy words

with col_opts:
    st.markdown("### 🔧 Constraint phrases (one per line)")
    st.markdown("Constraints define the semantic boundaries of your prompt. They help identify which tokens are most important for maintaining these constraints. Enter terms that define what your prompt should be about")
    raw_constraints = st.text_area("constraints", value="context: prompt engineering\nformat: Text\nstyle: strict", height=150)
    constraints = [c.strip() for c in raw_constraints.splitlines() if c.strip()]

# Add this function to compute adaptive thresholds based on percentiles
def compute_adaptive_thresholds(edge_vals, poly_vals):
    """
    Compute adaptive thresholds based on the distribution of values in the current prompt.
    Returns percentile-based thresholds rather than fixed constants.
    """
    import numpy as np
    
    # Sort the values for percentile computation
    sorted_edge = sorted(edge_vals)
    sorted_poly = sorted(poly_vals)
    
    # Use the 70th percentile for edge threshold (adjustable)
    edge_percentile = 70
    edge_threshold = np.percentile(sorted_edge, edge_percentile) if sorted_edge else EDGE_TAU
    
    # Use the 75th percentile for polysemy threshold (adjustable)
    poly_percentile = 75
    poly_threshold = np.percentile(sorted_poly, poly_percentile) if sorted_poly else POLY_STRESS_TAU
    
    # Use the 80th percentile for sense-locking threshold (slightly more selective)
    sense_lock_percentile = 80
    sense_lock_threshold = np.percentile(sorted_poly, sense_lock_percentile) if sorted_poly else POLY_STRESS_TAU * 1.2
    
    return edge_threshold, poly_threshold, sense_lock_threshold

# Modify the analysis button handler to include adaptive thresholds
if st.button("▶ Analyse") and user_prompt.strip():
    token_ids = ENC.encode(user_prompt, disallowed_special=())
    tokens    = [ENC.decode([tid]) for tid in token_ids]

    c_vecs   = embed(constraints)
    edge_vals = edge_scores(tokens, c_vecs)
    poly_vals = poly_stress(tokens)
    
    # Compute adaptive thresholds based on the distribution
    adaptive_edge_tau, adaptive_poly_tau, adaptive_sense_lock_tau = compute_adaptive_thresholds(edge_vals, poly_vals)
    
    # Store both the fixed and adaptive thresholds
    st.session_state.update(
        tokens=tokens, 
        edge_vals=edge_vals, 
        poly_vals=poly_vals,
        adaptive_edge_tau=adaptive_edge_tau,
        adaptive_poly_tau=adaptive_poly_tau,
        adaptive_sense_lock_tau=adaptive_sense_lock_tau
    )

# Then modify the threshold controls to use adaptive thresholds as defaults
if "tokens" in st.session_state:
    tokens    = st.session_state["tokens"]
    edge_vals = st.session_state["edge_vals"]
    poly_vals = st.session_state["poly_vals"]
    
    # Get adaptive thresholds from session state
    adaptive_edge_tau = st.session_state.get("adaptive_edge_tau", EDGE_TAU)
    adaptive_poly_tau = st.session_state.get("adaptive_poly_tau", POLY_STRESS_TAU)
    adaptive_sense_lock_tau = st.session_state.get("adaptive_sense_lock_tau", POLY_STRESS_TAU * 1.2)

    # Store original values for use in the top words summary
    orig_edge_vals = edge_vals.copy()
    orig_poly_vals = poly_vals.copy()
    
    # MOVE THRESHOLD CONTROLS TO THE TOP
    st.sidebar.markdown("## Threshold Controls")
    
    # Add option to use adaptive thresholds
    use_adaptive_thresholds = st.sidebar.checkbox(
        "Use adaptive thresholds", 
        value=True, 
        key="use_adaptive_thresholds",
        help="Automatically set thresholds based on the distribution of values in your prompt"
    )
    
    # Display the actual values being used if adaptive is enabled
    if use_adaptive_thresholds:
        st.sidebar.info(f"""
        **Adaptive thresholds for this prompt:**
        - Edge: {adaptive_edge_tau:.2f} (70th percentile)
        - Polysemy: {adaptive_poly_tau:.2f} (75th percentile)
        - Sense-lock: {adaptive_sense_lock_tau:.2f} (80th percentile)
        """)
        
    # Default values for sliders - use adaptive thresholds if enabled
    default_poly_threshold = int(adaptive_poly_tau * 100) if use_adaptive_thresholds else int(POLY_STRESS_TAU * 100)
    default_edge_threshold = int(adaptive_edge_tau * 100) if use_adaptive_thresholds else int(EDGE_TAU * 100)
    default_sense_lock_threshold = int(adaptive_sense_lock_tau * 100) if use_adaptive_thresholds else int(POLY_STRESS_TAU * 120)

    # Polysemy threshold slider - updated to use adaptive defaults
    poly_threshold_pct = st.sidebar.slider(
        "Polysemy threshold (σ)",
        min_value=0,
        max_value=100,
        value=default_poly_threshold,
        step=1,
        format="%d%%",
        key="poly_threshold_pct",
        help="Threshold for considering a word to have high polysemy (higher = fewer words flagged)"
    )
    
    # Convert percentage to decimal
    poly_threshold = poly_threshold_pct / 100.0  # Store in a separate variable

    # Edge threshold slider - updated to use adaptive defaults
    edge_threshold_pct = st.sidebar.slider(
        "Edge score threshold",
        min_value=0,
        max_value=100,
        value=default_edge_threshold,
        step=1,
        format="%d%%",
        key="edge_threshold_pct",
        help="Threshold for considering a token as an edge token (higher = fewer words flagged)"
    )
    edge_threshold = edge_threshold_pct / 100.0  # Store in a separate variable

    # Sense-locking threshold slider (now uses adaptive thresholds)
    sense_lock_threshold_pct = st.sidebar.slider(
        "Sense-locking threshold",
        min_value=0,
        max_value=100,
        value=default_sense_lock_threshold,
        step=1,
        format="%d%%",
        key="sense_lock_threshold_pct",
        help="Threshold for suggesting sense-locking on words (higher = fewer words flagged)"
    )
    sense_lock_threshold = sense_lock_threshold_pct / 100.0

    # Store thresholds in session state for persistence
    st.session_state.edge_threshold = edge_threshold
    st.session_state.poly_threshold = poly_threshold
    st.session_state.sense_lock_threshold = sense_lock_threshold

    # Update masks based on the new thresholds - this is the critical fix
    edge_mask = [v >= edge_threshold for v in orig_edge_vals]  # Use threshold variable directly
    poly_mask = [v >= poly_threshold for v in orig_poly_vals]  # Use threshold variable directly

    # NOW continue with the rest of the sidebar controls
    st.sidebar.markdown("## Heat‑map toggles")

    # Edge finder with gain control
    col_edge, col_edge_gain = st.sidebar.columns([2, 3])
    with col_edge:
        show_edge = st.checkbox("Edge‑Finder", 
                               value=True, 
                               key="show_edge",
                               help="Highlights tokens that define the semantic boundaries of your prompt")
    with col_edge_gain:
        edge_gain = st.slider("Gain", 
                            min_value=1.0,  # Changed from 0.0 to 1.0
                            max_value=100.0, 
                            value=41.0, 
                            step=1.0, 
                            disabled=not show_edge, 
                            key="edge_gain",
                            help="Increases highlighting intensity for better visibility of edge tokens")
        # Convert percentage to decimal for calculations
        edge_gain = edge_gain / 30.0

    # Polysemy with gain control
    col_poly, col_poly_gain = st.sidebar.columns([2, 3])
    with col_poly:
        show_poly = st.checkbox("Polysemy stress", 
                               value=True, 
                               key="show_poly",
                               help="Highlights tokens with high semantic ambiguity")
    with col_poly_gain:
        poly_gain = st.slider("Gain", 
                             min_value=1.0,  # Changed from 0.0 to 1.0
                             max_value=100.0, 
                             value=16.0, 
                             step=1.0, 
                             disabled=not show_poly, 
                             key="poly_gain",
                             help="Increases highlighting intensity for better visibility of polysemous tokens")
        # Convert percentage to decimal for calculations
        poly_gain = poly_gain / 30.0

    # Normalize toggle with unique key
    normalize = st.sidebar.checkbox("Normalize (maximize differences)", 
                                   value=True, 
                                   key="normalize",
                                   help="Adjusts the highlighting to emphasize the relative differences between tokens")

    # Remove the expanded Visualization Controls section since we now have tooltips
    # with st.sidebar.expander("Visualization Controls", expanded=True):
    #     st.markdown("""
    #     - **Gain**: Increases highlighting intensity for better visibility
    #     - **Normalize**: Rescales values to maximize contrast between tokens
    #     """)

    # Apply normalization if requested - FIXED implementation
    edge_vals_display = edge_vals.copy()  # Create display copies that will be modified
    poly_vals_display = poly_vals.copy()
    
    # Apply threshold masking to display values - this will zero out values below threshold
    if show_edge:
        for i, (val, masked) in enumerate(zip(edge_vals_display, edge_mask)):
            if not masked:  # If below threshold
                edge_vals_display[i] = 0.0  # Zero out the display value
    
    if show_poly:
        for i, (val, masked) in enumerate(zip(poly_vals_display, poly_mask)):
            if not masked:  # If below threshold
                poly_vals_display[i] = 0.0  # Zero out the display value
    
    if normalize:
        if show_edge and any(edge_vals_display):
            # Scale edge values to maximize differences
            max_edge = max(edge_vals_display)
            min_edge = min(edge_vals_display)
            if max_edge > min_edge:  # Avoid division by zero
                edge_vals_display = [(e - min_edge) / (max_edge - min_edge) for e in edge_vals_display]
        
        if show_poly and any(poly_vals_display):
            # Scale poly values to maximize differences
            max_poly = max(poly_vals_display)
            min_poly = min(poly_vals_display)
            if max_poly > min_poly:  # Avoid division by zero
                poly_vals_display = [(p - min_poly) / (max_poly - min_poly) for p in poly_vals_display]

    # Add separate sense-locking threshold with its own slider
    st.sidebar.markdown("## Sense-locking Controls")
    
    # REMOVE the duplicate slider that's causing the error
    # Instead, just use the value that was already set earlier
    # sense_lock_threshold_pct = st.sidebar.slider(...)

    # Just keep the checkbox for toggling sense-lock display
    show_sense_lock = st.sidebar.checkbox("Show sense-locking suggestions", value=True, key="show_sense_lock")

    # Add legend for the sense-locking feature in a minimized expander
    if show_sense_lock:
        with st.sidebar.expander("Sense-locking Legend", expanded=False):
            st.markdown(f"""
            🔒 - High polysemy word (σ ≥ {sense_lock_threshold:.2f})
            
            **Recommendation:** Follow these words with:
            - A micro-definition in parentheses
            - A synonym or role marker
            
            **Example:** "Bank {{financial institution}} ledger..."
            
            This reduces ambiguity by focusing the LLM's attention on the intended meaning.
            """)

    total_poly = sum(orig_poly_vals)  # Use original values for sum
    st.sidebar.markdown(f"### Polysemy budget\nTotal σ = **{total_poly:.2f}** (τ={POLY_STRESS_TAU:.2f})")
    
    # Create a container for results that only shows after analysis
    if "tokens" in st.session_state:
        
        # Create a single row with columns for both results
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Heat‑map section
            st.markdown("### Heat‑map")
            
            # Add toggle for word grouping
            group_words = st.checkbox("Group tokens into words", value=False, 
                                     help="Display heat map by words rather than individual tokens",
                                     key="group_words")
            
            if group_words:
                # Create word groups
                word_groups = create_word_groups(tokens)
                
                # Style each word group as a unit
                html_words = []
                for start_idx, end_idx, word, is_whitespace in word_groups:
                    if is_whitespace:
                        # Just add whitespace without styling
                        html_words.append(html.escape(word))
                    else:
                        # Style the word based on average scores
                        styled_word = style_word_group(word, 
                              edge_vals_display, poly_vals_display, 
                              start_idx, end_idx,
                              show_edge, show_poly, 
                              edge_gain, poly_gain, 
                              show_sense_lock,
                              sense_lock_threshold)  # Use the separate threshold
                        html_words.append(styled_word)

                st.markdown("<div style='font-family:monospace; line-height:2em'>"+"".join(html_words)+"</div>", unsafe_allow_html=True)
            else:
                # Original token-by-token display
                html_tokens = [style_token(t, ed, pd, show_edge, show_poly, edge_gain, poly_gain, False, show_sense_lock) 
                              for t, ed, pd in zip(tokens, edge_vals_display, poly_vals_display)]
                st.markdown("<div style='font-family:monospace; line-height:2em'>"+"".join(html_tokens)+"</div>", unsafe_allow_html=True)

            # Optimized prompt output
            st.markdown("### 🔗 Optimized prompt (copy‑ready)")
            st.markdown("""
            Fill in the `{definition}` placeholders with clarifying information for ambiguous terms.
            Example: "bank {financial institution}" or "bank {river embankment}"
            """)
            optimized = contractor(tokens, edge_mask, poly_mask, orig_poly_vals, orig_edge_vals)  # Add edge_vals parameter
            st.code(optimized, language="markdown")
            if st.button("📋 Copy optimized prompt", key="copy_btn_1"):
                if copy_to_clipboard(optimized):
                    st.success("Copied to clipboard! (On Streamlit Cloud, use the text above)")
                else:
                    st.error("Copy failed.")

        # Fix the edge and polysemy word displays in the right column
        with col2:
            st.markdown("### 🎯 Top Words Summary")
            
            # Create tabs for edge and poly words
            tab_edges, tab_poly = st.tabs(["Top Edge Words", "Top Polysemy Words"])
            
            # EDGE WORDS TAB
            with tab_edges:
                st.markdown(f"#### Top edge words (τ={EDGE_TAU:.2f})")
                
                # Create pairs of (token, edge_score) and sort by score
                edge_pairs = [(t, e) for t, e in zip(tokens, orig_edge_vals)]
                edge_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # Take top 10 or fewer if not enough
                top_edges = edge_pairs[:10]
                
                # Display as a table
                if top_edges:
                    st.markdown("**Words with highest constraint relevance:**")
                    for i, (token, score) in enumerate(top_edges):
                        # Color formatting based on threshold
                        if score >= EDGE_TAU:
                            token_html = f"<span style='color:blue; font-weight:bold'>{html.escape(token)}</span>"
                        else:
                            token_html = html.escape(token)
                        
                        st.markdown(f"{i+1}. {token_html} ({score:.2f})", unsafe_allow_html=True)
                else:
                    st.info("No edge words found.")
            
            # POLYSEMY WORDS TAB
            with tab_poly:
                st.markdown(f"#### Top polysemy words (σ={POLY_STRESS_TAU:.2f})")
                
                # Create pairs of (token, poly_score) and sort by score
                poly_pairs = [(t, p) for t, p in zip(tokens, orig_poly_vals)]
                poly_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # Take top 10 or fewer if not enough
                top_poly = poly_pairs[:10]
                
                # Display as a table
                if top_poly:
                    st.markdown("**Words with highest ambiguity:**")
                    for i, (token, score) in enumerate(top_poly):
                        # Color formatting based on threshold
                        if score >= POLY_STRESS_TAU:
                            token_html = f"<span style='color:green; font-weight:bold'>{html.escape(token)} 🔒</span>"
                        else:
                            token_html = html.escape(token)
                        
                        st.markdown(f"{i+1}. {token_html} ({score:.2f})", unsafe_allow_html=True)
                else:
                    st.info("No polysemy words found.")
        
        # Add visual separator here
        st.markdown("---")

        # Optimized prompt output with engineering principles
        st.markdown("### 🔗 Engineering-Optimized Prompt")

        # Generate the enhanced prompt
        optimized, total_poly, poly_budget_exceeded = enhanced_contractor(tokens, edge_vals, poly_vals, edge_mask, poly_mask)

        # Calculate metrics
        edge_count = len([e for e in edge_mask if e])
        poly_count = len([p for p in poly_mask if p])

        # Display polysemy budget status
        poly_budget_status = "✅ GOOD" if total_poly < 5.0 else "⚠️ HIGH" if total_poly < 10.0 else "❌ EXCESSIVE"

        st.markdown(f"""
        #### Prompt Engineering Metrics
        - **Polysemy Budget:** {total_poly:.2f} ({poly_budget_status})
        - **Edge Coverage:** {edge_count} critical constraint tokens
        - **Shape Stability:** {"High" if total_poly < 5.0 else "Medium" if total_poly < 10.0 else "Low"}
        """)

        # Show warning if polysemy budget is exceeded
        if poly_budget_exceeded:
            st.warning("""
            ⚠️ **High polysemy budget exceeded!**
            
            Your prompt contains too many ambiguous terms, which increases the risk of semantic drift.
            Consider:
            1. Adding more definitions to high-polysemy words
            2. Simplifying language to reduce total ambiguity
            3. Breaking complex tasks into separate, more focused prompts
            """)

        # Show the optimized prompt
        st.code(optimized, language="markdown")

        # Update the explanation of the engineering principles to align with the changes:
        st.markdown("""
        #### Applied Engineering Principles:

        1. **Sense-locking:** Words marked with `{definition}` need disambiguation
           - Add a brief definition, synonym, or role: "bank {financial institution}"
           - This collapses ambiguity and narrows the semantic space

        2. **Edge reinforcement:** Critical constraints highlighted using natural language
           - Key constraints are presented at the beginning for context
           - Important constraints are repeated at the end for reinforcement
           - Natural language framing improves readability and LLM understanding

        3. **Dimensional dropout:** Low-information modifiers removed
           - Vague intensifiers like "very", "quite" add variance without precision
           - Removal makes output more deterministic and focused

        4. **Polysemy budget:** Total ambiguity score calculated
           - Keep the total polysemy under control for reliable results
           - Add more definitions if the budget is exceeded
        """)

        if st.button("📋 Copy engineering-optimized prompt", key="copy_btn_2"):
            if copy_to_clipboard(optimized):
                st.success("Copied to clipboard! (On Streamlit Cloud, use the text above)")
            else:
                st.error("Copy failed.")

        # Add a reference to the engineering principles
        with st.expander("Learn more about the engineering approach"):
            st.markdown("""
            This optimization is based on the "Idea Shapes for Prompt Engineering" framework, which treats prompts as geometric shapes in vector space:
            
            - The **shape** of your prompt determines which outputs are possible
            - **Edge tokens** are the boundaries that constrain the idea
            - **Polysemy** (word ambiguity) creates dimensional variance
            - **Shape stability** measures how well-constrained your prompt is
            
            By applying engineering principles like sense-locking and edge reinforcement, you create more robust prompts with predictable outputs.
            """)
    else:
        st.info("Enter a prompt and press **Analyse**.")
