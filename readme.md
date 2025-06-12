# LLM Prompt Shape Inspector

## Overview

The LLM Prompt Shape Inspector is an advanced prompt engineering tool that helps you analyze and optimize prompts for large language models. Using principles from mechanical engineering and vector space semantics, it treats prompts as geometric shapes with precise boundaries, enabling you to build more reliable, deterministic prompts.

The tool provides three key features:
- **Edge-Finder**: Identifies the most crucial constraint-enforcing tokens in your prompt
- **Polysemy Stress**: Detects ambiguous words that may cause semantic drift
- **Contractor**: Generates an optimized version of your prompt with sense-locking and edge reinforcement

## Quick Start

1. Enter your prompt in the main text area
2. Add constraint phrases that define the semantic boundaries of your prompt
3. Click "Analyse" to process your prompt
4. Use the heat map to identify important edge tokens (blue) and ambiguous polysemy words (green)
5. Copy the optimized prompt and fill in any definition placeholders

## Features

### Heat Map Visualization

The heat map displays your prompt with color-coding to highlight:
- **Blue**: Edge tokens that strongly constrain your prompt's meaning
- **Green**: Words with high polysemy (ambiguity) that may cause semantic drift
- **🔒**: Lock icons indicating words that should be sense-locked with definitions

Controls:
- **Group tokens into words**: View heat map by whole words rather than individual tokens
- **Edge-Finder**: Toggle edge token highlighting
- **Polysemy stress**: Toggle ambiguous word highlighting
- **Gain sliders**: Adjust visualization intensity
- **Normalize**: Maximize differences between tokens for clearer visualization

### Top Words Summary

Displays the tokens with highest edge scores and polysemy values:
- **Top Edge Words**: Tokens most critical for maintaining your prompt's constraints
- **Top Polysemy Words**: The most ambiguous words that may benefit from sense-locking

### Engineering-Optimized Prompt

Produces an enhanced version of your prompt using four engineering principles:
1. **Sense-locking**: Adds `{definition}` placeholders after ambiguous words
2. **Edge reinforcement**: Marks critical tokens with `*` and repeats key constraints
3. **Dimensional dropout**: Removes low-information modifiers that add variance
4. **Polysemy budget**: Tracks total ambiguity to keep prompts deterministic

## Theoretical Framework: Ideas as Shapes in Vector Space

### Core Concepts

The app is built on a geometric understanding of prompts:

| Engineering analogue | NLP counterpart | Why it matters |
|----------------------|-----------------|----------------|
| **Design envelope** for an aircraft | **Concept envelope** of an idea | The aircraft must never leave its aerodynamic envelope; a prompt should stay inside its intended semantic envelope. |
| **Finite-element mesh** with nodes & boundary conditions | **Embedding cloud** with *edge tokens* & constraints | The mesh nodes with the highest strain are the places where failure begins; the tokens that most strongly constrain meaning ("edge tokens") are where an LLM will "tear" into ambiguity if you're not explicit. |
| **Tolerance stack-up** in manufacturing | **Polysemy stack-up** in a prompt | Ambiguous words add *dimensional variance*. Past a threshold the stack-up makes the output drift off-spec. |

### Engineering Playbook for Robust Prompts

| Technique | Mechanism | Why it works |
|-----------|-----------|--------------|
| **Sense-locking** | Immediately follow any high-σ word with a micro-definition, synonym, or role marker. <br>"Bank *(financial institution)* ledger compliance report …" | Reduces σ(t) by collapsing the LLM's attention onto the desired centroid; narrows 𝒮 early in the forward pass. |
| **Edge reinforcement** | Repeat or paraphrase the most critical constraints at least twice, once near the front and once near the end (positional bias). | Ensures high-e(t) tokens dominate global attention heads even if the middle expands creatively. |
| **Dimensional dropout** | Deliberately omit low-information adjectives/adverbs that merely add orthogonal variance. | Shrinks 𝒮 volume, making generation more deterministic and cheaper to steer during an agent loop. |
| **Polysemy budget** | Compute a quick heuristic: total σ ≤ σ_max (tunable). Flag the user or auto-expand definitions when the budget is exceeded. | Gives a measurable "tolerance stack-up" limit analogous to mechanical design. |

### Mathematical Model

1. **Token embeddings → point cloud**
   Each token *t* is a vector **v**ᵗ ∈ ℝᴰ.

2. **Idea = bounded region 𝒮**
   𝒮 = {**x** | g_i(**x**) ≤ 0, i=1..k}
   – where each *gᵢ* encodes a linguistic or factual constraint.

3. **Edge nodes**
   Define *edge score* e(t) = max_j |g_j(**v**^t)|
   Tokens with high *e(t)* exert the strongest bounding force.

4. **Polysemy stress tensor**
   For token *t*, let {**v**ᵗ¹, **v**ᵗ², … **v**ᵗᴺ} be the sense-cluster centroids.
   The *polysemy stress* is σ(t) = var({**v**^t_s}).

## Usage Tips

### Constraint Phrases

- Add 2-5 constraint phrases that define what your prompt should be about
- Use the format "category: value" (e.g., "domain: finance", "tone: professional")
- These help identify which tokens are most important for maintaining your constraints

### Working with the Heat Map

- Use the heat map to identify your prompt's most critical tokens (blue)
- Pay special attention to words with high polysemy (green with 🔒)
- Adjust thresholds to see more or fewer highlighted tokens

### Optimizing Your Prompt

1. Replace `{definition}` placeholders with clarifying information:
   - "Bank {financial institution}" instead of just "Bank"
   - "Cell {biology}" instead of just "Cell"

2. Pay attention to tokens marked with `*`:
   - These are critical constraint tokens
   - Consider emphasizing or repeating them in your prompt

3. Monitor your polysemy budget:
   - If it's "HIGH" or "EXCESSIVE", add more definitions
   - Break complex prompts into smaller, more focused steps

## Common Issues and Solutions

| Symptom | Likely geometric cause | Fast fix |
|---------|------------------------|----------|
| Output veers into unintended domain | 𝒮 not convex; missing edge tokens on that side | Add concrete examples that live on the missing face of the hull. |
| Repetition / looping | Hull too thin in some dimensions; model stuck in local minimum | Introduce orthogonal but relevant descriptors to widen 𝒮 slightly. |
| Hallucinated entities | Internal walk escaped the hull via ambiguous connector word | Replace connector with a univocal relationship phrase or split the prompt into two stages. |

## Key Takeaways

1. **Prompt = CAD drawing.** You're specifying tolerances; sloppiness propagates downstream.
2. **Polysemy is the enemy of repeatability.** Measure it, budget it, neutralize it.
3. **Edge tokens are your screws & welds.** Double-secure the ones that matter.
4. **Agent loops need hull tracking.** Treat every generation as a potential simulation step.
5. **Metrics enable automation.** Computing hull volume, polysemy stress, and edge coverage enables building linters and self-healing agents.

## Requirements

- Python 3.7+
- OpenAI API key (set in environment variable `OPENAI_API_KEY`)
- Required packages: streamlit, openai, numpy, tiktoken, pyperclip, nltk

## Running the App

```
streamlit run app.py
```

---

By instrumenting your prompts with polysemy-stress meters and edge-coverage heat maps, you'll convert the abstract philosophy of "the shape of ideas" into a concrete engineering control system that makes your LLM workflows steadier, safer, and more scalable.