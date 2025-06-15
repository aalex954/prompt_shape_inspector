#!/usr/bin/env python3
# ------------------------------------------------------------
#  LLM Prompt Shape Inspector - Production Testing Suite
# ------------------------------------------------------------
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import nltk
import streamlit as st
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dotenv import load_dotenv
import toml

# Try to load from .env file first
load_dotenv()

# Get API key from either Streamlit secrets or environment variable
def get_openai_api_key():
    # First try Streamlit secrets
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (KeyError, AttributeError, RuntimeError):
        # Fall back to environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in Streamlit secrets or environment variables")
        return api_key

# Add this function at the top of your file, before any imports from app.py
def load_streamlit_secrets():
    """Load API key from .streamlit/secrets.toml file"""
    import os
    
    # Path to the secrets.toml file
    secrets_path = os.path.join(os.path.dirname(__file__), '.streamlit', 'secrets.toml')
    
    if os.path.exists(secrets_path):
        secrets = toml.load(secrets_path)
        # Set the API key as an environment variable so app.py can find it
        os.environ["OPENAI_API_KEY"] = secrets.get("OPENAI_API_KEY", "")
        return True
    return False

# Call this function before importing from app.py
load_streamlit_secrets()

# Import the core functions from app.py
import sys
sys.path.append('.')
from app import embed, edge_scores, poly_stress, contractor, enhanced_contractor, EMBED_MODEL, POLY_STRESS_TAU, EDGE_TAU

# Initialize OpenAI client with API key from secrets or environment
try:
    client = OpenAI(api_key=get_openai_api_key())
except (RuntimeError, ValueError) as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Make sure your API key is set in .streamlit/secrets.toml or as an environment variable")
    sys.exit(1)

# Test case scenarios
test_scenarios = [
    {
        "name": "Legal Document Summarization",
        "prompt": "Summarize the following contract in plain English, highlighting key obligations and deadlines.",
        "constraints": ["context: legal document", "format: summary", "style: plain English", "focus: obligations and deadlines"],
        "ground_truth": "A concise summary of contractual obligations and deadlines, written in plain language without legalese.",
        "evaluation_metrics": ["relevance", "simplicity", "comprehensiveness"]
    },
    {
        "name": "Technical Instructions",
        "prompt": "Write step-by-step instructions for setting up a secure server on AWS, including firewall configuration.",
        "constraints": ["context: AWS security", "format: step-by-step guide", "style: technical", "focus: server security"],
        "ground_truth": "A detailed technical guide with clear security-focused steps for AWS server setup with proper firewall configuration.",
        "evaluation_metrics": ["accuracy", "completeness", "technical_correctness"]
    },
    {
        "name": "Medical Information",
        "prompt": "Explain how vaccines work in simple terms that a patient can understand. Include how they trigger an immune response.",
        "constraints": ["context: medical information", "format: explanation", "style: simple", "focus: vaccine mechanism"],
        "ground_truth": "A clear, non-technical explanation of how vaccines function to stimulate the immune system, suitable for patient education.",
        "evaluation_metrics": ["clarity", "medical_accuracy", "simplicity"]
    },
    {
        "name": "Marketing Copy",
        "prompt": "Create marketing copy for a premium organic coffee brand that emphasizes sustainability and flavor.",
        "constraints": ["context: marketing", "product: coffee", "style: premium", "focus: sustainability and flavor"],
        "ground_truth": "Compelling marketing text that highlights both the sustainable practices and exceptional taste of the organic coffee.",
        "evaluation_metrics": ["persuasiveness", "brand_alignment", "creativity"]
    },
    {
        "name": "Policy Explanation",
        "prompt": "Explain the GDPR data protection regulations and what they mean for small business owners.",
        "constraints": ["context: data regulations", "format: explanation", "style: business-friendly", "focus: GDPR compliance"],
        "ground_truth": "A business-oriented explanation of GDPR that clarifies compliance requirements specifically for small business contexts.",
        "evaluation_metrics": ["regulatory_accuracy", "relevance", "actionability"]
    }
]

# Integration test helper functions
def tokenize_prompt(prompt):
    """Tokenize a prompt using the same encoder as the embedding model."""
    import tiktoken
    enc = tiktoken.encoding_for_model(EMBED_MODEL)
    token_ids = enc.encode(prompt, disallowed_special=())
    tokens = [enc.decode([tid]) for tid in token_ids]
    return tokens

def generate_completion(prompt, model="gpt-4o"):
    """Generate a completion from the OpenAI API."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful and precise assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating completion: {e}")
        return None

def fill_in_definitions(prompt_with_placeholders):
    """Use LLM to intelligently fill in definition placeholders."""
    system_prompt = """
    You are an expert at making prompts more specific and less ambiguous.
    You will be given a prompt with {definition} placeholders.
    Replace each placeholder with a brief, specific definition or clarification that eliminates ambiguity.
    Only output the final prompt with placeholders filled in.
    """
    
    user_prompt = f"""
    Fill in all the definition placeholders in this prompt with specific, brief clarifications:
    {prompt_with_placeholders}
    
    Output only the prompt with placeholders filled in, nothing else.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error filling definitions: {e}")
        # Fallback: replace placeholders with generic definitions
        return prompt_with_placeholders.replace("{definition}", "(specifically defined)")

def evaluate_response(response, scenario, original_embedding=None):
    """Evaluate a model response against various metrics."""
    results = {}
    
    # 1. Measure how well the response addresses each constraint
    constraint_embeddings = embed(scenario["constraints"])
    response_embedding = embed(response)
    constraint_scores = []
    
    for constraint, constraint_emb in zip(scenario["constraints"], constraint_embeddings):
        if isinstance(constraint_emb, list):
            constraint_emb = constraint_emb[0]
        if isinstance(response_embedding, list):
            response_embedding = response_embedding[0]
        
        sim = cosine_similarity([constraint_emb], [response_embedding])[0][0]
        constraint_scores.append((constraint, float(sim)))
    
    avg_constraint_score = np.mean([score for _, score in constraint_scores])
    results["constraint_adherence"] = avg_constraint_score
    
    # 2. Compare to ground truth using semantic similarity
    ground_truth_emb = embed(scenario["ground_truth"])
    if isinstance(ground_truth_emb, list):
        ground_truth_emb = ground_truth_emb[0]
    if isinstance(response_embedding, list):
        response_embedding = response_embedding[0]
    
    ground_truth_sim = float(cosine_similarity([ground_truth_emb], [response_embedding])[0][0])
    results["ground_truth_similarity"] = ground_truth_sim
    
    # 3. Lexical similarity metrics (ROUGE)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(scenario["ground_truth"], response)
    results["rouge1"] = rouge_scores["rouge1"].fmeasure
    results["rouge2"] = rouge_scores["rouge2"].fmeasure
    results["rougeL"] = rouge_scores["rougeL"].fmeasure
    
    # 4. Compute semantic drift from original prompt (if provided)
    if original_embedding is not None:
        if isinstance(original_embedding, list):
            original_embedding = original_embedding[0]
        drift = float(1.0 - cosine_similarity([original_embedding], [response_embedding])[0][0])
        results["semantic_drift"] = drift
    
    return results

def run_integration_tests():
    """Run comprehensive tests comparing original and optimized prompts."""
    print("Starting LLM Prompt Shape Inspector Integration Tests")
    print("====================================================")
    results = []
    
    for scenario in tqdm(test_scenarios, desc="Testing scenarios"):
        print(f"\nTesting scenario: {scenario['name']}")
        
        # Original prompt setup
        original_prompt = scenario["prompt"]
        tokens = tokenize_prompt(original_prompt)
        constraint_vecs = embed(scenario["constraints"])
        edge_vals = edge_scores(tokens, constraint_vecs)
        poly_vals = poly_stress(tokens)
        
        # Create masks based on thresholds
        edge_mask = [v >= EDGE_TAU for v in edge_vals]
        poly_mask = [v >= POLY_STRESS_TAU for v in poly_vals]
        
        # Generate optimized prompts
        basic_optimized = contractor(tokens, edge_mask, poly_mask, poly_vals)
        enhanced_optimized, total_poly, _ = enhanced_contractor(tokens, edge_vals, poly_vals, edge_mask, poly_mask)
        
        # Fill in the definition placeholders
        filled_basic = fill_in_definitions(basic_optimized)
        filled_enhanced = fill_in_definitions(enhanced_optimized)
        
        # Store prompts for reporting
        scenario_results = {
            "scenario": scenario["name"],
            "original_prompt": original_prompt,
            "optimized_prompt": filled_basic,
            "enhanced_prompt": filled_enhanced,
            "total_polysemy": total_poly,
            "edge_tokens": sum(edge_mask),
            "polysemy_tokens": sum(poly_mask)
        }
        
        # Get embeddings for the original prompt for drift calculation
        original_embedding = embed(original_prompt)
        
        print("\n1. Testing Original Prompt:")
        print("--------------------------")
        print(original_prompt)
        original_response = generate_completion(original_prompt)
        if original_response:
            original_metrics = evaluate_response(original_response, scenario, original_embedding)
            scenario_results["original_metrics"] = original_metrics
            print(f"Constraint adherence: {original_metrics['constraint_adherence']:.4f}")
        
        print("\n2. Testing Basic Optimized Prompt:")
        print("--------------------------------")
        print(filled_basic)
        basic_response = generate_completion(filled_basic)
        if basic_response:
            basic_metrics = evaluate_response(basic_response, scenario, original_embedding)
            scenario_results["basic_metrics"] = basic_metrics
            print(f"Constraint adherence: {basic_metrics['constraint_adherence']:.4f}")
        
        print("\n3. Testing Enhanced Optimized Prompt:")
        print("-----------------------------------")
        print(filled_enhanced)
        enhanced_response = generate_completion(filled_enhanced)
        if enhanced_response:
            enhanced_metrics = evaluate_response(enhanced_response, scenario, original_embedding)
            scenario_results["enhanced_metrics"] = enhanced_metrics
            print(f"Constraint adherence: {enhanced_metrics['constraint_adherence']:.4f}")
        
        # Store responses
        scenario_results["original_response"] = original_response
        scenario_results["basic_response"] = basic_response
        scenario_results["enhanced_response"] = enhanced_response
        
        # Calculate improvements
        if original_response and basic_response and enhanced_response:
            constraint_improvement_basic = ((basic_metrics["constraint_adherence"] / original_metrics["constraint_adherence"]) - 1) * 100
            constraint_improvement_enhanced = ((enhanced_metrics["constraint_adherence"] / original_metrics["constraint_adherence"]) - 1) * 100
            
            ground_truth_improvement_basic = ((basic_metrics["ground_truth_similarity"] / original_metrics["ground_truth_similarity"]) - 1) * 100
            ground_truth_improvement_enhanced = ((enhanced_metrics["ground_truth_similarity"] / original_metrics["ground_truth_similarity"]) - 1) * 100
            
            scenario_results["constraint_improvement_basic"] = constraint_improvement_basic
            scenario_results["constraint_improvement_enhanced"] = constraint_improvement_enhanced
            scenario_results["ground_truth_improvement_basic"] = ground_truth_improvement_basic
            scenario_results["ground_truth_improvement_enhanced"] = ground_truth_improvement_enhanced
            
            print(f"\nConstraint adherence improvement (basic): {constraint_improvement_basic:.2f}%")
            print(f"Constraint adherence improvement (enhanced): {constraint_improvement_enhanced:.2f}%")
            print(f"Ground truth similarity improvement (basic): {ground_truth_improvement_basic:.2f}%")
            print(f"Ground truth similarity improvement (enhanced): {ground_truth_improvement_enhanced:.2f}%")
        
        results.append(scenario_results)
        
        # Add a short delay to avoid rate limits
        time.sleep(2)
    
    return results

def generate_report(results):
    """Generate a detailed report of the test results."""
    # Create a timestamp for the report
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create a folder for the reports if it doesn't exist
    os.makedirs("test_reports", exist_ok=True)
    
    # Save the raw results as JSON
    with open(f"test_reports/test_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Create a DataFrame for numeric analysis
    metrics_data = []
    for result in results:
        if all(k in result for k in ["original_metrics", "basic_metrics", "enhanced_metrics"]):
            metrics_data.append({
                "scenario": result["scenario"],
                "original_constraint": result["original_metrics"]["constraint_adherence"],
                "basic_constraint": result["basic_metrics"]["constraint_adherence"],
                "enhanced_constraint": result["enhanced_metrics"]["constraint_adherence"],
                "original_ground_truth": result["original_metrics"]["ground_truth_similarity"],
                "basic_ground_truth": result["basic_metrics"]["ground_truth_similarity"],
                "enhanced_ground_truth": result["enhanced_metrics"]["ground_truth_similarity"],
                "original_rouge1": result["original_metrics"]["rouge1"],
                "basic_rouge1": result["basic_metrics"]["rouge1"],
                "enhanced_rouge1": result["enhanced_metrics"]["rouge1"],
                "original_rouge2": result["original_metrics"]["rouge2"],
                "basic_rouge2": result["basic_metrics"]["rouge2"],
                "enhanced_rouge2": result["enhanced_metrics"]["rouge2"],
                "original_rougeL": result["original_metrics"]["rougeL"],
                "basic_rougeL": result["basic_metrics"]["rougeL"],
                "enhanced_rougeL": result["enhanced_metrics"]["rougeL"],
                "total_polysemy": result["total_polysemy"],
                "edge_tokens": result["edge_tokens"],
                "polysemy_tokens": result["polysemy_tokens"]
            })
    
    if not metrics_data:
        print("No complete metric data available for reporting.")
        return
    
    df = pd.DataFrame(metrics_data)
    
    # Calculate improvement percentages
    for idx, row in df.iterrows():
        # Handle division by zero for constraint scores
        if row["original_constraint"] > 0:
            df.loc[idx, "constraint_improvement_basic"] = ((row["basic_constraint"] / row["original_constraint"]) - 1) * 100
            df.loc[idx, "constraint_improvement_enhanced"] = ((row["enhanced_constraint"] / row["original_constraint"]) - 1) * 100
        else:
            df.loc[idx, "constraint_improvement_basic"] = 0 if row["basic_constraint"] == 0 else 100
            df.loc[idx, "constraint_improvement_enhanced"] = 0 if row["enhanced_constraint"] == 0 else 100
        
        # Handle division by zero for ground truth scores
        if row["original_ground_truth"] > 0:
            df.loc[idx, "ground_truth_improvement_basic"] = ((row["basic_ground_truth"] / row["original_ground_truth"]) - 1) * 100
            df.loc[idx, "ground_truth_improvement_enhanced"] = ((row["enhanced_ground_truth"] / row["original_ground_truth"]) - 1) * 100
        else:
            df.loc[idx, "ground_truth_improvement_basic"] = 0 if row["basic_ground_truth"] == 0 else 100
            df.loc[idx, "ground_truth_improvement_enhanced"] = 0 if row["enhanced_ground_truth"] == 0 else 100
    
    # Save the metrics DataFrame to CSV
    df.to_csv(f"test_reports/metrics_{timestamp}.csv", index=False)
    
    # Create a detailed HTML report
    html_report = f"""
    <html>
    <head>
        <title>LLM Prompt Shape Inspector - Test Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; margin-top: 30px; }}
            h3 {{ color: #2980b9; }}
            .metric {{ font-weight: bold; }}
            .improvement {{ color: green; }}
            .decline {{ color: red; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .prompt {{ background-color: #f9f9f9; padding: 10px; border-radius: 5px; white-space: pre-wrap; }}
            .summary {{ background-color: #e8f4f8; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>LLM Prompt Shape Inspector - Integration Test Report</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <p>Average constraint adherence improvement:</p>
            <ul>
                <li>Basic optimization: {df["constraint_improvement_basic"].mean():.2f}%</li>
                <li>Enhanced optimization: {df["constraint_improvement_enhanced"].mean():.2f}%</li>
            </ul>
            <p>Average ground truth similarity improvement:</p>
            <ul>
                <li>Basic optimization: {df["ground_truth_improvement_basic"].mean():.2f}%</li>
                <li>Enhanced optimization: {df["ground_truth_improvement_enhanced"].mean():.2f}%</li>
            </ul>
        </div>
        
        <h2>Detailed Results by Scenario</h2>
    """
    
    for result in results:
        scenario_name = result["scenario"]
        html_report += f"""
        <h3>{scenario_name}</h3>
        
        <h4>Prompts</h4>
        <p><strong>Original Prompt:</strong></p>
        <div class="prompt">{result["original_prompt"]}</div>
        
        <p><strong>Basic Optimized Prompt:</strong></p>
        <div class="prompt">{result["optimized_prompt"]}</div>
        
        <p><strong>Enhanced Optimized Prompt:</strong></p>
        <div class="prompt">{result["enhanced_prompt"]}</div>
        
        <h4>Metrics</h4>
        <table>
            <tr>
                <th>Metric</th>
                <th>Original</th>
                <th>Basic Optimized</th>
                <th>Enhanced Optimized</th>
                <th>Basic Improvement</th>
                <th>Enhanced Improvement</th>
            </tr>
        """
        
        if all(k in result for k in ["original_metrics", "basic_metrics", "enhanced_metrics"]):
            for metric in ["constraint_adherence", "ground_truth_similarity", "rouge1", "rouge2", "rougeL"]:
                orig_val = result["original_metrics"][metric]
                basic_val = result["basic_metrics"][metric]
                enhanced_val = result["enhanced_metrics"][metric]
                
                # Handle division by zero cases
                if orig_val > 0:
                    basic_improvement = ((basic_val / orig_val) - 1) * 100
                    enhanced_improvement = ((enhanced_val / orig_val) - 1) * 100
                else:
                    # If original value is zero, we can't calculate percentage improvement
                    # Instead, use absolute improvement
                    basic_improvement = 0 if basic_val == 0 else 100  # 0→0 is 0% change, 0→anything else is ∞% increase
                    enhanced_improvement = 0 if enhanced_val == 0 else 100
                
                basic_class = "improvement" if basic_improvement > 0 else "decline"
                enhanced_class = "improvement" if enhanced_improvement > 0 else "decline"
                
                html_report += f"""
                <tr>
                    <td>{metric.replace('_', ' ').title()}</td>
                    <td>{orig_val:.4f}</td>
                    <td>{basic_val:.4f}</td>
                    <td>{enhanced_val:.4f}</td>
                    <td class="{basic_class}">{basic_improvement:+.2f}%</td>
                    <td class="{enhanced_class}">{enhanced_improvement:+.2f}%</td>
                </tr>
                """
        
        html_report += """
        </table>
        
        <h4>Responses</h4>
        """
        
        if "original_response" in result and result["original_response"]:
            html_report += f"""
            <p><strong>Original Response:</strong></p>
            <div class="prompt">{result["original_response"]}</div>
            """
        
        if "basic_response" in result and result["basic_response"]:
            html_report += f"""
            <p><strong>Basic Optimized Response:</strong></p>
            <div class="prompt">{result["basic_response"]}</div>
            """
        
        if "enhanced_response" in result and result["enhanced_response"]:
            html_report += f"""
            <p><strong>Enhanced Optimized Response:</strong></p>
            <div class="prompt">{result["enhanced_response"]}</div>
            """
        
        html_report += "<hr>"
    
    # Add visualizations
    html_report += """
        <h2>Visualizations</h2>
    """
    
    # Plot the improvements
    plt.figure(figsize=(12, 6))
    improvement_data = df[["scenario", "constraint_improvement_basic", "constraint_improvement_enhanced", 
                          "ground_truth_improvement_basic", "ground_truth_improvement_enhanced"]]
    improvement_data_melted = pd.melt(
        improvement_data, 
        id_vars=["scenario"],
        var_name="metric", 
        value_name="improvement_percent"
    )
    
    sns.barplot(x="scenario", y="improvement_percent", hue="metric", data=improvement_data_melted)
    plt.title("Percentage Improvement by Scenario and Metric")
    plt.ylabel("Improvement (%)")
    plt.xlabel("Scenario")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"test_reports/improvement_chart_{timestamp}.png")
    
    html_report += f"""
        <img src="improvement_chart_{timestamp}.png" alt="Improvement Chart" style="width: 100%; max-width: 900px;">
    """
    
    # Plot correlation between polysemy and improvement
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="total_polysemy", y="constraint_improvement_enhanced", data=df)
    plt.title("Correlation: Total Polysemy vs. Enhanced Constraint Improvement")
    plt.xlabel("Total Polysemy")
    plt.ylabel("Constraint Improvement (%)")
    plt.savefig(f"test_reports/polysemy_correlation_{timestamp}.png")
    
    html_report += f"""
        <img src="polysemy_correlation_{timestamp}.png" alt="Polysemy Correlation" style="width: 100%; max-width: 900px;">
    """
    
    html_report += """
    </body>
    </html>
    """
    
    # Save the HTML report
    with open(f"test_reports/report_{timestamp}.html", "w") as f:
        f.write(html_report)
    
    print(f"\nTest report generated: test_reports/report_{timestamp}.html")
    return df

def main():
    # Download NLTK resources if not already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Run the tests
    results = run_integration_tests()
    
    # Generate and display the report
    metrics_df = generate_report(results)
    
    # Print summary statistics
    print("\n==== SUMMARY STATISTICS ====")
    print(f"Number of scenarios tested: {len(results)}")
    
    if metrics_df is not None:
        print("\nAverage improvements:")
        print(f"Basic optimization constraint adherence: {metrics_df['constraint_improvement_basic'].mean():.2f}%")
        print(f"Enhanced optimization constraint adherence: {metrics_df['constraint_improvement_enhanced'].mean():.2f}%")
        print(f"Basic optimization ground truth: {metrics_df['ground_truth_improvement_basic'].mean():.2f}%")
        print(f"Enhanced optimization ground truth: {metrics_df['ground_truth_improvement_enhanced'].mean():.2f}%")
    
    print("\nTest suite completed.")

if __name__ == "__main__":
    main()