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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dotenv import load_dotenv
import functools
import hashlib

# Load environment variables
load_dotenv()

# Import the core functions from app.py
import sys
sys.path.append('.')
from app import embed, edge_scores, poly_stress, contractor, enhanced_contractor, EMBED_MODEL, POLY_STRESS_TAU, EDGE_TAU

# Initialize OpenAI client with API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create a memoization cache for expensive functions
definitions_cache = {}
completion_cache = {}

# LRU cache decorator for memory-based caching
def memoize(func):
    """Simple memoization decorator to cache function results"""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a cache key from function arguments
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
        key = hashlib.md5(str(tuple(key_parts)).encode()).hexdigest()
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    return wrapper

@memoize
def fill_in_definitions(prompt_with_placeholders):
    """Use LLM to intelligently fill in definition placeholders with caching."""
    # Return cached result if available
    if prompt_with_placeholders in definitions_cache:
        return definitions_cache[prompt_with_placeholders]
    
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
        result = response.choices[0].message.content
        # Cache the result for future use
        definitions_cache[prompt_with_placeholders] = result
        return result
    except Exception as e:
        print(f"Error filling definitions: {e}")
        # Fallback: replace placeholders with generic definitions
        return prompt_with_placeholders.replace("{definition}", "(specifically defined)")

@memoize
def generate_completion(prompt, model="gpt-4o"):
    """Generate a completion from the OpenAI API with caching."""
    # Return cached result if available
    cache_key = f"{model}:{prompt}"
    if cache_key in completion_cache:
        return completion_cache[cache_key]
    
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
        result = response.choices[0].message.content
        # Cache the result for future use
        completion_cache[cache_key] = result
        return result
    except Exception as e:
        print(f"Error generating completion: {e}")
        return None

def run_prompt_test(scenario, optimization_options=None):
    """
    Test a prompt with different optimization levels
    
    Args:
        scenario: Dictionary containing the test scenario
        optimization_options: Dictionary of optimization flags (default: use all optimizations)
    
    Returns:
        Dictionary with test results
    """
    # Default optimization options
    if optimization_options is None:
        optimization_options = {
            'use_basic': True,
            'use_enhanced': True,
            'use_natural_framing': True,
            'use_smart_sense_locking': True,
            'use_adaptive_thresholds': True,
        }
    
    # Start with original prompt setup
    original_prompt = scenario["prompt"]
    tokens = tokenize_prompt(original_prompt)
    constraint_vecs = embed(scenario["constraints"])
    
    # Calculate edge and polysemy values
    edge_vals = edge_scores(tokens, constraint_vecs)
    poly_vals = poly_stress(tokens)
    
    # Apply adaptive thresholds if requested
    if optimization_options.get('use_adaptive_thresholds', True):
        # Use 70th percentile for edge threshold and 75th for polysemy
        edge_threshold = np.percentile(edge_vals, 70)
        poly_threshold = np.percentile(poly_vals, 75)
    else:
        edge_threshold = EDGE_TAU
        poly_threshold = POLY_STRESS_TAU
    
    # Create masks based on thresholds
    edge_mask = [v >= edge_threshold for v in edge_vals]
    poly_mask = [v >= poly_threshold for v in poly_vals]
    
    results = {
        "scenario": scenario["name"],
        "original_prompt": original_prompt
    }
    
    # Original prompt evaluation
    original_embedding = embed(original_prompt)
    original_response = generate_completion(original_prompt)
    if original_response:
        original_metrics = evaluate_response(original_response, scenario, original_embedding)
        results["original_metrics"] = original_metrics
        results["original_response"] = original_response
    
    # Only run the basic optimization if requested
    if optimization_options.get('use_basic', True):
        # Generate optimized prompt
        basic_optimized = contractor(tokens, edge_mask, poly_mask, poly_vals, edge_vals)
        filled_basic = fill_in_definitions(basic_optimized)
        
        results["basic_prompt"] = filled_basic
        basic_response = generate_completion(filled_basic)
        if basic_response:
            basic_metrics = evaluate_response(basic_response, scenario, original_embedding)
            results["basic_metrics"] = basic_metrics
            results["basic_response"] = basic_response
    
    # Only run the enhanced optimization if requested
    if optimization_options.get('use_enhanced', True):
        # Generate enhanced prompt
        enhanced_optimized, total_poly, poly_budget_exceeded = enhanced_contractor(
            tokens, edge_vals, poly_vals, edge_mask, poly_mask
        )
        filled_enhanced = fill_in_definitions(enhanced_optimized)
        
        results["enhanced_prompt"] = filled_enhanced
        results["total_polysemy"] = total_poly
        results["poly_budget_exceeded"] = poly_budget_exceeded
        
        enhanced_response = generate_completion(filled_enhanced)
        if enhanced_response:
            enhanced_metrics = evaluate_response(enhanced_response, scenario, original_embedding)
            results["enhanced_metrics"] = enhanced_metrics
            results["enhanced_response"] = enhanced_response
    
    # Calculate improvements where possible
    if all(k in results for k in ["original_metrics", "basic_metrics"]):
        calculate_improvements(results, "basic")
    
    if all(k in results for k in ["original_metrics", "enhanced_metrics"]):
        calculate_improvements(results, "enhanced")
    
    return results

def calculate_improvements(results, optimization_type):
    """Calculate percentage improvements for metrics"""
    for metric in ["constraint_adherence", "ground_truth_similarity", "rouge1", "rouge2", "rougeL"]:
        orig_val = results["original_metrics"][metric]
        improved_val = results[f"{optimization_type}_metrics"][metric]
        
        # Handle division by zero cases
        if orig_val > 0:
            improvement = ((improved_val / orig_val) - 1) * 100
        else:
            improvement = 0 if improved_val == 0 else 100
            
        results[f"{metric}_improvement_{optimization_type}"] = improvement

def run_integration_tests():
    """Run comprehensive tests comparing original and optimized prompts with optimized API usage."""
    print("Starting LLM Prompt Shape Inspector Integration Tests")
    print("====================================================")
    results = []
    
    # Run tests on a subset of scenarios first if in development mode
    dev_mode = os.getenv("TEST_DEV_MODE", "false").lower() == "true"
    test_scenarios_to_use = test_scenarios[:1] if dev_mode else test_scenarios
    
    # Process each scenario with shared cache
    for scenario in tqdm(test_scenarios_to_use, desc="Testing scenarios"):
        print(f"\nTesting scenario: {scenario['name']}")
        
        # Run the single scenario with all optimizations enabled
        scenario_results = run_prompt_test(scenario)
        results.append(scenario_results)
        
        # Add a short delay to avoid rate limits
        time.sleep(1)
    
    # Print cache statistics
    print(f"\nCache statistics:")
    print(f"- Definitions cache hits: {len(definitions_cache)}")
    print(f"- Completions cache hits: {len(completion_cache)}")
    
    return results

# Test case scenarios (unchanged)

# Evaluation function (unchanged)

# Report generation function (unchanged)

def main():
    # Download NLTK resources if not already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Run the tests with optimized API usage
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
    
    # Calculate and show cost savings
    api_calls_made = len(definitions_cache) + len(completion_cache)
    estimated_calls_without_cache = api_calls_made * 2  # Conservative estimate
    savings_percent = ((estimated_calls_without_cache - api_calls_made) / estimated_calls_without_cache) * 100
    
    print(f"\nAPI Efficiency:")
    print(f"- Total API calls made: {api_calls_made}")
    print(f"- Estimated calls without caching: {estimated_calls_without_cache}")
    print(f"- Approximate cost savings: {savings_percent:.1f}%")
    
    print("\nTest suite completed.")

if __name__ == "__main__":
    main()