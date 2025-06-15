# Prompt Shape Inspector Testing Suite

## Overview

The Prompt Shape Inspector Testing Suite is a robust evaluation framework designed to quantitatively measure the effectiveness of the Prompt Shape Inspector tool across various real-world scenarios. It compares original prompts against optimized versions to demonstrate measurable improvements in output quality, constraint adherence, and semantic precision.

## Features

- **Multi-domain testing**: Evaluates prompts across legal, technical, medical, marketing, and policy domains
- **Comprehensive metrics**: Measures constraint adherence, semantic similarity, lexical precision (ROUGE), and semantic drift
- **Three-way comparison**: Tests original prompts against both basic and enhanced optimizations
- **Automated placeholder resolution**: Intelligently fills in definition placeholders
- **Rich visualization**: Generates charts showing performance improvements across scenarios
- **Detailed HTML reports**: Creates comprehensive test reports with side-by-side comparisons

## Requirements

```
pip install openai numpy pandas matplotlib seaborn nltk rouge-score scikit-learn tqdm python-dotenv
```

You'll also need:
- An OpenAI API key (stored in the `OPENAI_API_KEY` environment variable)
- The Prompt Shape Inspector application (app.py) in the same directory

## Running the Tests

1. **Setup your environment**:
   ```bash
   # Create a .env file with your OpenAI API key
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

2. **Run the test suite**:
   ```bash
   python tests.py
   ```

3. **Wait for completion**:
   The test will process each scenario sequentially, showing a progress bar and live results. Testing all scenarios typically takes 5-10 minutes depending on API response times.

## Understanding the Results

### Console Output

The console will display:
- Progress through each scenario with real-time metrics
- Constraint adherence scores for each prompt version
- Percentage improvements for each optimization type
- Summary statistics showing average improvements across all scenarios

### HTML Report

A detailed HTML report is automatically generated in the `test_reports/` directory with:

- **Executive Summary**: Average improvements across metrics
- **Per-scenario details**:
  - Original, basic, and enhanced prompts
  - Complete metrics table with improvement percentages
  - Model responses for all three prompt versions
- **Visualizations**:
  - Bar chart showing percentage improvements by scenario and metric
  - Scatter plot showing the correlation between polysemy and improvements

### CSV and JSON Data

In addition to the HTML report, the following files are generated:
- `test_results_[timestamp].json`: Complete raw test data
- `metrics_[timestamp].csv`: Tabular data for further analysis

## Key Metrics

| Metric | Description | How to interpret |
|--------|-------------|------------------|
| **Constraint Adherence** | How well does the response follow the intended constraints? | Higher is better. Enhanced optimization should show the highest scores. |
| **Ground Truth Similarity** | How semantically close is the response to the expected output? | Higher is better. Look for consistent improvements in optimized prompts. |
| **ROUGE Scores** | Lexical overlap between response and ground truth | Higher indicates better lexical precision. ROUGE-L is especially important. |
| **Semantic Drift** | How far the response strays from the original intent | Lower is better. A small amount of drift is normal and may actually be beneficial. |

## Interpreting Improvement Percentages

- **0-5%**: Minor improvement, might not be consistently noticeable
- **5-15%**: Solid improvement, likely to be noticeable in most applications
- **15-30%**: Major improvement, significant positive impact on output quality
- **>30%**: Exceptional improvement, transformative effect on response quality

## Use Cases

1. **Demonstration of Value**: Show stakeholders the quantifiable benefits of using Prompt Shape Inspector
2. **Regression Testing**: Track performance of prompt optimizations over time
3. **Parameter Tuning**: Experiment with different threshold values to find optimal settings
4. **Domain Comparison**: Identify which types of content benefit most from optimization

## Troubleshooting

- **API Rate Limits**: The test includes delays between API calls, but you may need to adjust the `time.sleep()` value if you encounter rate limiting
- **Missing Metrics**: If the report shows missing metrics, check your API key and ensure the Prompt Shape Inspector core functions are working correctly
- **Visualization Errors**: If charts aren't generating, ensure matplotlib and seaborn are properly installed

## Extending the Tests

- Add your own scenarios to the `test_scenarios` list
- Modify the `evaluate_response()` function to add custom metrics
- Adjust the `generate_completion()` parameters to test different model settings

## License

This testing suite is released under the same license as the Prompt Shape Inspector tool.

---

By running these comprehensive tests, you can quantitatively demonstrate how the Prompt Shape Inspector improves prompt quality and output reliability across a variety of real-world scenarios.