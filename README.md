# Can LLMs Explain Themselves Counterfactually?

This repository contains the code for our paper:
**"[Can LLMs Explain Themselves Counterfactually?](https://arxiv.org/abs/2502.18156)"** \[1]

We study a specific class of self-explanations: **self-generated counterfactual explanations (SCEs)**. To evaluate them, we develop protocols that assess the ability of large language models (LLMs) to generate high-quality SCEs, analyzing performance across diverse *model families*, *sizes*, *temperature settings*, *chat histories*, *prompting strategies*, and *datasets*.

To capture failure cases, we combine human plausibility annotations with automated metrics:

* **Readability scores** → linguistic complexity
* **Cosine similarity in embedding space** → semantic drift
* **k-means clustering** → task misunderstanding

We further explore whether counterfactual reasoning emerges alongside broader model quality by correlating SCE validity with *model size*, *few-shot perplexity*, *Hugging Face leaderboard rank*, and reported *MMLU performance*.

Overall, our findings show that **despite strong reasoning abilities, modern LLMs remain far from reliable when asked to explain their own predictions counterfactually**.

## Supported benchmarks:

| Benchmark            | Paper      | Dataset Source        |
|----------------------|------------|-----------------------|
| DiscrimEval [2]      | [Link](https://arxiv.org/pdf/2312.03689)  | [Dataset (hf)](https://huggingface.co/datasets/Anthropic/discrim-eval)     |
| FolkTexts [3]        | [Link](https://arxiv.org/pdf/2407.14614)  | [Dataset (hf)](https://huggingface.co/datasets/acruz/folktexts)     |
| Twitter Financial News [4]| [Link](https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic)    |  [Dataset (hf)](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)                    |
| SST2 [5]            | [Link](https://aclanthology.org/D13-1170.pdf)   | [Dataset (hf)](https://huggingface.co/datasets/stanfordnlp/sst2)     |
| GSM8K [6]           | [Link](https://arxiv.org/pdf/2110.14168)   | [Dataset (hf)](https://huggingface.co/datasets/openai/gsm8k)     |
| MGNLI [7]           | [Link](https://aclanthology.org/N18-1101.pdf)   | [Dataset (hf)](https://huggingface.co/datasets/nyu-mll/multi_nli)     |


## Supported models:

| Family   | Model (Hugging Face) |
|----------|-----------------------|
| **Gemma-S** | [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)  |
| **Gemma-B** | [google/gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it)  |
| **Llama-S** | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| **Llama-B** | [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)|
| **Mistral-S** | [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |
|  **Mistral-B**| [mistralai/Mistral-Small-24B-Instruct-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) |
| **DeepSeek** | [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) |


## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/aisoc-lab/llm-sces.git
cd llm-sces
```

### 2. Install Dependencies
You can choose either Option A or Option B below to install the required dependencies:
#### Option A: Using Conda and requirements.txt

```bash
conda create --name SCEs python==3.12.2
conda activate SCEs
pip install -r requirements.txt
```
#### Option B: Using setup.py (Recommended for development)
```bash
pip install -e .
```

## Generating counterfactual explanations
```bash
python main.py \
--model meta-llama/Llama-3.1-8B-Instruct \
--gpu 2 \
--dataset SST2 \
--temperature 0.5 \
--n 500 \
--debug \
--prompt "Unconstraint" \
--max_length 120 \
--max_new_tokens 500 \
--truncation
```
## Post-processing and extracting model's answer
```bash
python run_postprocessing.py \
--input_dir $HOME/llm-sces/Unconstraint_meta-llama_Llama-3.1-8B-Instruct_SST2 \
--dataset SST2
```

## Testing
All tests are stored in the `tests/` folder.  

Run **all tests** with:
```bash
pytest -v
````

Run an **individual test file** (e.g., `test_complement.py`) with:

```bash
pytest -v tests/test_complement.py
```

Run a **single test function** (e.g., `test_complement_sst2_positive_to_negative`) with:

```bash
pytest -v tests/test_complement.py::test_complement_sst2_positive_to_negative
```

## Repository Structure

```bash
llm-sces/
├── main.py                     # Main entry point for generating counterfactual explanations (SCEs) across datasets
├── run_postprocessing.py       # Main entry point for post-processing tasks
├── setup.py                    # Package setup file
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment file
├── README.md                   # Project documentation
│
├── Datasets/                   # Datasets used across tasks
├── Prompts/                    # Prompt templates for different datasets
├── tests/                      # Unit tests
│
├── modules/                    # Core helper functions and task-specific logic
│   ├── case_classifier.py
│   ├── case_filtering.py
│   ├── complement.py
│   ├── conversation_unconstraint.py
│   ├── conversation_rational_based.py
│   ├── dataset_config.py
│   ├── extract_decision.py
│   ├── io_utils.py
│   ├── Mean_length.py
│   ├── metrics.py
│   ├── parsing.py
│   ├── plot_utils.py
│   ├── postprocess_utils.py
│   ├── pre_processing_input_datasets.py
│   ├── sampling.py
│   ├── utils.py
│   └── __init__.py
```

## Pipeline Overview

This project is structured into two main components:

---

## **1. Main Script – Inference / Generation**

This script runs **inference** using a specified model on a given dataset and generates **raw outputs**.

##### Expected Output

For each combination of **model** and **dataset**, the script saves a `.json` file containing the raw outputs.  
Each output file typically includes:

- **Input Fields**, which vary depending on the dataset:
  - **DiscrimEval**: `filled_template` refers to the input scenario where the original question has been intentionally removed
  - **Folktexts**: `description` (Respondent data), `question`, `choices`
  - **Twitter Financial News**: `text` (Twitter post)
  - **GSM8K**: `question` (math problem)
  - **SST2**: `sentence` (movie review)
  - **MGNLI**: `premise`, `hypothesis`

- **Generated Outputs**:
  - `Original Answer` – the model’s initial prediction or decision
  - `SCEs` – counterfactual scenario
  - `Revised Answer With History` – model’s prediction for SCEs **with** context
  - `Revised Answer Without History` – model’s prediction for SCEs **without** context
  - `Raw Edit Distance`
  - `Normalized Edit Distance Percentage`
  - `Character Overlap Percentage`
  - `Levenshtein Ratio`

---

>  These outputs form the foundation for the post-processing pipeline and further evaluation.

## 2. Post-Processing Outputs

After inference, results are passed through a post-processing pipeline which generates structured summaries and filtered subsets of the data.

---

## `summary.txt`

This file provides an overview of key statistics and information:

- **Total Cases**: Number of samples per dataset.
- **Unknown Cases**: Number of outputs where the model failed to generate a valid response — including empty outputs, invalid options, or non-numeric answers (for GSM8K).
- **Short Cases**: Generations deemed too short based on dataset-specific thresholds. Generations are considered *short* if they fall below the following word count or content criteria:

  - **DiscrimEval**  
    SCEs scenario with fewer than **15 words**.

  - **Twitter Financial News**  
    SCEs Twitter posts containing fewer than **3 words**.

  - **Folktexts**  
    SCEs respondent data shorter than **60 words**.

  - **MGNLI**  
    SCEs hypothesis with fewer than **2 words**.

  - **SST2**  
    SCEs sentences (movie review) with fewer than **1 word**.

  - **GSM8K**  
    SCEs math question with:
    - Fewer than **5 words**, **and**
    - Only **alphabetic characters** (i.e., no numbers or mathematical symbols).

- **Non-Empty Spans**: Number of cases where the model successfully generated a span — the rationale behind the `Original Answer`.

**SCE Length Statistics**  
Summarizes the length of counterfactual scenario (SCEs), broken down by:
- Match vs. Non-Match categories:
  - **Match Cases**: Cases where the revised answer (the model’s decision/prediction after applying SCEs) is the same as the original answer, indicating that the contrastive edits did **not** shift the model’s decision toward the target.
  - **Non-Match**: Cases where the revised answer **differs** from the original and matches the target answer. This indicates that the SCEs successfully flipped the model’s decision, making it a **valid** case.
- With vs. Without History context

For each group, mean, standard deviation, min, max, and median word counts are reported.

**Normalized Length Differences**  
Reports the relative change in SCE length between original and revised generations. Confidence intervals are computed using standard statistical formulas:

```
CI = mean ± 1.96 × (sd / √n)
```

  Here, *mean* is the average value, *sd* is the standard deviation, and *n* is the number of samples. 
  The factor **1.96** corresponds to a 95% confidence level under a normal distribution.

**SCE Histograms**  
Length distribution plots are saved as:
- `With_History_SCE_Lengths.pdf`
- `Without_History_SCE_Lengths.pdf`

**Classification Counts**  
Breakdown of SCE classifications into:
- Match Cases
- Non-Match Cases
- Non-Responses / Other / Unclassified Cases  
This is reported separately for both *With Context* and *Without Context* generations.

---
## Confidence Intervals (95%)

This section reports key metrics with their associated 95% confidence intervals (CIs), calculated using standard error formulas:

- **Gen**:  
  Percentage of generations containing successful SCEs  
  → Reported as: `Mean ± Margin`  
  → 95% CI: `[Lower Bound, Upper Bound]`

- **Val** (Valid SCEs without context):  
  Percentage of revised answers that flipped the original decision  
  → `Mean ± Margin`  
  → 95% CI: `[Lower Bound, Upper Bound]`

- **Val_c** (Invalid SCEs with context):  
  Percentage of revised answers that matched the original decision  
  → `Mean ± Margin`  
  → 95% CI: `[Lower Bound, Upper Bound]`

- **CI Comparison** between **Val** and **Val_c**:  
  If their confidence intervals **do not overlap**, the difference is considered **statistically significant**.
---

## Generated Files and Folders

**With History:**
- `match_cases.json`
- `non_match_cases.json`
- `non_responses.json`
- `other_cases.json`
- `unclassified_cases.json`

**Without History:**
- Same structure as above for the **Without Context** setting

Additional outputs:
- `short_cases.json`: All short generations across datasets
- `unknown_cases.json`: Invalid or missing responses
- `non_match_cases_with_history_statistics.txt`: Detailed stats for non-match cases (valid SCEs) with context
- `non_match_cases_without_history_statistics.txt`: Same for the without context setting
- `empty_spans_cases.json`: Only present for **Rationale-based prompting**

---

## Additional Statistical Outputs

1. **Flesch–Kincaid Readability (FK Scores)**  
   - Evaluates the readability level of SCEs (school grade equivalents).  
   - Outputs:  
     - `fk_with_history_statistics.txt`  
     - `fk_without_history_statistics.txt`  
   - Each file reports mean, standard deviation, 95% confidence intervals, CI overlap checks, and results of a paired permutation test comparing *With History* vs. *Without History*.

2. **Permutation Tests**  
   The pipeline conducts several statistical tests, with results saved in `summary.txt` and corresponding `*_statistics.txt` files:  
   - Validity rate differences (flipped vs. non-flipped answers)  
   - SCE length differences (valid SCEs)  
   - Edit distance differences (normalized edit distance %)  
   - Readability differences (Flesch–Kincaid scores)  

3. **Edit Distance Metrics**  
   - Extracted from the `"Normalized Edit Distance Percentage"` field.  
   - Compared between *With History* and *Without History*.  
   - Reported in the non-match statistics files, with significance testing.

4. **Confidence Interval (CI) Notes**  
   - For readability, validity, and edit distance, reports explicitly state whether confidence intervals overlap.  
   - Non-overlap indicates a statistically significant difference.  
   - Saved in both `summary.txt` and `*_statistics.txt`.

5. **Bootstrap Estimates**  
   - Normalized differences in SCE lengths are estimated via bootstrapping.  
   - Results include mean values with 95% CI bounds.  
   - Reported in `summary.txt`.
---

## References

[1] Dehghanighobadi, Z., Fischer, A., & Zafar, M. B. (2025). Can LLMs Explain Themselves Counterfactually?. arXiv preprint arXiv:2502.18156.

[2] Tamkin, A., Askell, A., Lovitt, L., Durmus, E., Joseph, N., Kravec, S., ... & Ganguli, D. (2023). Evaluating and mitigating discrimination in language model decisions. arXiv.

[3] Cruz, A. F., Hardt, M., & Mendler-Dünner, C. (2024). Evaluating language models as risk scores. arXiv preprint arXiv:2407.14614.

[4] ZeroShot. 2022. Twitter financial news dataset. https://huggingface.co/datasets/zeroshot/ twitter-financial-news-sentiment. Accessed: Feb 2025.

[5] Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A. Y., & Potts, C. (2013, October). Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on empirical methods in natural language processing (pp. 1631-1642).

[6] Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., ... & Schulman, J. (2021). Training verifiers to solve math word problems, 2021. https://arxiv.org/abs/2110.14168, 9.

[7] Williams, A., Nangia, N., & Bowman, S. R. (2017). A broad-coverage challenge corpus for sentence understanding through inference. arXiv preprint arXiv:1704.05426.
