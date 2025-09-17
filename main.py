import sys
import builtins
import warnings
import os
import torch
import argparse
from transformers.utils import logging as hf_logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from modules.utils import set_seed


def setup_debug_printing(debug_enabled: bool):
    """Redirect DEBUG prints to console or log file depending on --debug flag."""
    original_print = builtins.print

    def debug_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        if msg.startswith("DEBUG"):
            if debug_enabled:
                original_print(*args, **kwargs)
            else:
                with open("debug.log", "a") as f:
                    f.write(msg + "\n")
        else:
            original_print(*args, **kwargs)

    builtins.print = debug_print


def setup_warnings_and_logging():
    """Silence unnecessary warnings from libraries."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    hf_logging.set_verbosity_error()


def initialize_environment(debug: bool, seed: int = 42):
    """Set up debug printing, logging, and random seed."""
    setup_debug_printing(debug)
    setup_warnings_and_logging()
    set_seed(seed)


# crude detection of --debug before arg parsing
DEBUG_FLAG = "--debug" in sys.argv

# initialize at import time
initialize_environment(DEBUG_FLAG, seed=42)


# Argument parser
parser = argparse.ArgumentParser(description="Process scenarios with different models and GPU settings.")
parser.add_argument("--model", type=str, required=True, help="Model name from Hugging Face")
parser.add_argument("--gpu", type=str, default="0", help="GPU device(s) to use: '0' (default), '1', or '0,1' for multi-GPU")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., DiscrimEval, Folktexts, etc.)")
parser.add_argument("--n", type=int, default=None, help="Number of examples to process.")
parser.add_argument("--debug", action="store_true", help="Enable debug output")
parser.add_argument("--temperature", type=float, nargs="+", default=[0.0], help="List of temperatures to try")
parser.add_argument("--sample", action="store_true", help="Optional: sample and save from HF datasets")
parser.add_argument("--prompt", type=str, required=True, help="'Rational_based' or 'Unconstraint' or 'CoT' prompting")
parser.add_argument("--truncation", action="store_true", help="Enable truncation when tokenizing")
parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization")
parser.add_argument("--max_new_tokens", type=int, default=500, help="Maximum number of new tokens to generate")
parser.add_argument("--k", type=int, default=2, help="Number of clusters for KMeans")
parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
args = parser.parse_args()

# re-initialize with actual CLI seed
initialize_environment(args.debug, seed=args.seed)

# Imports after setting CUDA
if args.prompt in ["Unconstraint", "CoT"]: 
    from modules.conversation_unconstraint import process_scenarios_with_questions
else:
    from modules.conversation_rational_based import process_scenarios_with_questions

from modules.io_utils import (
    load_prompt_templates,
    load_inputs,
    infer_file_paths_from_dataset
)
from modules.complement import get_complement_function

# Optional: sample from HF datasets if requested
if args.sample:
    from modules import sampling
    sample_fn = {
        "discrimeval": sampling.sample_discrimeval,
        "sst2": sampling.sample_sst2,
        "twitter": sampling.sample_twitter,
        "folktexts": sampling.sample_folktexts,
        "gsm8k": sampling.sample_gsm8k,
        "mgnli": sampling.sample_mgnli,
    }.get(args.dataset.lower())
    if sample_fn:
        sample_fn()
    else:
        print(f"No sampling function for {args.dataset}")

# Infer input file paths
paths = infer_file_paths_from_dataset(args.dataset, args.prompt)
scenario_file = paths["scenario_file"]
question_file = paths["question_file"]
prompt_file = paths["prompt_file"]

# Load input files
scenarios, questions, metadata = load_inputs(
    scenario_file=scenario_file,
    question_file=question_file,
    metadata_file=None
)

# Load prompt templates
templates = load_prompt_templates(prompt_file)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

gpu_list = [int(x) for x in args.gpu.split(",")]
if len(gpu_list) == 1:
    device = torch.device(f"cuda:{gpu_list[0]}")
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
else:
    # multi-GPU sharding
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")

# Complement function
complement_fn = get_complement_function(args.dataset)

# Run experiments across temperatures
for temp in args.temperature:
    print(f"Processing with temperature {temp}")
    kwargs = dict(
        scenarios=scenarios,
        questions=questions,
        metadata=metadata,
        model_name=args.model,
        dataset_name=args.dataset,
        temperature=temp,
        templates=templates,
        tokenizer=tokenizer,
        model=model,
        N=args.n,
        complement_fn=complement_fn,
        debug=args.debug,
        prompt=args.prompt,
        truncation=args.truncation,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )

    if args.prompt == "Unconstraint":
        kwargs["k"] = args.k   # only pass k for Unconstraint

    process_scenarios_with_questions(**kwargs)
