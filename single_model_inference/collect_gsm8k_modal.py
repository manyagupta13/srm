#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
collect_gsm8k_modal.py

GSM8K benchmark evaluation using Modal GPU (A100) inference.
Evaluates LLMs on math reasoning tasks.
- Uses local GSM8K JSON dataset
- Prompts model to reason step by step and output final answer prefixed with `####`
- Extracts answers by parsing substring after last `####`
"""

import os
import sys
import json
import logging
import argparse
import random
import hashlib
import re
from datetime import datetime
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import modal
from tqdm import tqdm
from datasets import load_dataset

# ===== CHANGE THIS TO SWITCH MODELS =====
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
# DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# =========================================

app = modal.App("gsm8k-inference-v3-cached")
image = (modal.Image.debian_slim()
    .pip_install(["torch>=2.0.0", "transformers>=4.30.0", "accelerate", "datasets", "tqdm", "sentencepiece"]))

def extract_answer_gsm(text: str) -> str:
    """Extract the answer following the GSM convention `#### <answer>`."""
    if text is None:
        return ""
    matches = list(re.finditer(r"####\s*([-]?\d+(?:\.\d+)?)", text))
    if matches:
        return matches[-1].group(1)
    nums = re.findall(r"[-]?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""

def normalize_answer(ans: str) -> str:
    """Normalize answer by stripping spaces, commas, and leading zeros."""
    if ans is None:
        return ""
    ans = ans.strip()
    if ans.startswith("####"):
        ans = ans[4:].strip()
    ans = ans.replace(",", "")
    if re.fullmatch(r"0+", ans):
        return "0"
    ans = re.sub(r"^0+", "", ans)
    return ans

def create_chat_prompt(system_msg: str, user_msg: str):
    """Returns a chat structure: [system role, user role]"""
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# ===== MODEL CACHE CLASS =====
# Loads models once and keeps them in GPU memory for all subsequent inference calls
@app.cls(image=image, gpu="A100-40GB", timeout=600)
class InferenceEngine:
    def __init__(self):
        """Load and cache all models on initialization"""
        self.models = {}
        self.tokenizers = {}
        self.model_names = ["Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]
    
    def load_models(self):
        """Load all models into GPU memory - called once"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[*] Loading {len(self.model_names)} models on device {device}...")
        
        for model_name in self.model_names:
            if model_name in self.models:
                print(f"    ✓ {model_name} already loaded")
                continue
            
            print(f"    - Loading {model_name}...")
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
            self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            )
            print(f"    ✓ {model_name} ready")
        
        print(f"[✓] All {len(self.model_names)} models cached in GPU memory")
    
    def infer(self, model_name: str, messages: list, max_tokens: int = 512, 
              temperature: float = 0.0, top_p: float = 1.0, top_k: int = 1):
        """Run inference using cached model"""
        import torch
        
        if model_name not in self.models or not self.models[model_name]:
            raise ValueError(f"Model {model_name} not loaded. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Build prompt text
        text = ""
        for msg in messages:
            if msg.get("role") == "system":
                text += f"{msg.get('content', '')}\n\n"
            else:
                text += msg.get("content", "")
        
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            gen_kwargs = {"max_new_tokens": max_tokens, "top_p": top_p, "top_k": top_k}
            if temperature > 0.0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["do_sample"] = True
            else:
                gen_kwargs["do_sample"] = False
            outputs = model.generate(**inputs, **gen_kwargs)
        
        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return {"content": generated_text, "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


# Create singleton instance with both models preloaded
@app.function(image=image, gpu="A100-40GB", timeout=600)
def run_inference(model_name: str, messages, max_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0, top_k: int = 1, decode_seed: int = 42):
    """Legacy function for backward compatibility - loads model per call (slow)"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    torch.manual_seed(decode_seed)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    text = ""
    for msg in messages:
        if msg.get("role") == "system":
            text += f"{msg.get('content', '')}\n\n"
        else:
            text += msg.get("content", "")
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen_kwargs = {"max_new_tokens": max_tokens, "top_p": top_p, "top_k": top_k}
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False
        outputs = model.generate(**inputs, **gen_kwargs)
    
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return {"content": generated_text, "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
DEBUG = int(os.environ.get("DEBUG", "0"))

def parse_arguments():
    parser = argparse.ArgumentParser(description='GSM8K benchmark using Modal GPU (A100) inference.')
    parser.add_argument('--model_list', type=str, default=DEFAULT_MODEL,
                        help='Comma-separated list of HuggingFace model IDs')
    parser.add_argument('--sample_size', type=int, default=-1,
                        help='Number of questions to sample from GSM8K dataset. -1 for all.')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Nucleus sampling p')
    parser.add_argument('--top_k', type=int, default=1,
                        help='Top-k sampling')
    parser.add_argument('--decode_seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--max_new_tokens', type=int, default=4096,
                        help='Max tokens in each generation')
    parser.add_argument('--num_threads', type=int, default=5,
                        help='Concurrent threads for processing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--dataset_path', type=str, default="data/gsm8k_500.json",
                        help='Path to the GSM8K dataset JSON file')
    parser.add_argument('--output_dir', type=str, default="./GSM8K_response",
                        help='Directory to save benchmark results')
    return parser.parse_args()

def main():
    args = parse_arguments()
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, f"GSM8K_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    model_list = [m.strip() for m in args.model_list.split(",")]

    # Load GSM8K dataset (from JSON file)
    try:
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset from {args.dataset_path}: {e}")
        sys.exit(1)

    if 0 < args.sample_size < len(data_list):
        data_list = random.sample(data_list, args.sample_size)

    # Write run metadata
    run_meta = {
        "timestamp": run_timestamp,
        "provider": "modal-gpu-a100",
        "dataset": "gsm8k_500",
        "sample_size": len(data_list),
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    # Process models one by one
    for model in model_list:
        logger.info(f"Processing model: {model} (via Modal GPU A100)")

        results_for_this_model = []
        total_tokens = 0

        def build_prompt(question_text: str):
            system_msg = (
                "You are a helpful math tutor. Reason step by step to solve the problem. "
                "After your reasoning, output the final answer on a new line prefixed with '####'. "
                "Example: #### 42"
            )
            user_msg = f"Problem:\n{question_text}\n\nProvide your reasoning, then the final answer."
            return create_chat_prompt(system_msg, user_msg)

        def process_one(item, _model=model):
            question_text = item["question"]
            reference_answer = normalize_answer(extract_answer_gsm(item["answer"]))

            messages = build_prompt(question_text)

            try:
                resp = run_inference.remote(
                    model_name=_model,
                    messages=messages,
                    max_tokens=args.max_new_tokens if args.max_new_tokens > 0 else 4096,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    decode_seed=args.decode_seed,
                )
            except Exception as e:
                logger.error(f"Inference error: {e}")
                resp = {"content": f"ERROR: {e}", "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

            raw_output = resp.get("content", "")
            usage_details = resp.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

            extracted_answer = normalize_answer(extract_answer_gsm(raw_output))
            is_correct = (extracted_answer == reference_answer)

            return {
                "model": _model,
                "question": question_text,
                "reference_answer": reference_answer,
                "model_response": raw_output,
                "extracted_answer": extracted_answer,
                "is_correct": is_correct,
                "token_usage": usage_details
            }

        # Multi-threaded concurrency
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = [executor.submit(process_one, item) for item in data_list]
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {model}", unit="q"):
                results_for_this_model.append(f.result())

        # Calculate accuracy
        correct_count = sum(1 for r in results_for_this_model if r["is_correct"])
        total_count = len(results_for_this_model)
        accuracy = correct_count / total_count if total_count else 0.0

        # Calculate token usage
        for r in results_for_this_model:
            token_usage = r["token_usage"]
            total_tokens += token_usage.get("total_tokens", 0)

        # Save results
        out_file = f"{run_dir}/{model.replace('/', '_')}.json"
        output_data = {
            "model": model,
            "timestamp": run_timestamp,
            "provider": "modal-gpu-a100",
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "sample_size": total_count,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_tokens": total_tokens,
            "responses": results_for_this_model
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"[*] Model={model} (Modal GPU)  Accuracy={accuracy:.2%} "
              f"({correct_count}/{total_count}), Total Tokens: {total_tokens}, saved to: {out_file}")

    logger.info("All models processed. Finished.")


def build_prompt(question_text: str):
    """Build chat messages for GSM: reason step-by-step and output final answer prefixed with ####."""
    system_msg = (
        "You are a helpful math tutor. Reason step by step to solve the problem. "
        "After your reasoning, output the final answer on a new line prefixed with '####'. "
        "Example: #### 42"
    )
    user_msg = f"Problem:\n{question_text}\n\nProvide your reasoning, then the final answer."
    
    prompt = f"{system_msg}\n\n{user_msg}"
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


@app.local_entrypoint()
def run(
    model_list: str = DEFAULT_MODEL,
    sample_size: int = -1,
    temperature: float = 0.3,
    top_p: float = 1.0,
    top_k: int = 1,
    decode_seed: int = 42,
    max_new_tokens: int = 4096,
    num_threads: int = 5,
    seed: int = 42,
    dataset_path: str = "data/gsm8k_500.json",
    output_dir: str = "./GSM8K_response",
    # Single question mode for orchestrator
    question: str = None,
    model_name: str = None,
):
    # Single question mode for orchestrator calls
    if question is not None:
        if model_name is None:
            model_name = DEFAULT_MODEL
        messages = build_prompt(question)
        result = run_inference.remote(model_name, messages, max_new_tokens, temperature)
        raw_content = result.get("content", "")
        extracted = normalize_answer(extract_answer_gsm(raw_content))
        print(f"ANSWER:{extracted}")
        return
    
    # Batch mode (original)
    import sys
    sys.argv = [
        sys.argv[0],
        "--model_list", model_list,
        "--sample_size", str(sample_size),
        "--temperature", str(temperature),
        "--top_p", str(top_p),
        "--top_k", str(top_k),
        "--decode_seed", str(decode_seed),
        "--max_new_tokens", str(max_new_tokens),
        "--num_threads", str(num_threads),
        "--seed", str(seed),
        "--dataset_path", dataset_path,
        "--output_dir", output_dir,
    ]
    main()


if __name__ == "__main__":
    main()
