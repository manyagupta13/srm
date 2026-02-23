# collect_GPQA_unify.py

import os
import sys
import json
import logging
import argparse
import random
import hashlib
from datetime import datetime
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import modal
from tqdm import tqdm
from datasets import load_dataset
import re

app = modal.App("gpqa-inference-v2")
image = (modal.Image.debian_slim()
    .pip_install(["torch>=2.0.0", "transformers>=4.30.0", "accelerate", "datasets", "tqdm"]))

def extract_answer_gpqa(text: str, valid_choices: List[str]) -> str:
    """
    A more robust answer extractor for GPQA multiple-choice questions.
    It tries several common formats in order of priority, returning the uppercase letter upon the first match, otherwise returns an empty string.
    """
    valid = {c[:1].upper() for c in valid_choices}

    # 1) "Answer: X" - most reliable
    m = re.search(r'answer[^A-Za-z0-9]*([A-D])', text, flags=re.I)
    if m:
        ans = m.group(1).upper()
        if ans in valid:
            return ans

    # 2) Wrapped in double hashes like "## ... X ##"
    m = re.search(r'##\s*([A-D])\s*##', text)
    if m:
        ans = m.group(1).upper()
        if ans in valid:
            return ans

    # 3) A single letter on its own line
    for line in text.splitlines():
        s = line.strip().upper()
        if s in valid and len(s) == 1:
            return s

    # 4) Parentheses or at the end like "...(X)" or " X)" (takes the last one in reverse order)
    for m in reversed(re.findall(r'\(([A-D])\)', text)):
        ans = m.upper()
        if ans in valid:
            return ans

    # 5) Fallback to the original "space + letter" strategy (still takes the last one)
    for m in reversed(re.findall(r'\s([A-D])', text)):
        ans = m.upper()
        if ans in valid:
            return ans

    m = re.search(r"Final Answer: \(([A-Z])\)", text)
    if m:
        ans = m.group(1).upper()
        if ans in valid:
            return ans
    return ""

@app.function(image=image, gpu="A100-40GB", timeout=600)
def run_inference(model_name: str, messages, max_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0, top_k: int = 1, decode_seed: int = 42):
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

# ===================== Inline Prompt Templates (Editable) =====================
# Edit these three strings when doing prompt tuning. They will also be saved to JSON.
SYSTEM_PROMPT = (
    "You are a PhD student in science. "
    "Reason step by step through the following question and provide the best answer. "
    "Review what you have done and make sure you have not made any mistakes. "
    "Give your single-letter choice as '##Answer: X##'."
)

# Will be formatted with {question} and {choices}
USER_PROMPT_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Choices:\n{choices}\n"
)

# Appended to the user message to enforce answer format compatible with extractor
FORMAT_PROMPT = (
    "\nPlease provide your final single-letter answer in the format: ##Answer: X##."
)

def _build_messages_from_templates(question_text: str, choices: List[str]) -> List[Dict[str, str]]:
    choices_text = "\n".join(choices)
    user_content = USER_PROMPT_TEMPLATE.format(question=question_text, choices=choices_text) + FORMAT_PROMPT
    messages: List[Dict[str, str]] = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_content})
    return messages

def parse_arguments():
    parser = argparse.ArgumentParser(description='GPQA benchmark using Modal GPU (A100) inference.')
    parser.add_argument('--model_list', type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help='Comma-separated list of HuggingFace model IDs')
    parser.add_argument('--sample_size', type=int, default=-1,
                        help='Number of questions to sample from GPQA dataset. -1 for all.')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Nucleus sampling p')
    parser.add_argument('--top_k', type=int, default=1,
                        help='Top-k sampling')
    parser.add_argument('--decode_seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Max tokens in each generation')
    parser.add_argument('--num_threads', type=int, default=5,
                        help='Concurrent threads for processing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--dataset_path', type=str, default="data/gpqa_shuffled.json",
                        help='Path to the GPQA dataset JSON file')
    parser.add_argument('--output_dir', type=str, default="./GPQA_response",
                        help='Directory to save benchmark results')
    return parser.parse_args()

def main():
    args = parse_arguments()
    random.seed(args.seed)

    # Create output directory (parent directory + subdirectory for this run)
    os.makedirs(args.output_dir, exist_ok=True)

    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prompt_hash_src = (SYSTEM_PROMPT or "") + (USER_PROMPT_TEMPLATE or "") + (FORMAT_PROMPT or "")
    prompt_hash = hashlib.sha256(prompt_hash_src.encode('utf-8')).hexdigest()[:8]
    run_dir = os.path.join(args.output_dir, f"GPQA_{run_timestamp}_p{prompt_hash}")
    os.makedirs(run_dir, exist_ok=True)

    # Parse model list
    model_list = [m.strip() for m in args.model_list.split(",")]

    # Load GPQA dataset
    try:
        dataset = load_dataset("json", data_files=args.dataset_path, split="train")
        data_list = list(dataset)
    except Exception as e:
        logger.error(f"Failed to load dataset from {args.dataset_path}: {e}")
        sys.exit(1)

    # Sampling
    if 0 < args.sample_size < len(data_list):
        data_list = random.sample(data_list, args.sample_size)

    # Write the settings for this run to a metadata file for easy comparison of different prompts
    run_meta = {
        "timestamp": run_timestamp,
        "dataset_path": args.dataset_path,
        "model_list": model_list,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "sample_size_requested": args.sample_size,
        "num_threads": args.num_threads,
        "seed": args.seed,
        "provider": "modal-gpu-a100",
        "prompts": {
            "system": SYSTEM_PROMPT,
            "user_template": USER_PROMPT_TEMPLATE,
            "format": FORMAT_PROMPT,
        },
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    # Process models one by one (output to a separate folder for this run)
    for model in model_list:
        logger.info(f"Processing model: {model} (via Modal GPU A100)")

        results_for_this_model = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        def process_one(item, _model=model):
            question_text = item["question"]
            choices = item["choices"]
            correct_answer = item.get("answer", "")

            messages = _build_messages_from_templates(question_text, choices)

            try:
                resp = run_inference.remote(
                    model_name=_model,
                    messages=messages,
                    max_tokens=args.max_new_tokens if args.max_new_tokens > 0 else 512,
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

            extracted_answer = extract_answer_gpqa(raw_output, choices)
            is_correct = (extracted_answer == correct_answer)

            system_prompt = ""
            user_prompt = ""
            for m in messages:
                if m.get("role") == "system" and not system_prompt:
                    system_prompt = m.get("content", "")
                elif m.get("role") == "user" and not user_prompt:
                    user_prompt = m.get("content", "")
            format_prompt = FORMAT_PROMPT

            return {
                "model": _model,
                "question": question_text,
                "choices": choices,
                "correct_answer": correct_answer,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "format_prompt": format_prompt,
                "model_response": raw_output,
                "reasoning_content": "",
                "extracted_answer": extracted_answer,
                "is_correct": is_correct,
                "token_usage": usage_details
            }

        # Multi-threaded concurrency
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = [executor.submit(process_one, item) for item in data_list]
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Collecting responses for {model}", unit="q"):
                results_for_this_model.append(f.result())

        # Calculate accuracy
        correct_count = sum(1 for r in results_for_this_model if r["is_correct"])
        total_count = len(results_for_this_model)
        accuracy = correct_count / total_count if total_count else 0.0

        # Calculate token usage
        for r in results_for_this_model:
            token_usage = r["token_usage"]
            total_prompt_tokens += token_usage.get("prompt_tokens", 0)
            total_completion_tokens += token_usage.get("completion_tokens", 0)
            total_tokens += token_usage.get("total_tokens", 0)

        # Save results to the subdirectory for this run; the filename no longer includes a timestamp for easy comparison
        out_file = f"{run_dir}/{model.replace('/', '_')}.json"

        output_data = {
            "model": model,
            "timestamp": run_timestamp,
            "provider": "modal-gpu-a100",
            "prompts": {
                "system": SYSTEM_PROMPT,
                "user_template": USER_PROMPT_TEMPLATE,
                "format": FORMAT_PROMPT,
            },
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "sample_size": total_count,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "token_usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens
            },
            "responses": results_for_this_model
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"[*] Model={model} (Modal GPU)  Accuracy={accuracy:.2%} "
              f"({correct_count}/{total_count}), Tokens: Prompt={total_prompt_tokens}, "
              f"Completion={total_completion_tokens}, Total={total_tokens}, saved to: {out_file}")

    logger.info("All models processed. Finished.")


@app.local_entrypoint()
def run(
    model_list: str = "Qwen/Qwen2.5-7B-Instruct",
    sample_size: int = -1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
    decode_seed: int = 42,
    max_new_tokens: int = 512,
    num_threads: int = 5,
    seed: int = 42,
    dataset_path: str = "data/gpqa_shuffled.json",
    output_dir: str = "./GPQA_response",
):
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
