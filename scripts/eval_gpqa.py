#!/usr/bin/env python3
"""
GPQA Benchmark Evaluation Script

Evaluates language models on the GPQA dataset (198 graduate-level science questions).

Usage:
    python scripts/eval_gpqa.py --model "mistralai/Mistral-7B-Instruct-v0.3"
    python scripts/eval_gpqa.py --model "Qwen/Qwen2.5-7B-Instruct" --dry-run

For more options, run:
    python scripts/eval_gpqa.py --help
"""

import os
import json
import argparse
import torch
import gc
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List

# Add parent directory to path for utils import
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    extract_answer_gpqa,
    load_model_and_tokenizer,
    generate_response,
    load_gpqa_dataset
)


# Prompt templates
SYSTEM_PROMPT = (
    "You are a PhD student in science. Reason step by step through the following question "
    "and provide the best answer. Review what you have done and make sure you have not made "
    "any mistakes. Give your single-letter choice as '##Answer: X##'."
)

USER_PROMPT_TEMPLATE = "Question:\n{question}\n\nChoices:\n{choices}\n"
FORMAT_PROMPT = "\nPlease provide your final single-letter answer in the format: ##Answer: X##."


def create_messages(question: str, choices: List[str]) -> List[Dict[str, str]]:
    """Create chat messages for the model."""
    choices_text = "\n".join(choices)
    user_content = USER_PROMPT_TEMPLATE.format(
        question=question,
        choices=choices_text
    ) + FORMAT_PROMPT
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


def evaluate(args):
    """Run evaluation on GPQA dataset."""
    
    # Setup
    print("="*60)
    print("GPQA BENCHMARK EVALUATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Mode: {'DRY RUN (5 samples)' if args.dry_run else 'FULL EVALUATION (198 samples)'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print("="*60 + "\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_gpqa_dataset()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Determine number of samples
    num_samples = 5 if args.dry_run else len(dataset)
    
    # Run evaluation
    results = []
    correct = 0
    
    print(f"\nEvaluating {num_samples} samples...\n")
    
    for idx in tqdm(range(num_samples), desc="Evaluating"):
        item = dataset[idx]
        
        # Generate prediction
        messages = create_messages(item["question"], item["choices"])
        response = generate_response(
            model, tokenizer, messages,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        # Extract and check answer
        extracted = extract_answer_gpqa(response, item["choices"])
        is_correct = (extracted == item["answer"])
        
        if is_correct:
            correct += 1
        
        # Store result
        result = {
            'index': idx,
            'question': item["question"],
            'choices': item["choices"],
            'correct_answer': item["answer"],
            'model_response': response,
            'extracted_answer': extracted,
            'is_correct': is_correct
        }
        results.append(result)
        
        # Print progress in dry run mode
        if args.dry_run:
            status = '✓' if is_correct else '✗'
            print(f"Q{idx+1}: Predicted={extracted}, Correct={item['answer']}, {status}")
        
        # Periodic GPU cache clearing
        if (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    # Calculate final accuracy
    accuracy = correct / num_samples if num_samples > 0 else 0.0
    
    # Print results
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Accuracy: {accuracy:.2%} ({correct}/{num_samples})")
    print("="*60 + "\n")
    
    # Save results
    save_results(args, results, accuracy, correct)


def save_results(args, results: List[Dict], accuracy: float, correct_count: int):
    """Save evaluation results to disk."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = args.model.replace('/', '_')
    mode = "dryrun" if args.dry_run else "full"
    
    # Save detailed JSON
    json_file = os.path.join(
        args.output_dir,
        f"{model_name}_gpqa_{mode}_{timestamp}.json"
    )
    
    output_data = {
        'model': args.model,
        'benchmark': 'gpqa',
        'accuracy': accuracy,
        'correct_count': correct_count,
        'total': len(results),
        'timestamp': timestamp,
        'config': {
            'temperature': args.temperature,
            'max_new_tokens': args.max_tokens,
            'dry_run': args.dry_run
        },
        'prompts': {
            'system': SYSTEM_PROMPT,
            'user_template': USER_PROMPT_TEMPLATE,
            'format': FORMAT_PROMPT
        },
        'results': results
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Detailed results saved to: {json_file}")
    
    # Save summary
    summary_file = os.path.join(
        args.output_dir,
        f"summary_gpqa_{model_name}_{timestamp}.txt"
    )
    
    with open(summary_file, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Benchmark: GPQA\n")
        f.write(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config: temperature={args.temperature}, max_tokens={args.max_tokens}\n")
        f.write(f"Mode: {'DRY RUN' if args.dry_run else 'FULL EVALUATION'}\n")
    
    print(f"✓ Summary saved to: {summary_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate language models on GPQA benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/eval_gpqa.py --model "mistralai/Mistral-7B-Instruct-v0.3"
  python scripts/eval_gpqa.py --model "Qwen/Qwen2.5-7B-Instruct" --dry-run
  python scripts/eval_gpqa.py --model "meta-llama/Llama-3.2-3B-Instruct" --output-dir ./my_results
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='HuggingFace model identifier (e.g., "mistralai/Mistral-7B-Instruct-v0.3")'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run on 5 samples only for testing'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results/gpqa',
        help='Directory to save results (default: ./results/gpqa)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=2048,
        help='Maximum tokens to generate (default: 2048)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature (default: 0.0 for greedy)'
    )
    
    args = parser.parse_args()
    
    evaluate(args)


if __name__ == "__main__":
    main()