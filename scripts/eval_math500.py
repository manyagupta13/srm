#!/usr/bin/env python3
"""
MATH-500 Benchmark Evaluation Script

Evaluates language models on the MATH-500 dataset (500 mathematical problems).

Usage:
    python scripts/eval_math500.py --model "Qwen/Qwen2.5-7B-Instruct"
    python scripts/eval_math500.py --model "Qwen/Qwen2.5-7B-Instruct" --dry-run

For more options, run:
    python scripts/eval_math500.py --help
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
    extract_answer_math,
    normalize_answer,
    check_answer_correct,
    load_model_and_tokenizer,
    generate_response,
    load_math500_dataset
)


# Prompt templates
SYSTEM_PROMPT = (
    "Solve the following problem step by step. "
    "Review what you have done and make sure you have not made any mistakes. "
    "Be careful with intervals and plus or minus signs. Those parts are very easy to make mistakes. "
    "Provide the final answer enclosed in LaTeX \\boxed{...}."
)

USER_PROMPT_TEMPLATE = "Problem:\n{problem}\n"
FORMAT_PROMPT = "\nPlease provide your final answer in the form: \\boxed{{...}}"


def create_messages(problem: str) -> List[Dict[str, str]]:
    """Create chat messages for the model."""
    user_content = USER_PROMPT_TEMPLATE.format(problem=problem) + FORMAT_PROMPT
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


def evaluate(args):
    """Run evaluation on MATH-500 dataset."""
    
    # Setup
    print("="*60)
    print("MATH-500 BENCHMARK EVALUATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Mode: {'DRY RUN (5 samples)' if args.dry_run else 'FULL EVALUATION (500 samples)'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print("="*60 + "\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_math500_dataset()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Determine number of samples
    num_samples = 5 if args.dry_run else len(dataset)
    
    # Run evaluation
    results = []
    correct = 0
    
    print(f"\nEvaluating {num_samples} samples...\n")
    
    for idx in tqdm(range(num_samples), desc="Evaluating"):
        sample = dataset[idx]
        
        # Extract problem and ground truth
        problem = sample.get('problem', sample.get('question', sample.get('Problem', '')))
        ground_truth_raw = sample.get('answer', sample.get('solution', sample.get('Answer', '')))
        ground_truth = normalize_answer(ground_truth_raw)
        
        # Generate prediction
        messages = create_messages(problem)
        response = generate_response(
            model, tokenizer, messages,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        # Extract and check answer
        predicted_answer = extract_answer_math(response)
        is_correct = check_answer_correct(predicted_answer, ground_truth)
        
        if is_correct:
            correct += 1
        
        # Store result
        result = {
            'index': idx,
            'problem': problem,
            'reference_answer': ground_truth,
            'model_response': response,
            'extracted_answer': predicted_answer,
            'is_correct': is_correct
        }
        results.append(result)
        
        # Print sample in dry run mode
        if args.dry_run and idx < 3:
            print(f"\n{'='*60}")
            print(f"Sample {idx + 1}")
            print(f"{'='*60}")
            print(f"Problem: {problem[:200]}...")
            print(f"Reference: {ground_truth[:80]}")
            print(f"Predicted: {predicted_answer}")
            print(f"Status: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
        
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
        f"{model_name}_math500_{mode}_{timestamp}.json"
    )
    
    output_data = {
        'model': args.model,
        'benchmark': 'math500',
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
        f"summary_math500_{model_name}_{timestamp}.txt"
    )
    
    with open(summary_file, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Benchmark: MATH-500\n")
        f.write(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config: temperature={args.temperature}, max_tokens={args.max_tokens}\n")
        f.write(f"Mode: {'DRY RUN' if args.dry_run else 'FULL EVALUATION'}\n")
    
    print(f"✓ Summary saved to: {summary_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate language models on MATH-500 benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/eval_math500.py --model "Qwen/Qwen2.5-7B-Instruct"
  python scripts/eval_math500.py --model "Qwen/Qwen2.5-7B-Instruct" --dry-run
  python scripts/eval_math500.py --model "meta-llama/Llama-3.2-3B-Instruct" --output-dir ./my_results
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='HuggingFace model identifier (e.g., "Qwen/Qwen2.5-7B-Instruct")'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run on 5 samples only for testing'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results/math500',
        help='Directory to save results (default: ./results/math500)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=4096,
        help='Maximum tokens to generate (default: 4096)'
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