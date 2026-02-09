# Adding New Benchmarks

This guide explains how to add a new benchmark to the repository.

## Quick Checklist

- [ ] Create dataset loading function in `scripts/utils/data_utils.py`
- [ ] Create answer extraction function in `scripts/utils/answer_extraction.py`
- [ ] Create evaluation script in `scripts/eval_<benchmark>.py`
- [ ] Add documentation in `docs/<BENCHMARK>.md`
- [ ] Update results table in `README.md`
- [ ] Test with `--dry-run` flag

## Step-by-Step Guide

### 1. Data Loading

Add a function to `scripts/utils/data_utils.py`:

```python
def load_<benchmark>_dataset(cache_dir: Optional[str] = None):
    """
    Load <BENCHMARK> dataset.
    
    Returns:
        Dataset or list of examples
    """
    # Implementation here
    pass
```

### 2. Answer Extraction

Add extraction logic to `scripts/utils/answer_extraction.py`:

```python
def extract_answer_<benchmark>(text: str) -> str:
    """
    Extract answer from model output for <BENCHMARK>.
    
    Args:
        text: Model output
        
    Returns:
        Extracted answer string
    """
    # Implementation here
    pass
```

### 3. Evaluation Script

Create `scripts/eval_<benchmark>.py` following this template:

```python
#!/usr/bin/env python3
"""
<BENCHMARK> Evaluation Script

Description of the benchmark.

Usage:
    python scripts/eval_<benchmark>.py --model "model_name"
"""

import argparse
import os
import sys

# Add utils to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    extract_answer_<benchmark>,
    load_model_and_tokenizer,
    generate_response,
    load_<benchmark>_dataset
)

# Define prompts
SYSTEM_PROMPT = "..."
USER_PROMPT_TEMPLATE = "..."

def create_messages(problem):
    """Create chat messages."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(**problem)}
    ]

def evaluate(args):
    """Main evaluation logic."""
    # 1. Load dataset
    # 2. Load model
    # 3. Run evaluation loop
    # 4. Save results
    pass

def main():
    parser = argparse.ArgumentParser(description="Evaluate on <BENCHMARK>")
    parser.add_argument('--model', required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output-dir', default='./results/<benchmark>')
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.0)
    
    args = parser.parse_args()
    evaluate(args)

if __name__ == "__main__":
    main()
```

### 4. Documentation

Create `docs/<BENCHMARK>.md` with:

- Overview of the benchmark
- Dataset details (size, format, source)
- Answer format expected
- Evaluation methodology
- Prompt template
- Known issues
- Baseline results table
- Citations and references

### 5. Update README

Add your benchmark to:

1. The results table
2. The "Supported Benchmarks" section
3. Usage examples

### 6. Testing

Always test before submitting:

```bash
# Dry run with 5 samples
python scripts/eval_<benchmark>.py --model "test_model" --dry-run

# Full evaluation (if computationally feasible)
python scripts/eval_<benchmark>.py --model "test_model"
```

## Code Style Guidelines

### Naming Conventions

- Files: `snake_case` (e.g., `eval_math500.py`)
- Functions: `snake_case` (e.g., `extract_answer_math`)
- Classes: `PascalCase` (if needed)
- Constants: `UPPER_SNAKE_CASE` (e.g., `SYSTEM_PROMPT`)

### Documentation

- All functions must have docstrings
- Use Google-style docstrings
- Include type hints
- Add inline comments for complex logic

### Error Handling

```python
try:
    dataset = load_dataset(...)
except Exception as e:
    raise RuntimeError(f"Failed to load dataset: {str(e)}")
```

### Output Format

All evaluations should save:

1. **Detailed JSON** with:
   - Model name
   - Benchmark name
   - Accuracy and counts
   - Timestamp
   - Configuration
   - Full results array

2. **Summary TXT** with:
   - Model name
   - Benchmark name
   - Accuracy (percentage and fraction)
   - Date
   - Configuration summary

## Common Patterns

### Standard Evaluation Loop

```python
results = []
correct = 0

for idx in tqdm(range(num_samples)):
    sample = dataset[idx]
    
    # Generate
    messages = create_messages(sample)
    response = generate_response(model, tokenizer, messages)
    
    # Extract
    predicted = extract_answer(response)
    
    # Check
    is_correct = (predicted == sample['answer'])
    if is_correct:
        correct += 1
    
    # Store
    results.append({
        'index': idx,
        'is_correct': is_correct,
        # ... other fields
    })
```

### Argument Parsing

Always include:
- `--model`: Required model identifier
- `--dry-run`: Test with 5 samples
- `--output-dir`: Where to save results
- `--max-tokens`: Generation limit
- `--temperature`: Sampling temperature

## Pull Request Checklist

Before submitting:

- [ ] Code follows style guidelines
- [ ] All functions have docstrings
- [ ] Dry run test passes
- [ ] Documentation is complete
- [ ] README is updated
- [ ] No hardcoded paths
- [ ] Handles errors gracefully
- [ ] Results saved in standard format

## Questions?

Open an issue or start a discussion if you need help!