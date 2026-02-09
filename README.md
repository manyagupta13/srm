# LLM Benchmark Baselines

Official baseline implementations for evaluating language models on academic benchmarks. This repository provides reproducible evaluation scripts for multiple models across standard reasoning and math benchmarks.

## 📊 Benchmark Results

| Model | GSM8K | GPQA | MATH500 | SVAMP |
|-------|-------|------|---------|-------|
| **Qwen2.5-7B-Instruct** | 89.20% | 35.35% | **70.00%** | - |
| **Llama 3.2 3B** | 86.80% | 33.33% | - | - |
| **Mistral-7B-Instruct-v0.3** | 51.60% | **34.85%** | - | - |

*Last updated: February 2026 | "-" indicates evaluation not yet completed*

## 🎯 Supported Benchmarks

### MATH-500
Mathematical problem solving with 500 problems from the MATH dataset.
- **Dataset Size**: 500 questions
- **Answer Format**: LaTeX `\boxed{...}` 
- **Evaluation**: Normalized string matching
- **Status**: ✅ Complete scripts available

### GPQA
Graduate-level science questions testing PhD-level knowledge.
- **Dataset Size**: 198 multiple-choice questions  
- **Answer Format**: Single letter (A-D)
- **Evaluation**: Exact match
- **Status**: ✅ Complete scripts available

### GSM8K
Grade school math word problems.
- **Dataset Size**: 1,319 questions
- **Status**: 🚧 Coming soon

### SVAMP
Math word problems with varying structural complexity.
- **Status**: 🚧 Coming soon

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/llm-benchmark-baselines.git
cd llm-benchmark-baselines
pip install -r requirements.txt
```

### Running Evaluations

**MATH-500 Evaluation:**
```bash
python scripts/eval_math500.py --model "Qwen/Qwen2.5-7B-Instruct"
```

**GPQA Evaluation:**
```bash
python scripts/eval_gpqa.py --model "mistralai/Mistral-7B-Instruct-v0.3"
```

**Dry Run (5 samples for testing):**
```bash
python scripts/eval_math500.py --model "Qwen/Qwen2.5-7B-Instruct" --dry-run
```

### Command-Line Options

All evaluation scripts support the following arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | HuggingFace model identifier | Required |
| `--dry-run` | Test on 5 samples only | False |
| `--output-dir` | Directory for results | `./results` |
| `--max-tokens` | Maximum generation length | 4096 (MATH), 2048 (GPQA) |
| `--temperature` | Sampling temperature | 0.0 (greedy) |

## 📁 Repository Structure

```
llm-benchmark-baselines/
├── scripts/
│   ├── eval_math500.py          # MATH-500 evaluation
│   ├── eval_gpqa.py             # GPQA evaluation  
│   └── utils/                   # Shared utilities
│       ├── __init__.py
│       ├── answer_extraction.py # Answer parsing functions
│       ├── model_utils.py       # Model loading utilities
│       └── data_utils.py        # Dataset loading utilities
├── results/                     # Evaluation outputs (gitignored)
│   ├── math500/
│   ├── gpqa/
│   └── summaries/
├── docs/
│   ├── MATH500.md              # MATH-500 documentation
│   ├── GPQA.md                 # GPQA documentation
│   └── ADDING_BENCHMARKS.md    # Guide for adding new benchmarks
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 📝 Output Format

Each evaluation produces two types of output files:

### 1. Detailed JSON Results

Filename: `{model_name}_{benchmark}_{timestamp}.json`

```json
{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "benchmark": "math500",
  "accuracy": 0.70,
  "correct_count": 350,
  "total": 500,
  "timestamp": "2026-02-08T13:06:08",
  "config": {
    "temperature": 0.0,
    "max_new_tokens": 4096
  },
  "results": [
    {
      "index": 0,
      "problem": "What is 2+2?",
      "reference_answer": "4",
      "model_response": "The answer is \\boxed{4}",
      "extracted_answer": "4",
      "is_correct": true
    }
  ]
}
```

### 2. Summary Text File

Filename: `summary_{benchmark}_{model_name}.txt`

```
Model: Qwen/Qwen2.5-7B-Instruct
Benchmark: MATH-500
Accuracy: 70.00% (350/500)
Date: 2026-02-08 13:06:08
Config: temperature=0.0, max_tokens=4096
```

## 🔧 Implementation Details

### MATH-500
- Downloads from [SLM-MUX repository](https://github.com/slm-mux/slm-mux.github.io) with automatic fallback to HuggingFace
- Uses greedy decoding (temperature=0.0) for reproducibility
- Extracts answers from `\boxed{...}` with proper brace matching
- Normalizes mathematical expressions:
  - Fraction standardization (`\frac`, `\tfrac`, `\dfrac`)
  - Square root formatting
  - Unit removal
  - Decimal/fraction normalization

### GPQA
- 198 graduate-level multiple-choice questions in physics, chemistry, biology
- Robust answer extraction with multiple patterns:
  - `##Answer: X##` format (preferred)
  - Standalone letter on its own line
  - Letters in parentheses `(A)`
  - Pattern-based extraction as fallback
- Handles edge cases where models don't follow format

## 🤝 Contributing

We welcome contributions! To add a new benchmark or model:

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/new-benchmark`
3. **Follow the existing patterns** in `scripts/eval_*.py`
4. **Update the results table** in README.md
5. **Add documentation** in `docs/`
6. **Submit a pull request**

### Adding a New Benchmark

See [`docs/ADDING_BENCHMARKS.md`](docs/ADDING_BENCHMARKS.md) for detailed instructions.

Key requirements:
- Use argparse for CLI arguments
- Include dry-run mode (5 samples)
- Save results in standardized JSON format
- Add proper logging and error handling
- Follow PEP 8 style guidelines

## 📚 Citation

If you use these baselines in your research, please cite:

```bibtex
@software{llm_benchmark_baselines_2024,
  title = {LLM Benchmark Baselines: Reproducible Evaluation Scripts},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/llm-benchmark-baselines}
}
```

## 🙏 Acknowledgments

This repository builds on excellent prior work:

- **Datasets**:
  - [SLM-MUX](https://github.com/slm-mux/slm-mux.github.io) - Dataset curation and benchmarking framework
  - [MATH Dataset](https://github.com/hendrycks/math) - Hendrycks et al.
  - [GPQA](https://github.com/idavidrein/gpqa) - Rein et al.
  - [GSM8K](https://github.com/openai/grade-school-math) - OpenAI
  
- **Models**:
  - [Qwen Team](https://huggingface.co/Qwen) - Qwen model series
  - [Meta AI](https://huggingface.co/meta-llama) - Llama models
  - [Mistral AI](https://huggingface.co/mistralai) - Mistral models

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🐛 Issues & Support

- **Found a bug?** [Open an issue](https://github.com/yourusername/llm-benchmark-baselines/issues)
- **Have a question?** Check our [documentation](docs/) or start a [discussion](https://github.com/yourusername/llm-benchmark-baselines/discussions)
- **Want to contribute?** See our [contributing guidelines](#contributing)

## 📈 Reproducibility

All evaluation scripts use:
- Fixed random seeds for reproducibility
- Greedy decoding (temperature=0.0) by default
- Standardized prompting formats
- Deterministic answer extraction

Hardware used for reported results:
- GPU: NVIDIA Tesla T4 (15GB VRAM)
- Precision: bfloat16
- Batch size: 1 (for consistency)

---

 
**Last Updated:** February 2026