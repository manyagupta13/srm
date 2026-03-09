"""
Dataset loading utilities for different benchmarks.
"""

import os
import json
import subprocess
from datasets import load_dataset, Dataset
from typing import Optional


def load_math500_dataset(cache_dir: Optional[str] = None) -> Dataset:
    """
    Load MATH-500 dataset from SLM-MUX repository or HuggingFace.
    
    Args:
        cache_dir: Optional cache directory for downloaded data
        
    Returns:
        Dataset object with 500 MATH problems
    """
    print("="*60)
    print("LOADING MATH-500 DATASET")
    print("="*60)
    
    # Try to clone SLM-MUX repository
    repo_url = "https://github.com/slm-mux/slm-mux.github.io.git"
    repo_dir = cache_dir or "./slm-mux-repo"
    
    try:
        if not os.path.exists(repo_dir):
            print(f"Cloning SLM-MUX repository...")
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, repo_dir],
                check=True,
                capture_output=True,
                timeout=60
            )
        
        # Check for data file
        data_file = os.path.join(repo_dir, "slm_mux_code", "data", "math_500.json")
        
        if os.path.exists(data_file):
            print(f"✓ Found math_500.json in repository")
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            # Convert to Dataset format
            if isinstance(data, list):
                dataset = Dataset.from_list(data)
            else:
                dataset = Dataset.from_dict(data)
            
            print(f"✓ Loaded {len(dataset)} samples from SLM-MUX")
            return dataset
            
    except Exception as e:
        print(f"⚠ Could not load from SLM-MUX repository: {str(e)}")
    
    # Fallback: Create from standard MATH dataset
    print("\nFalling back to standard MATH dataset...")
    return _create_math500_from_huggingface()


def _create_math500_from_huggingface() -> Dataset:
    """Create MATH-500 subset from HuggingFace MATH dataset."""
    print("Loading from HuggingFace MATH dataset...")
    
    try:
        # Try lighteval/MATH first
        dataset = load_dataset("lighteval/MATH", split="test")
    except:
        try:
            # Fallback to hendrycks/math
            dataset = load_dataset("hendrycks/math", "all", split="test")
        except Exception as e:
            raise RuntimeError(f"Failed to load MATH dataset: {str(e)}")
    
    # Select first 500 samples
    dataset = dataset.select(range(min(500, len(dataset))))
    
    print(f"✓ Created MATH-500 subset: {len(dataset)} samples")
    return dataset


def load_gpqa_dataset(cache_dir: Optional[str] = None) -> list:
    """
    Load GPQA dataset from SLM-MUX repository.
    
    Args:
        cache_dir: Optional cache directory for downloaded data
        
    Returns:
        List of question dictionaries
    """
    print("="*60)
    print("LOADING GPQA DATASET")
    print("="*60)
    
    # Clone or use existing SLM-MUX repository
    repo_url = "https://github.com/slm-mux/slm-mux.github.io.git"
    repo_dir = cache_dir or "./slm-mux-repo"
    
    try:
        if not os.path.exists(repo_dir):
            print(f"Cloning SLM-MUX repository...")
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, repo_dir],
                check=True,
                capture_output=True,
                timeout=60
            )
        
        # Load GPQA data
        data_file = os.path.join(repo_dir, "slm_mux_code", "data", "gpqa_shuffled.json")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"GPQA data file not found: {data_file}")
        
        print(f"✓ Found gpqa_shuffled.json")
        with open(data_file, 'r') as f:
            dataset = json.load(f)
        
        print(f"✓ Loaded {len(dataset)} questions from SLM-MUX")
        return dataset
        
    except Exception as e:
        raise RuntimeError(f"Failed to load GPQA dataset: {str(e)}")