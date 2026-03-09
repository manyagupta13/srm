
"""
Model loading and inference utilities.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List

def load_model_and_tokenizer(
    model_name: str,
    device: str = "auto",
    trust_remote_code: bool = True,
    use_unsloth: bool = False
):
    """
    Load model and tokenizer from HuggingFace or Unsloth.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device placement ("auto", "cuda", "cpu")
        trust_remote_code: Whether to trust remote code
        use_unsloth: Whether to use Unsloth for loading
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    
    if use_unsloth or "llama" in model_name.lower():
        print("Using Unsloth for model loading...")
        try:
            from unsloth import FastLanguageModel
            
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=8192,
                dtype=None,
                load_in_4bit=False,
            )
            
            FastLanguageModel.for_inference(model)
            print(f"✓ Model loaded successfully via Unsloth")
            
        except ImportError:
            print("⚠ Unsloth not installed, falling back to transformers")
            return _load_with_transformers(model_name, device, trust_remote_code)
    else:
        return _load_with_transformers(model_name, device, trust_remote_code)
    
    return model, tokenizer

def _load_with_transformers(model_name, device, trust_remote_code):
    """Fallback loading with transformers."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    
    if device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=trust_remote_code
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=trust_remote_code
        )
        model = model.to(device)
    
    model.eval()
    print(f"✓ Model loaded successfully on {model.device}")
    
    return model, tokenizer

def generate_response(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 2048,
    temperature: float = 0.0
) -> str:
    """
    Generate response from model using chat template.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        messages: List of message dicts with 'role' and 'content'
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 for greedy)
        
    Returns:
        Generated response text
    """
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return response

