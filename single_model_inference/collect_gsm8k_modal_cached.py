#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cached Modal inference - loads models ONCE and keeps them in GPU memory
"""

import os
import re
import modal

app = modal.App("gsm8k-cached-inference")
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


# ===== PERSISTENT CACHED INFERENCE ENGINE =====
# Models load ONCE via @modal.enter(), then stay in GPU memory for all inferences
@app.cls(image=image, gpu="A100-40GB", timeout=600)
class CachedInferenceEngine:
    def __init__(self):
        """Initialize instance variables (called locally by Modal)"""
        self.torch = None
        self.models = {}
        self.tokenizers = {}
        self.device = None
    
    @modal.enter()
    def load_models(self):
        """Load models ONCE when container starts - they persist for all calls"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("[*] LOADING MODELS INTO GPU MEMORY (ONE TIME ONLY)...")
        
        self.torch = torch
        self.torch.manual_seed(42)
        self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        
        model_names = ["Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]
        
        for model_name in model_names:
            print(f"  Loading {model_name}...")
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
            self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=self.torch.float16, device_map="auto"
            )
            print(f"  ✓ {model_name} cached in GPU")
        
        print("[✓] ALL MODELS LOADED - WILL NEVER RELOAD")
    
    @modal.method()
    def infer(self, model_name: str, messages: list, max_tokens: int = 512, 
              temperature: float = 0.0, top_p: float = 1.0, top_k: int = 1):
        """
        Fast inference using pre-loaded cached models.
        NO model loading happens here - models are already in GPU memory.
        """
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not in cache")
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Build prompt
        text = ""
        for msg in messages:
            if msg.get("role") == "system":
                text += f"{msg.get('content', '')}\n\n"
            else:
                text += msg.get("content", "")
        
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with self.torch.no_grad():
            gen_kwargs = {"max_new_tokens": max_tokens, "top_p": top_p, "top_k": top_k}
            if temperature > 0.0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["do_sample"] = True
            else:
                gen_kwargs["do_sample"] = False
            outputs = model.generate(**inputs, **gen_kwargs)
        
        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return {"content": generated_text, "token_usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
