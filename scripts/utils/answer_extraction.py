"""
Answer extraction and normalization utilities for different benchmarks.
"""

import re
from typing import List, Optional


# ================================
# MATH-500 Answer Extraction
# ================================

def extract_boxed_text(output: str) -> Optional[str]:
    """
    Extract text from \\boxed{...} with proper brace matching.
    
    Args:
        output: Model output text
        
    Returns:
        Extracted text or None if no boxed content found
    """
    start_tag = r'\boxed{'
    start_idx = output.find(start_tag)
    if start_idx == -1:
        return None

    idx = start_idx + len(start_tag)
    depth = 1
    out_chars = []

    while idx < len(output) and depth > 0:
        c = output[idx]
        if c == '{':
            depth += 1
            out_chars.append(c)
        elif c == '}':
            depth -= 1
            if depth == 0:
                break
            else:
                out_chars.append(c)
        else:
            out_chars.append(c)
        idx += 1

    if depth != 0:
        return None

    return ''.join(out_chars).strip()


def extract_answer_math(output: str) -> str:
    """
    Extract and normalize answer from MATH benchmark model output.
    Handles multiple formats including truncated responses.
    
    Args:
        output: Model output text
        
    Returns:
        Normalized answer string
    """
    # First try: boxed format
    raw_in_box = extract_boxed_text(output)
    if raw_in_box is not None:
        normalized = normalize_answer(raw_in_box)
        return normalized
    
    # Second try: Look for "answer is X" or "= X" patterns
    # Common in step-by-step solutions even if truncated
    
    # Pattern: "answer is X" or "answer: X"
    patterns = [
        r'(?:answer|Answer|ANSWER)\s*(?:is|:)\s*\$?([^\$\n]+)\$?',
        r'(?:final answer|Final Answer)\s*(?:is|:)\s*\$?([^\$\n]+)\$?',
        r'(?:solution|Solution)\s*(?:is|:)\s*\$?([^\$\n]+)\$?',
        r'(?:therefore|Therefore)\s*,?\s*\$?([^\$\n]+)\$?',
        # Pattern for "= X" at end of line
        r'=\s*\$?([^\$\n\.]+)\$?\s*$',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, output, re.MULTILINE)
        if matches:
            # Take the last match (most likely the final answer)
            candidate = matches[-1].strip()
            # Remove trailing punctuation
            candidate = re.sub(r'[,\.]$', '', candidate)
            if candidate:
                return normalize_answer(candidate)
    
    # Third try: Extract last mathematical expression in $ $ or \[ \]
    math_patterns = [
        r'\$([^\$]+)\$',
        r'\\\[([^\]]+)\\\]',
    ]
    
    for pattern in math_patterns:
        matches = re.findall(pattern, output)
        if matches:
            # Take last match
            candidate = matches[-1].strip()
            if len(candidate) < 50:  # Reasonable answer length
                return normalize_answer(candidate)
    
    # Last resort: empty string
    return ""


def normalize_answer(answer: str) -> str:
    """
    Normalize mathematical answer for comparison.
    
    Args:
        answer: Raw answer string
        
    Returns:
        Normalized answer string
    """
    if answer is None:
        return ""
    answer = answer.strip()

    # Remove outer \text{...} if present
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
    if m is not None:
        answer = m.group("text").strip()

    return _strip_string(answer)


def _strip_string(string: str) -> str:
    """Strip and normalize string for comparison."""
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) > 0 and string[0] == ".":
        string = "0" + string
    if "=" in string:
        parts = string.split("=")
        if len(parts) == 2 and len(parts[0]) <= 2:
            string = parts[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def _remove_right_units(string: str) -> str:
    """Remove units from the right side of the string."""
    if "\\text{" in string:
        splits = string.split("\\text{")
        return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    """Fix sqrt notation."""
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split.startswith("{"):
            a = split[0]
            new_substr = f"\\sqrt{{{a}}}{split[1:]}"
            new_string += new_substr
        else:
            new_string += "\\sqrt" + split
    return new_string


def _fix_fracs(string: str) -> str:
    """Fix fraction notation."""
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for sub in substrs[1:]:
            new_str += "\\frac"
            if sub.startswith("{"):
                new_str += sub
            else:
                if len(sub) >= 2:
                    a, b = sub[0], sub[1]
                    if b != "{":
                        if len(sub) > 2:
                            new_str += f"{{{a}}}{{{b}}}{sub[2:]}"
                        else:
                            new_str += f"{{{a}}}{{{b}}}"
                    else:
                        new_str += f"{{{a}}}{sub[1:]}"
                else:
                    new_str += sub
    return new_str


def _fix_a_slash_b(string: str) -> str:
    """Convert simple fractions a/b to \\frac{a}{b}."""
    if string.count("/") == 1:
        a, b = string.split("/")
        try:
            a_int = int(a)
            b_int = int(b)
            return f"\\frac{{{a_int}}}{{{b_int}}}"
        except:
            return string
    return string


def check_answer_correct(prediction: str, ground_truth: str) -> bool:
    """
    Check if predicted answer matches ground truth.
    
    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        True if answers match after normalization
    """
    pred_norm = normalize_answer(prediction) if prediction else ""
    gt_norm = normalize_answer(ground_truth) if ground_truth else ""
    return pred_norm == gt_norm


# ================================
# GPQA Answer Extraction
# ================================

def extract_answer_gpqa(text: str, valid_choices: List[str]) -> Optional[str]:
    """
    Extract single-letter answer from GPQA model output.
    
    Args:
        text: Model output text
        valid_choices: List of valid choice strings (e.g., ['A) ...', 'B) ...'])
        
    Returns:
        Extracted letter (A-D) or None if extraction failed
    """
    valid = {c[:1].upper() for c in valid_choices}
    
    # Try ##Answer: X## format first
    m = re.search(r'##\s*[Aa]nswer\s*:\s*([A-D])\s*##', text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    
    # Try "answer is X" pattern
    m = re.search(r'answer[^A-Za-z0-9]*([A-D])', text, flags=re.I)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    
    # Check for standalone letter on its own line
    for line in text.splitlines():
        s = line.strip().upper()
        if s in valid and len(s) == 1:
            return s
    
    # Try letters in parentheses (reversed to get last occurrence)
    for m in reversed(list(re.finditer(r'\(([A-D])\)', text))):
        if m.group(1).upper() in valid:
            return m.group(1).upper()
    
    # Last resort: any single letter with whitespace around it
    for m in reversed(list(re.finditer(r'\s([A-D])\s', text))):
        if m.group(1).upper() in valid:
            return m.group(1).upper()
    
    return None