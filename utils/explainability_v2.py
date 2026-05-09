"""
Explainability Engine v2.1 — Feature Attribution via Captum
Implements Integrated Gradients for token-level contribution analysis.
"""

import torch
import numpy as np
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from typing import List, Dict, Tuple

class CaptumExplainer:
    """
    Advanced explainability using Captum attribution.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def attribute_tokens(self, full_input: str, target_class: int, n_steps: int = 20):
        """
        Compute Integrated Gradients attribution for tokens in the full input string.
        Returns both the list of all attributions and a split for the dashboard.
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Tokenize full input
        encoding = self.tokenizer(
            full_input, 
            return_tensors='pt', 
            truncation=True, 
            max_length=256, 
            padding='max_length'
        )
        input_ids = encoding["input_ids"].to(next(self.model.parameters()).device)
        attention_mask = encoding["attention_mask"].to(next(self.model.parameters()).device)
        
        # We need a wrapper that returns just the emotion logits for Captum
        def model_forward_wrapper(ids, mask):
            outputs = self.model(ids, mask)
            return outputs["emotion_logits"]
            
        # Layer Integrated Gradients on embeddings
        lig = LayerIntegratedGradients(model_forward_wrapper, self.model.roberta.embeddings)
        
        # Baselines (use pad_token_id)
        baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)
        
        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=target_class,
            n_steps=n_steps,
            return_convergence_delta=True
        )
        
        # Summarize across embedding dimensions
        attributions = attributions.sum(dim=-1).squeeze(0)
        # Normalize
        attributions = attributions / (torch.norm(attributions) + 1e-8)
        
        # Map to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        
        # 1. Identify current span boundaries
        current_start_idx = -1
        current_end_idx = len(tokens)
        
        for i, t in enumerate(tokens):
            ct = t.upper()
            if "URRENT" in ct:
                # Check for opening vs closing
                is_closing = any("/" in tokens[j] for j in range(max(0, i-2), i))
                if is_closing:
                    if current_end_idx == len(tokens):
                        # Start of [/CURRENT] is usually 2 tokens back (e.g., [, / or [/)
                        current_end_idx = i - 1
                        while current_end_idx > 0 and "[" not in tokens[current_end_idx]:
                            current_end_idx -= 1
                else:
                    if current_start_idx == -1:
                        # End of [CURRENT] is usually 1 token ahead (the ']')
                        current_start_idx = i + 1
                        while current_start_idx < len(tokens) and "]" not in tokens[current_start_idx]:
                            current_start_idx += 1
                        current_start_idx += 1 # Move past the ']'
        
        # 2. Process tokens and filter tags
        all_results = []
        context_results = []
        current_results = []
        
        tag_parts = {"[", "]", "/", " [", "Ġ[", "Ġ[/", "Ġ/", "C", "URRENT", "CON", "TEXT", "SC", "EN", "AR", "IO", "TOP", "IC"}
        
        for i, token in enumerate(tokens):
            if token in ["<s>", "</s>", "<pad>"] or token == self.tokenizer.pad_token:
                continue
            
            clean_token = token.strip().upper()
            if clean_token in tag_parts or token in tag_parts:
                continue
            
            # If the token is basically just punctuation inside brackets, ignore it
            if len(clean_token) == 1 and clean_token in "[]/":
                continue
                
            entry = {
                "token": token.replace("Ġ", " "), 
                "score": float(attributions[i].item())
            }
            
            all_results.append(entry)
            
            if current_start_idx <= i < current_end_idx:
                current_results.append(entry)
            else:
                context_results.append(entry)
            
        return {
            "token_attributions": all_results,
            "context_span_top": sorted(context_results, key=lambda x: abs(x["score"]), reverse=True)[:10],
            "current_span_top": sorted(current_results, key=lambda x: abs(x["score"]), reverse=True)[:10]
        }




