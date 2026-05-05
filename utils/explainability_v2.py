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
        
    def attribute_tokens(self, text: str, target_class: int):
        """
        Compute Integrated Gradients attribution for tokens.
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Tokenize
        encoding = self.tokenizer(
            text, 
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
        
        # Baselines (use pad_token_id instead of literal 0)
        baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)
        
        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=target_class,
            return_convergence_delta=True
        )
        
        # Summarize across embedding dimensions
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / (torch.norm(attributions) + 1e-8)
        
        # Map to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        
        # Filter non-zero tokens (exclude padding)
        results = []
        for i, token in enumerate(tokens):
            if token == self.tokenizer.pad_token:
                continue
            results.append({
                "token": token,
                "score": float(attributions[i].item())
            })
            
        return results
