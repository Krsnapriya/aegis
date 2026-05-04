"""
Explainability Pipeline: Tokenization Visualization, Embeddings PCA, and Cosine Similarity.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
from transformers import RobertaTokenizer


class ExplainabilityEngine:
    """
    Extracts and visualizes model internals: tokens, embeddings, and similarity.
    """
    
    def __init__(self, tokenizer_name: str = "roberta-base", emotion_dict: Optional[Dict] = None):
        """
        Initialize explainability engine.
        
        Args:
            tokenizer_name: HuggingFace tokenizer
            emotion_dict: Plutchik emotion dictionary (for names)
        """
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        self.emotion_dict = emotion_dict or {}
        self.pca_2d = None
        self.pca_3d = None
        self.emotion_centroids = {}  # Will be populated during training
    
    def tokenize_with_visualization(self, text: str) -> Dict:
        """
        Tokenize text and prepare for visualization.
        
        Args:
            text: Input text
        
        Returns:
            Dict with token_ids, tokens, token_strings
        """
        encoding = self.tokenizer(
            text,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding["input_ids"].squeeze().cpu().numpy()
        
        # Convert token IDs to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        # Remove padding tokens for display
        attention_mask = encoding["attention_mask"].squeeze().cpu().numpy()
        active_tokens = [tokens[i] for i in range(len(tokens)) if attention_mask[i] == 1]
        active_ids = [input_ids[i] for i in range(len(input_ids)) if attention_mask[i] == 1]
        
        return {
            "token_ids": input_ids.tolist(),
            "tokens": tokens,
            "active_tokens": active_tokens,
            "active_token_ids": active_ids,
            "attention_mask": attention_mask.tolist()
        }
    
    def extract_embeddings(self, last_hidden_state: torch.Tensor, 
                          attention_mask: torch.Tensor) -> Dict:
        """
        Extract and process embeddings from last hidden state.
        
        Args:
            last_hidden_state: [batch_size, seq_length, hidden_dim] or [seq_length, hidden_dim]
            attention_mask: Attention mask for filtering padding
        
        Returns:
            Dict with CLS embedding, mean pooling, and token embeddings
        """
        if last_hidden_state.dim() == 3:
            last_hidden_state = last_hidden_state.squeeze(0)
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.squeeze(0)
        
        # CLS token (first token)
        cls_embedding = last_hidden_state[0, :]  # [hidden_dim]
        
        # Mean pooling over non-padding tokens
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden_state)
        sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=0)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=0), min=1e-9)
        mean_embedding = sum_embeddings / sum_mask  # [hidden_dim]
        
        return {
            "cls_embedding": cls_embedding.detach().cpu().numpy(),
            "mean_embedding": mean_embedding.detach().cpu().numpy(),
            "all_token_embeddings": last_hidden_state.detach().cpu().numpy(),
            "attention_mask": attention_mask.detach().cpu().numpy()
        }
    
    def reduce_embeddings_pca(self, embeddings_list: List[np.ndarray], 
                             n_components: int = 2) -> Tuple[np.ndarray, PCA]:
        """
        Apply PCA to reduce high-dimensional embeddings.
        
        Args:
            embeddings_list: List of [hidden_dim] embeddings
            n_components: 2 or 3
        
        Returns:
            (reduced_embeddings, fitted_pca)
        """
        embeddings_array = np.vstack(embeddings_list)  # [n_samples, hidden_dim]
        
        # Standardize
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_array)
        
        # PCA
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(embeddings_scaled)
        
        return reduced, pca
    
    def visualize_embedding_heatmap(self, embeddings: np.ndarray) -> Dict:
        """
        Prepare embedding as heatmap data (useful for Plotly heatmap).
        
        Args:
            embeddings: [seq_length, hidden_dim]
        
        Returns:
            Dict with heatmap-friendly data
        """
        # Normalize to [0, 1] for visualization
        em_min = embeddings.min(axis=0, keepdims=True)
        em_max = embeddings.max(axis=0, keepdims=True)
        em_max = np.where(em_max == em_min, 1.0, em_max)  # Avoid division by zero
        
        normalized = (embeddings - em_min) / (em_max - em_min + 1e-8)
        
        # Sample columns for visualization (768 is too many)
        sample_cols = np.linspace(0, normalized.shape[1] - 1, 30, dtype=int)
        sampled = normalized[:, sample_cols]
        
        return {
            "heatmap_data": sampled,
            "seq_length": embeddings.shape[0],
            "hidden_dim": embeddings.shape[1],
            "sampled_dims": sample_cols.tolist()
        }
    
    def compute_cosine_similarity(self, embedding: np.ndarray, 
                                 centroid: np.ndarray) -> float:
        """
        Compute cosine similarity between an embedding and emotion centroid.
        
        Args:
            embedding: [hidden_dim] numpy array
            centroid: [hidden_dim] numpy array
        
        Returns:
            Cosine similarity score [0, 1]
        """
        similarity = np.dot(embedding, centroid) / (
            np.linalg.norm(embedding) * np.linalg.norm(centroid) + 1e-8
        )
        # Map from [-1, 1] to [0, 1]
        similarity = (similarity + 1.0) / 2.0
        return float(similarity)
    
    def register_emotion_centroids(self, emotion_embeddings: Dict[str, List[np.ndarray]]):
        """
        Register emotion centroids from training data.
        Each emotion has a list of embeddings; compute mean as centroid.
        
        Args:
            emotion_embeddings: Dict[emotion_name -> List[embeddings]]
        """
        for emotion, embeddings_list in emotion_embeddings.items():
            if len(embeddings_list) > 0:
                centroid = np.mean(embeddings_list, axis=0)
                self.emotion_centroids[emotion] = centroid
    
    def explain_prediction(self, input_text: str, model_output: Dict,
                          predicted_emotion: str, top_k: int = 3) -> Dict:
        """
        Generate full explainability report for a single prediction.
        
        Args:
            input_text: Original input text
            model_output: Output dict from model (contains embeddings)
            predicted_emotion: Predicted emotion name
            top_k: Top K similar emotions in embedding space
        
        Returns:
            Comprehensive explainability dict
        """
        # Tokenization
        token_info = self.tokenize_with_visualization(input_text)
        
        # Embeddings
        embedding_info = self.extract_embeddings(
            model_output["last_hidden_state"],
            model_output.get("attention_mask", torch.ones(model_output["last_hidden_state"].shape[:-1]))
        )
        
        cls_emb = embedding_info["cls_embedding"]
        
        # Heatmap
        heatmap_info = self.visualize_embedding_heatmap(embedding_info["all_token_embeddings"])
        
        # Cosine similarity to emotion centroids
        similarities = {}
        for emotion, centroid in self.emotion_centroids.items():
            sim = self.compute_cosine_similarity(cls_emb, centroid)
            similarities[emotion] = sim
        
        # Top K
        sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        top_k_emotions = sorted_sims[:top_k]
        
        return {
            "tokens": token_info,
            "embeddings": embedding_info,
            "heatmap": heatmap_info,
            "cosine_similarities": similarities,
            "top_k_similar_emotions": top_k_emotions,
            "predicted_emotion": predicted_emotion,
            "prediction_confidence": float(similarities.get(predicted_emotion, 0.0))
        }
