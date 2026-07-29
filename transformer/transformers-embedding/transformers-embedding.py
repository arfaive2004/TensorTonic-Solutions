import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    # Your code here
    torch.manual_seed(42) # to generate the same random numbers every time, 42 does not signify anything, it is random
    embedding = nn.Embedding(vocab_size, d_model)
    return embedding
    

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    # Your code here
    looked_up = embedding(tokens)
    scaled = looked_up * math.sqrt(d_model)
    return scaled