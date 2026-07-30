import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    positions = np.arange(seq_length).reshape(seq_length, 1)

    i = np.arange(0, d_model, 2)
    div_term = 1.0 / (10000 ** (i / d_model))

    angles = positions * div_term

    pe = np.zeros((seq_length, d_model))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)

    return pe
        