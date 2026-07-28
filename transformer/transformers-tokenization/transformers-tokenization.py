import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        self.word_to_id: Dict[str, int] = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3
        }
        self.id_to_word: Dict[int, str] = {
            0: self.pad_token,
            1: self.unk_token,
            2: self.bos_token,
            3: self.eos_token
        }
        self.vocab_size = 0
        
        # Special tokens
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        unique_words = set()
        for text in texts:
            words = text.lower().split()
            unique_words.update(words)

        nextid = len(self.word_to_id)
        for word in sorted(unique_words):
            if word not in self.word_to_id:
                self.word_to_id[word] = nextid
                self.id_to_word[nextid] = word
                nextid += 1
                
        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        words = text.lower().split()
        return[self.word_to_id.get(w, self.word_to_id["<UNK>"]) for w in words]
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        words = [self.id_to_word.get(i, "<UNK>") for i in ids]
        return " ".join(words)
