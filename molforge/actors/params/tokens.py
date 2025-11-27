from typing import Optional
from dataclasses import dataclass

import os

from .base import BaseParams


@dataclass
class TokenizeDataParams(BaseParams):
    """Configuration parameters for SMILES tokenization component."""
    
    vocab_file: Optional[str] = None
    SMILES_column: str = 'curated_smiles'
    dynamically_update_vocab: bool = True
    
    def _validate_params(self) -> None:
        if not self.dynamically_update_vocab and self.vocab_file is not None and not os.path.isfile(self.vocab_file):
            raise ValueError(f"Invalid vocab_file. Does not exist: {self.vocab_file}")