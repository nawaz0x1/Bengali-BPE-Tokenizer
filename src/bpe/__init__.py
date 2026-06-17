"""Bengali BPE public package API.

Import the most-used classes directly from the package::

    from bpe import BPETokenizer, BPETrainer, TrainerConfig

Or use the full path::

    from bpe.trainer import BPETrainer, TrainerConfig, BPEModel
    from bpe.tokenizer import BPETokenizer
    from bpe.vocabulary import Vocabulary
"""

from .trainer import BPEModel, BPETrainer, TrainerConfig
from .tokenizer import BPETokenizer
from .vocabulary import Vocabulary

__version__ = "0.1.0"
__all__ = [
    "BPETrainer",
    "BPETokenizer",
    "BPEModel",
    "TrainerConfig",
    "Vocabulary",
    "__version__",
]
