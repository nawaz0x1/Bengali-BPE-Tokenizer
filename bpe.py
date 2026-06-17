#!/usr/bin/env python3
"""Entry-point script for the Bengali BPE Tokenizer.

Adds the ``src/`` directory to the Python path so the package can be
used without installing it first, then delegates to the CLI main function.

Usage::

    python bpe.py train  --file corpus.txt --vocab-size 8000 --output output/
    python bpe.py encode --model output/   --text "আমি বাংলাদেশে থাকি।"
    python bpe.py decode --model output/   --tokens 245 891 7
    python bpe.py stats  --model output/
    python bpe.py inspect --model output/  --word বাংলাদেশ
"""

import sys
from pathlib import Path

# Make the src/ layout importable when running this script directly
_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bpe.cli import main  # noqa: E402  (import after path manipulation)

if __name__ == "__main__":
    main()
