# MakeMore

Character-level name generation with a PyTorch attention model.

## Status

Personal learning and experimentation project, published from the existing local source. Training quality and end-to-end execution have not been independently benchmarked for this upload.

## Setup

Use a Python virtual environment, then install the direct dependencies:

```sh
python -m pip install -r requirements.txt
python makemore.py
```

Dependencies are not version-pinned; this is not a tested environment lockfile. Install a PyTorch build appropriate for your CPU or CUDA environment. Training can be computationally intensive.

Provide a UTF-8 `names.txt` file with one training name per line. The local dataset is omitted because its source and redistribution terms have not been documented. Use the menu to train first, then generate from the saved model.

## Source files

- `makemore.py`

## Data and checkpoints

This repository contains source code. Local model weights, training caches, downloaded datasets, logs and private configuration are excluded. Obtain required datasets from their original providers and follow their terms.

## Project notes

This snapshot preserves the existing experiments, including older variants. No accuracy or performance claims are made here. Dataset references and upstream libraries are identified where visible in the source; detailed project history and any tutorial or collaboration credits still need to be supplied by the author.
