# Protein Secondary-Structure Prediction

A small PyTorch learning project for residue-level protein secondary-structure
classification. The repository contains two experimental neural-network scripts and
the accompanying train/test text datasets.

## Files

- `Prediction of Protein.py`: embedding-based classifier with padded mini-batches.
- `2.py`: an earlier one-hot encoded baseline experiment.
- `protein-secondary-structure.train.txt`: training samples.
- `protein-secondary-structure.test.txt`: evaluation samples.

Both scripts resolve dataset paths relative to the repository, so no personal or
machine-specific path is required.

## Run

Install Python 3, PyTorch and NumPy, then run either experiment:

```bash
python "Prediction of Protein.py"
```

This is an educational experiment, not a production protein-structure predictor.
