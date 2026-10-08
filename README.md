# Polynomial Regression 

This repository contains the code and methodology for building polynomial regression models to solve two complex prediction tasks (Phase 1 and Phase 2). The solution employs penalised regression techniques (Ridge, Lasso, Elastic Net) and cross validation.

## Project Structure

- `code/train.py`: The main training script. Handles polynomial expansion, K-fold Cross Validation, model selection, hold-out evaluation, and test prediction generation.
- `code/config.py`: Configuration settings, hyper parameter grids, and file paths.
- `code/data_loader.py`: Handles CSV and the 80/20 splitting.

## How to Run

To execute the training pipeline and generate predictions, run the following commands from the root directory:

```bash
# Run Phase 1
python code/train.py --var 1 

# Run Phase 2
python code/train.py --var 2 
```

The script will automatically identify the optimal model, evaluate it on an untouched 20% hold-out set, and output the final test predictions to the respective `results/` folder as `BT2024162_pred_varX.csv`.
