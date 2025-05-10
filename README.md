# 🔬 Solubility Prediction with Pretrained AI Model

This project evaluates a **pretrained chemistry AI model** on the **Delaney solubility dataset**, predicting the water solubility of organic molecules. It uses the `ChemSolubilityBERTa` model from Hugging Face and standard cheminformatics tools like RDKit to assess how well AI can predict log-solubility values (log(mol/L)) for real compounds.

---

## 📁 Project Overview

The goal of this project is to:

- Use an **established AI model** trained on chemical data.
- Evaluate its **accuracy** on a trusted dataset.
- Generate visual and numeric results to understand performance.
- Interpret the results in a chemistry context.

---

## 🧪 Dataset

- **Name**: Delaney Solubility Dataset
- **Source**: [DeepChem](https://deepchemdata.s3.amazonaws.com/datasets/delaney-processed.csv)
- **Size**: 1128 molecules
- **Target property**: Measured log(solubility) in water (log(mol/L))

---

## 🤖 AI Model Used

- **Model**: [`khanfs/ChemSolubilityBERTa`](https://huggingface.co/khanfs/ChemSolubilityBERTa)
- **Type**: Transformer-based model (like BERT) trained on chemical SMILES strings
- **Task**: Predict log-solubility (log(mol/L)) from molecular structure

---

## 🧰 Tools and Libraries

| Tool | Purpose |
|------|---------|
| `RDKit` | Converts SMILES to molecular descriptors |
| `Transformers` (Hugging Face) | Loads the pretrained model |
| `scikit-learn` | Evaluation metrics like MSE, R² |
| `Matplotlib` | Visualization (scatter plots) |
| `Pandas` | Data manipulation and export |
| `NumPy` | Numerical operations |

---

## 📊 Evaluation Metrics

| Metric | Value | Meaning |
|--------|-------|---------|
| **MSE** | 0.6839 | Average squared error |
| **RMSE** | 0.8270 | Root mean squared error (more interpretable) |
| **MAE** | 0.6258 | Average absolute error |
| **R²** | 0.8443 | Explains ~84% of the variance in solubility data |

> Interpretation: The model performs well, with strong correlation between predicted and actual solubility values and average error less than 1 log unit.

---

## 📈 Output Files

- `solubility_prediction_results.png` – Scatter plot of predicted vs. actual solubility
- `solubility_prediction_results.csv` – Detailed table of predictions and errors for each molecule

---

## 🚀 How to Run

Make sure you have Python 3.7+ and install dependencies:

```bash
pip install -r requirements.txt
```
Then run the main script:

```bash
python solubility_predictor.py
```
