import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings

warnings.filterwarnings('ignore')

def load_dataset():
    url = "https://deepchemdata.s3.amazonaws.com/datasets/delaney-processed.csv"
    print(f"Loading dataset from {url}…")
    data = pd.read_csv(url)
    print(f"Loaded {len(data)} molecules; data shape: {data.shape}")
    smiles_list = data['smiles'].tolist()
    y_true = data['measured log solubility in mols per litre'].values
    return smiles_list, y_true

def load_model():
    model_name = "khanfs/ChemSolubilityBERTa"
    print(f"Loading model '{model_name}' from Hugging Face…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print("Model and tokenizer loaded.")
    return tokenizer, model

def predict_solubility(smiles_list, tokenizer, model):
    print("Predicting solubility values…")
    y_pred = []
    batch_size = 32
    model.eval()
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i+batch_size]
            inputs = tokenizer(batch_smiles, return_tensors='pt', padding=True, truncation=True)
            outputs = model(**inputs)
            predictions = outputs.logits.squeeze().cpu().numpy()
            if predictions.ndim == 0:
                predictions = [predictions.item()]
            y_pred.extend(predictions)
    print("Prediction completed.")
    return np.array(y_pred)

def evaluate_model(y_true, y_pred, smiles_list):
    print("\nEvaluating model…")
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("Model performance:")
    print(f"  • MSE : {mse:.4f}")
    print(f"  • RMSE: {rmse:.4f}")
    print(f"  • MAE : {mae:.4f}")
    print(f"  • R²  : {r2:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--')
    plt.xlabel("Actual Solubility (log mol/L)")
    plt.ylabel("Predicted Solubility (log mol/L)")
    plt.title("Actual vs Predicted Solubility")
    plt.tight_layout()
    plt.savefig("solubility_prediction_results.png")
    print("Scatter plot saved to solubility_prediction_results.png")

    df = pd.DataFrame({
        "SMILES": smiles_list,
        "Actual": y_true,
        "Predicted": y_pred,
        "Error": y_true - y_pred
    })
    df["AbsError"] = df["Error"].abs()
    df.to_csv("solubility_prediction_results.csv", index=False)
    print("Detailed results saved to solubility_prediction_results.csv")

    worst = df.sort_values("AbsError", ascending=False).head(5)
    print("\nTop 5 molecules by absolute error:")
    for _, row in worst.iterrows():
        print(f"  • {row.SMILES} | Actual={row.Actual:.2f}, Pred={row.Predicted:.2f}, Err={row.Error:.2f}")

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "predictions": y_pred}

def main():
    print("="*50)
    print("Chemistry Model Evaluation")
    print("="*50)
    smiles_list, y_true = load_dataset()
    tokenizer, model = load_model()
    y_pred = predict_solubility(smiles_list, tokenizer, model)
    results = evaluate_model(y_true, y_pred, smiles_list)
    print("\nSummary:")
    print(f"  • RMSE: {results['rmse']:.4f}")
    print(f"  • R²  : {results['r2']:.4f}")
    print("="*50)
    return results

if __name__ == "__main__":
    main()