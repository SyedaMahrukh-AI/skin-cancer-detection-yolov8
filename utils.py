import os
import pandas as pd

def print_final_metrics():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_csv = os.path.join(BASE_DIR, "results", "results.csv")
    
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        df.columns = [c.strip() for c in df.columns]
        final_row = df.iloc[-1]
        
        print(f"\n--- Final Performance ---")
        print(f"Precision: {final_row['metrics/precision(B)']:.4f}")
        print(f"Recall:    {final_row['metrics/recall(B)']:.4f}")
        print(f"mAP50:     {final_row['metrics/mAP50(B)']:.4f}")
    else:
        print(f"Error: results.csv not found at: {results_csv}")

def list_class_names():
    return ["Melanoma", "Nevus", "BCC", "AK", "BKL", "DF", "VASC"]