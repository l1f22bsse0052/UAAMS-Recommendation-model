import pandas as pd
import numpy as np
import re
import pickle
import os

ASSUMED_TEST_PCT = 0.50

def extract_weights(formula):
    """Extract matric, inter, and test weights from formula string"""
    if pd.isna(formula):
        return 0.17, 0.50, 0.33
    
    formula_str = str(formula)
    matric_match = re.search(r'matric\s*/\s*matricTotal\)\s*\*\s*([\d.]+)', formula_str)
    inter_match = re.search(r'inter\s*/\s*interTotal\)\s*\*\s*([\d.]+)', formula_str)
    test_match = re.search(r'test\s*/\s*400\)\s*\*\s*([\d.]+)', formula_str)
    
    w_matric = float(matric_match.group(1)) / 100 if matric_match else 0.17
    w_inter = float(inter_match.group(1)) / 100 if inter_match else 0.50
    w_test = float(test_match.group(1)) / 100 if test_match else 0.33
    
    return w_matric, w_inter, w_test

def estimate_marks(aggr, w_matric, w_fsc, w_test, random_seed=None):
    """Estimate required Matric% and FSc% to achieve AGGR"""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    test_contribution = ASSUMED_TEST_PCT * w_test * 100
    remaining = aggr - test_contribution
    
    if remaining <= 0:
        return 50.0, 50.0
    
    total_weight = w_matric + w_fsc
    if total_weight > 0:
        estimated_score = remaining / total_weight
        matric_est = estimated_score * np.random.uniform(0.95, 1.05)
        fsc_est = (remaining - (w_matric * matric_est)) / w_fsc if w_fsc > 0 else matric_est
    else:
        matric_est = fsc_est = 50.0
    
    matric_est = np.clip(matric_est, 30, 100)
    fsc_est = np.clip(fsc_est, 30, 100)
    
    return matric_est, fsc_est

def main():
    print("="*60)
    print("🎓 TRAINING PROGRAM RECOMMENDER MODEL")
    print("="*60)
    
    # Load CSV
    csv_path = "/home/zar/program_recommender/data/data.csv"
    print(f"📂 Loading data from {csv_path}...")
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found at {csv_path}")
        print("Please ensure your CSV file is in the data folder")
        return False
    
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['AGGR', 'CAMPUS|PROGRAM'])
    df = df[df['AGGR'] > 0]
    df = df.reset_index(drop=True)
    
    print(f"✅ Loaded {len(df)} programs")
    
    # Extract weights
    print("🔍 Extracting formula weights...")
    df['W_Matric'], df['W_FSc'], df['W_Test'] = zip(*df['FORMULA'].apply(extract_weights))
    
    # Estimate marks
    print("📊 Estimating Matric/FSc scores...")
    np.random.seed(42)
    estimates = []
    for idx, row in df.iterrows():
        matric_est, fsc_est = estimate_marks(
            row['AGGR'], row['W_Matric'], row['W_FSc'], row['W_Test'], idx
        )
        estimates.append((matric_est, fsc_est))
    
    df['Estimated_Matric_Pct'] = [e[0] for e in estimates]
    df['Estimated_FSc_Pct'] = [e[1] for e in estimates]
    
    # Calculate average weights
    avg_weights = {
        'matric': df['W_Matric'].mean(),
        'fsc': df['W_FSc'].mean(),
        'test': df['W_Test'].mean()
    }
    
    # Prepare model data
    model_data = {
        'programs': df[['CAMPUS|PROGRAM', 'AGGR', 'Estimated_Matric_Pct', 
                        'Estimated_FSc_Pct', 'FORMULA', 'SOURCE_LINK', 
                        'W_Matric', 'W_FSc', 'W_Test']].to_dict('records'),
        'avg_weights': avg_weights,
        'metadata': {
            'total_programs': len(df),
            'assumed_test_pct': ASSUMED_TEST_PCT,
            'features': ['Matric_Pct', 'FSc_Pct'],
            'version': '1.0'
        }
    }
    
    # Save model
    os.makedirs('../models', exist_ok=True)
    model_path = '/home/zar/program_recommender/models/program_recommender_id.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ Model saved to {model_path}")
    print(f"📊 Model contains {len(df)} programs")
    print(f"⚖️  Average weights - Matric: {avg_weights['matric']:.2f}, FSc: {avg_weights['fsc']:.2f}")
    
    # Show sample
    print("\n📋 Sample of prepared data:")
    print(df[['CAMPUS|PROGRAM', 'AGGR', 'Estimated_Matric_Pct', 'Estimated_FSc_Pct']].head(5).to_string(index=False))
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Model training completed successfully!")
    else:
        print("\n❌ Model training failed!")
