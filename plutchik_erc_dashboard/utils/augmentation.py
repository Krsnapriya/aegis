import pandas as pd
import json
import os

def extract_sarcasm_candidates(csv_path: str) -> pd.DataFrame:
    """Pull sarcastic samples from the training set as candidates for augmentation."""
    df = pd.read_csv(csv_path)
    # Ensure we only use training data for augmentation to prevent leakage
    df = df[(df['split'] == 'train') & (df['sarcasm_flag'] == True)]
    # We need emotion_cause to help the human write a literal twin
    df = df[df['emotion_cause'].notna()]
    return df.reset_index(drop=True)

def build_context_string(df: pd.DataFrame, dialogue_id: str, turn_id: int, window: int = 2) -> str:
    """Build context window string from CSV, mirroring preprocessing logic."""
    dlg = df[(df['dialogue_id'] == dialogue_id) & (df['turn_id'] < turn_id)].tail(window)
    turns = [f"{r['speaker']}: {r['text']}" for _, r in dlg.iterrows()]
    return ' | '.join(turns) if turns else '[NO_CONTEXT]'

def create_pair_template(row: pd.Series, df: pd.DataFrame) -> dict:
    """Create a JSON template for a contrastive pair."""
    context = build_context_string(df, row['dialogue_id'], row['turn_id'])
    return {
        'original_id': f"{row['dialogue_id']}_T{row['turn_id']}",
        'text': row['text'],
        'original_context': context,
        'twin_context': '',          # To be filled by human
        'original_emotion': row['emotion'],
        'twin_emotion': '',          # To be filled by human
        'dissonance_score': 1.0,     # Original sarcastic sample is always 1.0
        'scenario': row['scenario'],
        'human_verified': False,     # Gate for training
        'verifier_notes': ''
    }

def generate_templates(csv_path: str, out_path: str = 'data/processed/ERC/pair_templates.jsonl'):
    """Generate all possible templates for sarcastic samples in the dataset."""
    df = pd.read_csv(csv_path)
    candidates = extract_sarcasm_candidates(csv_path)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    templates = [create_pair_template(r, df) for _, r in candidates.iterrows()]
    
    with open(out_path, 'w') as f:
        for t in templates:
            f.write(json.dumps(t) + '\n')
            
    print(f"✅ Generated {len(templates)} templates → {out_path}")
    print("👉 Action: Fill 'twin_context' and 'twin_emotion' in the templates, then run pair_verifier.py")

if __name__ == "__main__":
    generate_templates('data/processed/ERC/plutchik_v2_production.csv')
