import json
import os
import sys

# Add parent dir to path to import constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import PLUTCHIK

VALIDATION_RULES = [
    ('text_length',      lambda p: len(p['text'].split()) >= 6,
                         'Text must be ≥ 6 words'),
    ('twin_context_set', lambda p: len(p['twin_context'].strip()) >= 20,
                         'twin_context must be ≥ 20 chars'),
    ('twin_emotion_set', lambda p: p['twin_emotion'] in PLUTCHIK,
                         'twin_emotion must be a valid Plutchik emotion'),
    ('emotion_differs',  lambda p: p['twin_emotion'] != p['original_emotion'],
                         'twin_emotion must differ from original_emotion'),
    ('contexts_differ',  lambda p: p['twin_context'] != p['original_context'],
                         'twin_context must differ from original_context'),
    ('dissonance_orig',  lambda p: p['dissonance_score'] == 1.0,
                         'original dissonance_score must be 1.0'),
]

def verify_pair(pair: dict) -> tuple[bool, list[str]]:
    """Check a pair against all validation rules."""
    failures = []
    for name, rule, msg in VALIDATION_RULES:
        if not rule(pair):
            failures.append(f"{name}: {msg}")
    return len(failures) == 0, failures

def run_verifier(template_path='data/processed/ERC/pair_templates.jsonl', 
                 output_path='data/processed/ERC/contrastive_pairs.jsonl',
                 auto_verify=False):
    """Interactive CLI for human verification of contrastive pairs."""
    if not os.path.exists(template_path):
        print(f"❌ Error: Template file {template_path} not found.")
        return

    with open(template_path, 'r') as f:
        templates = [json.loads(l) for l in f]

    unverified = [t for t in templates if not t.get('human_verified', False)]
    print(f"🔍 {len(unverified)} pairs awaiting verification\n")

    for pair in unverified:
        print("="*60)
        print(f"TEXT:     {pair['text']}")
        print(f"ORIGINAL: {pair['original_context']}")
        print(f"TWIN CTX: {pair['twin_context']}")
        print(f"ORIG EMO: {pair['original_emotion']} → TWIN EMO: {pair['twin_emotion']}")
        print("-"*60)

        ok, failures = verify_pair(pair)
        if not ok:
            print(f"⚠️  VALIDATION FAILED: {failures}")
            print("Please fix the template file and re-run.\n")
            continue

        # Human gating
        if auto_verify:
            print("🤖 Auto-verifying (skipping human gate)...")
            q1, q2, q3 = 'y', 'y', 'y'
        else:
            print("🤔 Three-question human gate (Must pass all):")
            q1 = input("1. Does twin_context genuinely make the text read as literal? [y/n]: ").lower()
            q2 = input("2. Is twin_emotion plausible (not just the opposite)? [y/n]: ").lower()
            q3 = input("3. Would a human annotator agree without coaching? [y/n]: ").lower()
        
        if q1 == 'y' and q2 == 'y' and q3 == 'y':
            pair['human_verified'] = True
            # Create the twin sample (label 0.0)
            twin = pair.copy()
            twin['dissonance_score'] = 0.0
            twin['original_id'] = pair['original_id'] + '_twin'
            
            with open(output_path, 'a') as f:
                f.write(json.dumps(pair) + '\n')
                f.write(json.dumps(twin) + '\n')
            
            print("✅ Pair + twin appended to contrastive_pairs.jsonl\n")
        else:
            print("❌ Rejected — please revise the twin_context and re-run.\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="Skip human verification")
    args = parser.parse_args()
    
    run_verifier(auto_verify=args.auto)
