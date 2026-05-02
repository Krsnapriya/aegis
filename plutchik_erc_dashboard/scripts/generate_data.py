"""Fast data generator for Plutchik ERC."""
import csv, random
from datetime import datetime
from pathlib import Path

EMOTIONS = ["sere","acce","dist","pens","bore","inte","anno","opti","joy","trus","fear","surp","sadn","disg","ange","appr","ecst","admi","terr","amaz","grie","loat","rage","vigi","love","subm","awe","disa","remo","cont","aggr","supr"]
SCENARIOS = ["workplace","romance","family","friendship","customer_service","healthcare","education","online_debate"]
SPEAKERS = ["user","agent","partner","friend","stranger"]

TEMPLATES = {
    "joy": ["This is amazing!", "So happy about this!", "Best day ever!"],
    "ecst": ["ABSOLUTELY THRILLED!", "BEST DAY OF MY LIFE!", "OVER THE MOON!"],
    "trust": ["I can count on you.", "Your support means everything.", "I believe in us."],
    "fear": ["I'm worried something might go wrong.", "This makes me nervous.", "Afraid of consequences."],
    "terr": ["ABSOLUTELY TERRIFIED!", "PARALYZED WITH FEAR!", "NIGHTMARE COME TRUE!"],
    "sadness": ["Feeling down about this.", "Makes me sad.", "Wish things were different."],
    "grie": ["DEVASTATED by this loss.", "HEARTBROKEN.", "Drowning in sorrow."],
    "anger": ["This makes me angry.", "Not acceptable.", "Really frustrated."],
    "rage": ["BLIND RAGE!", "ABSOLUTELY FURIOUS!", "SEEING RED!"],
    "disgust": ["That's repulsive.", "Makes me sick.", "Can't stand this."],
    "loat": ["ABSOLUTELY DESPISE THIS!", "REVOLTED!", "HATE EVERY FIBER!"],
    "surprise": ["Didn't expect that!", "Caught me off guard.", "Wow, surprising!"],
    "amaz": ["MIND-BLOWING!", "COMPLETELY STUNNED!", "ASTONISHING!"],
    "anticipation": ["Looking forward to this.", "Eager to start.", "Ready for what's next."],
    "vigi": ["On high alert.", "HYPERVIGILANT.", "Scanning for threats."],
    "serenity": ["Calm and peaceful.", "At ease.", "Inner peace."],
    "acceptance": ["Can live with this.", "It is what it is.", "Okay with it."],
    "apprehension": ["A bit nervous.", "Hesitant.", "Some doubt."],
    "distraction": ["Mind wandering.", "Hard to focus.", "Distracted."],
    "boredom": ["Pretty boring.", "Not engaging.", "Meh, whatever."],
    "interest": ["Intriguing.", "Want to know more.", "Captivating."],
    "annoyance": ["Bit irritating.", "Slightly bothersome.", "Minor inconvenience."],
    "optimism": ["Bright side.", "Hopeful.", "Positive outcome."],
    "love": ["Care deeply.", "Mean so much.", "Full of affection."],
    "submission": ["Defer to you.", "You know best.", "Follow your lead."],
    "awe": ["Profoundly moving.", "Overwhelmed.", "Filled with wonder."],
    "disapproval": ["Don't agree.", "Seems wrong.", "Have reservations."],
    "remorse": ["Regret this.", "Feeling guilty.", "Wish I could undo."],
    "contempt": ["Beneath consideration.", "Dismissive.", "View with disdain."],
    "aggressiveness": ["Push back hard.", "Confrontational.", "Combative."],
    "supr": ["Unexpected!", "What a surprise!", "Didn't see that coming!"],
}

SARCASM = {
    "joy": ["Oh great, another meeting.", "Fantastic, more work."],
    "anger": ["Oh I'm not mad at all.", "Perfect, nothing wrong here."],
    "sadness": ["Oh wonderful, love being ignored.", "Great, another disappointment."],
    "disgust": ["Mmm yes, love cleaning messes.", "Delicious, another bug."],
}

def gen():
    samples = []
    for emo in EMOTIONS:
        for _ in range(47):
            text = random.choice(TEMPLATES.get(emo, TEMPLATES["joy"]))
            sarc = 0
            if emo in SARCASM and random.random() < 0.35:
                text = random.choice(SARCASM[emo])
                sarc = 1
            
            intense = ["ecst","admi","terr","amaz","grie","loat","rage","vigi"]
            mild = ["sere","acce","dist","pens","bore","inte","anno","opti","love","subm","awe","disa","remo","cont","aggr"]
            ring = 3 if emo in intense else (1 if emo in mild else 2)
            
            iaa = random.uniform(0.65, 0.95) * (0.85 if ring == 1 else 1.0)
            
            samples.append({
                "text": text,
                "emotion_code": emo,
                "scenario": random.choice(SCENARIOS),
                "speaker_role": random.choice(SPEAKERS),
                "sarcasm": sarc,
                "ring_level": ring,
                "iaa_score": round(iaa, 3),
                "emotion_cause": f"Reaction to {random.choice(SCENARIOS)}",
                "conversation_id": f"conv_{random.randint(1000,9999)}",
                "turn_number": random.randint(1, 12),
                "timestamp": datetime.now().isoformat()
            })
    
    random.shuffle(samples)
    
    out = Path("plutchik_erc_dashboard/data/plutchik_dataset.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["text","emotion_code","scenario","speaker_role","sarcasm","ring_level","iaa_score","emotion_cause","conversation_id","turn_number","timestamp"])
        w.writeheader()
        w.writerows(samples)
    
    print(f"✅ Generated {len(samples)} samples to {out}")
    return samples

if __name__ == "__main__":
    gen()
