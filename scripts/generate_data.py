import json
import random
from pathlib import Path

EMOTIONS = [
    'ecstasy', 'admiration', 'terror', 'amazement', 'grief', 'loathing',
    'rage', 'vigilance', 'joy', 'trust', 'fear', 'surprise', 'sadness',
    'disgust', 'anger', 'anticipation', 'serenity', 'acceptance',
    'apprehension', 'distraction', 'pensiveness', 'boredom', 'annoyance',
    'interest', 'optimism', 'love', 'submission', 'awe', 'disapproval',
    'remorse', 'contempt', 'aggressiveness'
]

INTENSITIES = ['mild', 'primary', 'intense']

TEMPLATES = {
    'joy': [
        "This is amazing! I'm so happy about this!",
        "What a wonderful day!",
        "I couldn't be more pleased with the results.",
        "This made my entire week!",
        "Absolutely delighted with how things turned out."
    ],
    'trust': [
        "I believe in you and your abilities.",
        "You can count on me to be there.",
        "I have complete faith in this team.",
        "Your honesty means everything to me.",
        "I trust your judgment on this matter."
    ],
    'fear': [
        "I'm really worried about what might happen.",
        "This situation makes me nervous.",
        "I'm afraid things could go wrong.",
        "What if we fail? I can't stop thinking about it.",
        "The uncertainty is terrifying."
    ],
    'surprise': [
        "Wow, I didn't see that coming at all!",
        "That's completely unexpected!",
        "I'm shocked by this news.",
        "What a surprise! I had no idea.",
        "This caught me completely off guard."
    ],
    'sadness': [
        "I'm feeling really down about this.",
        "It hurts to think about what we lost.",
        "I can't shake this feeling of grief.",
        "Everything feels heavy right now.",
        "I'm struggling to find joy in anything."
    ],
    'disgust': [
        "That's absolutely repulsive.",
        "I can't stand the thought of it.",
        "This makes me sick to my stomach.",
        "How can anyone tolerate this?",
        "I'm revolted by this behavior."
    ],
    'anger': [
        "I'm furious about what happened!",
        "This is unacceptable and I'm angry!",
        "They have no right to treat us this way!",
        "I'm seething with rage right now.",
        "This makes my blood boil!"
    ],
    'anticipation': [
        "I'm looking forward to what's next.",
        "Can't wait to see how this unfolds.",
        "There's so much potential here.",
        "I'm excited about the possibilities.",
        "The future looks promising."
    ],
    'serenity': [
        "I feel at peace with everything.",
        "Everything is calm and centered.",
        "There's a quiet contentment in this moment.",
        "I'm grateful for this tranquility.",
        "No worries, just pure calm."
    ],
    'acceptance': [
        "I've come to terms with this.",
        "This is how things are, and that's okay.",
        "I accept what I cannot change.",
        "Moving forward with understanding.",
        "Embracing reality as it is."
    ],
    'apprehension': [
        "I'm a bit uneasy about this.",
        "Something doesn't feel quite right.",
        "I have some concerns moving forward.",
        "Nervous but trying to stay positive.",
        "There's a nagging doubt in my mind."
    ],
    'distraction': [
        "My mind keeps wandering elsewhere.",
        "I can't seem to focus on anything.",
        "Too many thoughts competing for attention.",
        "Scattered and unable to concentrate.",
        "Everything feels disjointed today."
    ],
    'pensiveness': [
        "I've been reflecting deeply on this.",
        "There's a lot to contemplate here.",
        "Lost in thought about what this means.",
        "Mulling over the implications.",
        "Quiet contemplation leads to insight."
    ],
    'boredom': [
        "This is so tedious and uninteresting.",
        "I can't engage with this at all.",
        "Nothing holds my attention.",
        "Feeling completely disengaged.",
        "The monotony is overwhelming."
    ],
    'annoyance': [
        "This is really irritating.",
        "I'm getting frustrated with this.",
        "Why does this keep happening?",
        "It's such a minor thing but it bothers me.",
        "Grating on my nerves constantly."
    ],
    'interest': [
        "This is fascinating! Tell me more.",
        "I'm genuinely curious about this.",
        "What an intriguing concept.",
        "I want to dive deeper into this.",
        "Captivated by the details."
    ],
    'optimism': [
        "I'm confident things will work out.",
        "The outlook is positive.",
        "Hopeful for the best possible outcome.",
        "Seeing the silver lining in everything.",
        "Believing in a bright future."
    ],
    'love': [
        "I care about you more than words can say.",
        "My heart is full of affection.",
        "You mean the world to me.",
        "Deeply connected and devoted.",
        "Overwhelmed with tenderness."
    ],
    'submission': [
        "I'll defer to your expertise on this.",
        "Your guidance is appreciated.",
        "Following your lead on this one.",
        "Yielding to the group's decision.",
        "Accepting direction gracefully."
    ],
    'awe': [
        "This is absolutely breathtaking.",
        "I'm humbled by the magnitude of this.",
        "Stunned by the beauty and complexity.",
        "Overwhelmed with wonder.",
        "Transcendent experience beyond words."
    ],
    'disapproval': [
        "I don't agree with this approach.",
        "This falls short of expectations.",
        "Disappointed by this outcome.",
        "Cannot endorse this decision.",
        "Finding fault with the execution."
    ],
    'remorse': [
        "I deeply regret my actions.",
        "Wishing I could undo what I did.",
        "Guilt weighs heavily on me.",
        "Sorry doesn't begin to cover it.",
        "Haunted by my mistake."
    ],
    'contempt': [
        "This is beneath consideration.",
        "Their incompetence is staggering.",
        "Looking down on this pathetic display.",
        "Dismissive of their worthless input.",
        "Scoffing at the audacity."
    ],
    'aggressiveness': [
        "Back off or there will be consequences.",
        "I'm not backing down from this fight.",
        "Challenge me and see what happens.",
        "Ready to confront this head-on.",
        "Forceful in demanding resolution."
    ],
    'ecstasy': [
        "I'm over the moon with euphoria!",
        "Pure bliss coursing through me!",
        "Transcendent joy beyond comprehension!",
        "Ecstatic beyond all measure!",
        "Heavenly rapture fills my soul!"
    ],
    'admiration': [
        "I'm in awe of your accomplishments.",
        "Deeply impressed by your skill.",
        "Respect and admiration overflow.",
        "Your excellence inspires me.",
        "Reverent appreciation for your work."
    ],
    'terror': [
        "Absolute horror grips my heart!",
        "Paralyzed with sheer terror!",
        "Nightmare fuel, I can't escape!",
        "Dread beyond all rationality!",
        "Soul-crushing fear consumes me!"
    ],
    'amazement': [
        "Mind completely blown right now!",
        "Astonished beyond belief!",
        "Staggering revelation leaves me speechless!",
        "Wonderstruck by the impossible!",
        "Flabbergasted by this marvel!"
    ],
    'grief': [
        "Devastated by this profound loss.",
        "Heartbroken beyond repair.",
        "Mourning deeply and inconsolably.",
        "Agonizing pain of bereavement.",
        "Crushed by the weight of sorrow."
    ],
    'loathing': [
        "Utter hatred fills my being.",
        "Reviled with every fiber of my soul.",
        "Abhorrent and detestable.",
        "Visceral disgust and hatred combined.",
        "Despising with intense passion."
    ],
    'rage': [
        "Blinding fury overtakes me!",
        "Incandescent with white-hot rage!",
        "Uncontrollable wrath erupts!",
        "Frenzied anger beyond control!",
        "Apocalyptic fury consumes all!"
    ],
    'vigilance': [
        "Hyper-alert to any potential threat.",
        "Constantly scanning for danger.",
        "Guarded and watchful always.",
        "Defensive posture maintained.",
        "Ever-vigilant against betrayal."
    ]
}

SARCASM_TEMPLATES = {
    'joy': ["Oh great, another perfect day. Just what I needed. Not.", "Sure, this is 'amazing'. If amazing means terrible."],
    'trust': ["Yeah, I totally 'trust' this process. Said no one ever.", "Oh absolutely, faith is my middle name. Roll eyes."],
    'anger': ["Oh I'm not angry at all. Can't you tell?", "Totally fine. No rage here whatsoever."],
    'disapproval': ["Brilliant strategy. Really. Earth-shattering.", "What a brilliant idea. Truly groundbreaking work there."]
}

def generate_utterance(emotion, intensity='primary', sarcastic=False):
    templates = TEMPLATES.get(emotion, ["Generic statement here."])
    text = random.choice(templates)
    
    # Add intensity markers
    if intensity == 'mild':
        prefixes = ["Somewhat ", "A bit ", "Slightly "]
        text = random.choice(prefixes) + text.lower()
    elif intensity == 'intense':
        suffixes = ["!!!", " SO MUCH!", " Beyond words!", " EXTREMELY!"]
        text = text.upper() if random.random() > 0.5 else text + random.choice(suffixes)
    
    # Add sarcasm markers
    if sarcastic and emotion in SARCASM_TEMPLATES:
        text = random.choice(SARCASM_TEMPLATES[emotion])
    elif sarcastic:
        text = "Oh sure, because that's exactly what I wanted. /s"
    
    return text

def generate_dataset(num_samples=1200):
    data = []
    conversation_id = 1
    
    for i in range(num_samples):
        emotion = random.choice(EMOTIONS)
        intensity = random.choices(INTENSITIES, weights=[0.3, 0.5, 0.2])[0]
        sarcastic = random.random() < 0.15  # 15% sarcasm rate
        
        utterance = generate_utterance(emotion, intensity, sarcastic)
        
        sample = {
            'conversation_id': f'conv_{conversation_id}',
            'turn_number': (i % 8) + 1,
            'speaker': 'A' if i % 2 == 0 else 'B',
            'text': utterance,
            'emotion': emotion,
            'intensity': intensity,
            'sarcasm': sarcastic,
            'emotion_cause': f"Response to previous turn about {emotion}",
            'iaa_score': round(random.uniform(0.65, 0.95), 2),
            'scenario': random.choice(['workplace', 'romance', 'family', 'friendship', 'customer_service'])
        }
        
        data.append(sample)
        
        if (i + 1) % 8 == 0:
            conversation_id += 1
    
    return data

if __name__ == '__main__':
    print("Generating Plutchik ERC dataset...")
    dataset = generate_dataset(1200)
    
    output_path = Path('/workspace/plutchik_erc/data/train.jsonl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for sample in dataset:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Generated {len(dataset)} samples → {output_path}")
    
    # Show distribution
    from collections import Counter
    emotions = Counter(s['emotion'] for s in dataset)
    intensities = Counter(s['intensity'] for s in dataset)
    sarcasm_rate = sum(s['sarcasm'] for s in dataset) / len(dataset)
    
    print(f"\nEmotion distribution: {len(emotions)} classes covered")
    print(f"Intensity distribution: {dict(intensities)}")
    print(f"Sarcasm rate: {sarcasm_rate:.1%}")
