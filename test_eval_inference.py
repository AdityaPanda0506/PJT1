import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from streamlit_app import load_nssg_dimnet, analyze_full_sentence

p, err = load_nssg_dimnet()
if err:
    print(f"Error loading: {err}")
    exit(1)

test_sentences = [
    "khaana bahut swaad tha lekin service bilkul slow thi",
    "Staff bahut helpful tha lekin ambience thoda noisy tha",
    "Price reasonable hai aur quality top notch hai"
]

print("\n" + "="*80)
print("MULTI-ASPECT DIMENSIONAL INFERENCE RESULTS (STREAMLIT ENGINE)")
print("="*80)

for sent in test_sentences:
    print(f"\nSentence: \"{sent}\"")
    results = analyze_full_sentence(p, sent)
    for r in results:
        print(f"  -> Aspect: '{r['aspect']}' | Opinion: '{r['opinion']}'")
        print(f"     Valence: {r['valence']:.4f} | Arousal: {r['arousal']:.4f}")
        print(f"     Emotion: {r['emotion_name']} {r['emoji']} ({r['quadrant']}) | Intensity: {r['intensity_percent']}%")
print("="*80)
