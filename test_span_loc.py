from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
sentence = "khaana bahut swaad tha lekin service bilkul slow thi aur staff ka behaviour rude tha"

def locate_span(sentence, phrase, tokenizer):
    phrase_clean = phrase.strip().lower()
    sent_lower = sentence.lower()
    char_start = sent_lower.find(phrase_clean)
    if char_start == -1:
        words = phrase_clean.split()
        if words:
            char_start = sent_lower.find(words[0])
            char_end = char_start + len(words[0]) if char_start != -1 else 0
        else:
            return 0, 0
    else:
        char_end = char_start + len(phrase_clean)
        
    encoding = tokenizer(sentence, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    
    start_token = None
    end_token = None
    
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            continue
        if start_token is None and e > char_start:
            start_token = i
        if start_token is not None and s < char_end:
            end_token = i
            
    if start_token is None:
        start_token = 0
    if end_token is None:
        end_token = start_token
        
    return start_token, end_token

tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence)["input_ids"])
tokens_clean = [t.replace("\u2581", "_") for t in tokens]
print("Tokens:", list(enumerate(tokens_clean)))


for p in ["khaana", "bahut swaad tha", "service", "bilkul slow thi", "staff", "rude tha"]:
    s, e = locate_span(sentence, p, tokenizer)
    matched = [t.replace("\u2581", "_") for t in tokens[s:e+1]]
    print(f"Phrase: '{p}' -> Span: ({s}, {e}) -> Tokens: {matched}")

