import json

file_path = '/home/daniel/Proyectos/tiny_speak/experiments/cf349304.json'

with open(file_path, 'r') as f:
    data = json.load(f)

languages = data['languages']
models = ['eyes', 'ears_phonemes', 'ears_words', 'speller', 'reader']

print("| Language | Model | Train Loss | Val Loss | Val Acc |")
print("|---|---|---|---|---|")

for lang in languages:
    for model in models:
        metrics_list = data['metrics'][lang].get(model, [])
        if not metrics_list:
            continue
            
        final_metric = metrics_list[-1]
        
        # Determine accuracy key
        acc_key = None
        for k in final_metric.keys():
            if 'acc' in k or 'top1' in k:
                if 'val' in k:
                    acc_key = k
                    break
        
        val_loss = final_metric.get('val_loss', -1)
        val_acc = final_metric.get(acc_key, -1)
        train_loss = final_metric.get('train_loss', -1)
        
        val_acc_str = f"{val_acc:.2f}%" if val_acc != -1 else "N/A"
        
        print(f"| {lang} | {model} | {train_loss:.4f} | {val_loss:.4f} | {val_acc_str} |")
