
from training.visual_dataset import _stratified_split

# Mock samples
samples = {
    "n=1": [{"file_path": "p"}] * 1,
    "n=2": [{"file_path": "p"}] * 2,
    "n=3": [{"file_path": "p"}] * 3,
    "n=10": [{"file_path": "p"}] * 10,
}

print("Testing _stratified_split logic...")
splits = _stratified_split(samples)

counts = {}
for split_name, items in splits.items():
    for label, _ in items:
        if label not in counts: counts[label] = {}
        counts[label][split_name] = counts[label].get(split_name, 0) + 1

print(f"{'Label':<10} | {'Total':<5} | {'Train':<5} | {'Val':<5} | {'Test':<5}")
print("-" * 40)
for label in sorted(samples.keys()):
    total = len(samples[label])
    train = counts.get(label, {}).get("train", 0)
    val = counts.get(label, {}).get("val", 0)
    test = counts.get(label, {}).get("test", 0)
    print(f"{label:<10} | {total:<5} | {train:<5} | {val:<5} | {test:<5}")
