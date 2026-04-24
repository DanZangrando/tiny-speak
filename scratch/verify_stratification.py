from training.audio_dataset import _compute_split_counts

ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

print("Verificando estratificación robusta:")
print("-" * 30)
for n in range(1, 11):
    counts = _compute_split_counts(n, ratios)
    print(f"Total: {n:2} samples | {counts} | Sum: {sum(counts.values())}")
    
    if n >= 2:
        assert counts["val"] >= 1, f"Error: N={n} no tiene val"
    if n >= 3:
        assert counts["test"] >= 1, f"Error: N={n} no tiene test"
    assert sum(counts.values()) == n, f"Error: Suma incorrecta para N={n}"

print("-" * 30)
print("¡Verificación exitosa! La estratificación garantiza al menos 1 muestra en val si N>=2.")
