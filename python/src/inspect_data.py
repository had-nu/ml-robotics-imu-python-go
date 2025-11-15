with open('../../data/raw/UCI-HAR Dataset/test/X_test.txt', 'r') as f:
    linha1 = f.readline()
    linha2 = f.readline()

print(f"Linha 1 tem {len(linha1)} caracteres")
print(f"Primeiros 1000: [{linha1[:1000]}]")

valores = linha1.split()
print(f"\nNúmero de valores: {len(valores)}")
print(f"Primeiros 5: {valores[:5]}")
