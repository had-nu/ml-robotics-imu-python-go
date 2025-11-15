print("🔵 Este print SEMPRE executa")

def somar(a, b):
	return a + b

if __name__ == '__main__':
	print("🟢 Este print SÓ executa se correr direto")
	resultado = somar(5, 3)
	print(f"Resultado: {resultado}")