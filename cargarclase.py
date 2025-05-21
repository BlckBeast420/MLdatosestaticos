import numpy as np

# ✅ Esto permite cargar arrays de objetos como strings
etiquetas = np.load("etiquetas.npy", allow_pickle=True)
print(etiquetas.tolist())
