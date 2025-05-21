import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# 1. Cargar los datos
df = pd.read_csv('gestos.csv')

# 2. Separar características (X) y etiquetas (y)
X = df.drop('letra', axis=1).values.astype('float32')
y = df['letra'].values

# 3. Codificar las letras en números
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# 4. Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# 5. Crear el modelo
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(42,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(np.unique(y)), activation='softmax'))  # Salida = número de clases

# 6. Compilar y entrenar
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# 7. Guardar modelo y clases
model.save('modelo_gestos.h5')
np.save('clases.npy', encoder.classes_)

print("✅ Modelo entrenado y guardado como 'modelo_gestos.h5'")
