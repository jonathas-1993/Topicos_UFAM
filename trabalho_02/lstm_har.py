#==================================================================
# LSTM para Reconhecimento de Atividades Humanas (UniMib-SHAR)
# Autor: Jonathas Tavares Neves
# Data: Setembro/2025
# Descrição:
#    - Carrega dados do dataset UniMib-SHAR
#    - Pré-processa sinais de acelerômetro (3 eixos, 151 timesteps)
#    - Treina um modelo LSTM para classificação em 9 atividades
#==================================================================

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.initializers import GlorotUniform

# ======================================================
# 1. Carregamento dos dados
# ======================================================
data_path = r"C:\Users\Jonathas\Documents\UFAM 2025_2\Topicos Avancados em Aprendizado de Maquina e Otimizacao\Trabalho 2 - UniMibSHAR\9_labels_ADL\9_labels_ADL"

# Carregar dados
data_mat = sio.loadmat(f"{data_path}\\data.mat")
labels_mat = sio.loadmat(f"{data_path}\\labels.mat")

# Confirmar chaves
print("Chaves em data.mat:", list(data_mat.keys()))
print("Chaves em labels.mat:", list(labels_mat.keys()))

# Usar nomes reais das variáveis
X = data_mat['X']  # Shape: (453, 1724) -> precisa ser transposto
y = labels_mat['Y'].ravel()  # Shape: (1724,)

# Transpor X para (1724, 453)
X = X.T

# Ajustar labels: de [1..9] para [0..8]
y = y - 1

print("Formato original de X:", X.shape)
print("Formato ajustado de y:", y.shape)
print("Classes únicas em y:", np.unique(y))

# ======================================================
# 2. Pré-processamento
# ======================================================
# Cada linha tem 453 valores = 151 (x) + 151 (y) + 151 (z)
X = X.reshape(-1, 3, 151)       # (N, 3, 151)
X = np.transpose(X, (0, 2, 1))  # (N, 151, 3) => timesteps=151, features=3

# Normalização dos eixos
scaler = StandardScaler()
n_samples, n_timesteps, n_features = X.shape
X = X.reshape(-1, n_features)
X = scaler.fit_transform(X)
X = X.reshape(n_samples, n_timesteps, n_features)

# ======================================================
# 3. Divisão em treino e validação (85% / 15%)
# ======================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

print("Treino:", X_train.shape, y_train.shape)
print("Validação:", X_val.shape, y_val.shape)

# Calcular class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = np.clip(class_weights, 0.5, 5.0)
class_weights_dict = dict(enumerate(class_weights))

# ======================================================
# 4. Definição da Rede LSTM (OTIMIZADA)
# ======================================================
model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(151, 3), kernel_initializer=GlorotUniform()),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(128, kernel_initializer=GlorotUniform()),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_initializer=GlorotUniform()),
    BatchNormalization(),
    Dense(9, activation='softmax', kernel_initializer=GlorotUniform())
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

print(model.summary())

# ======================================================
# 5. Treinamento (COM PACIÊNCIA AUMENTADA)
# ======================================================
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,  # Aumentado
    batch_size=32,  # Reduzido
    callbacks=[es, lr_scheduler],
    class_weight=class_weights_dict,
    verbose=1
)

# ======================================================
# 6. Avaliação
# ======================================================
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n✅ Acurácia no conjunto de validação: {val_acc*100:.2f}%")

# Curvas de acurácia
plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'], label='Treino', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validação', linewidth=2)
plt.xlabel("Épocas", fontsize=12)
plt.ylabel("Acurácia", fontsize=12)
plt.legend(fontsize=12)
plt.title("Curva de Acurácia", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()

# Curvas de perda
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Treino', linewidth=2)
plt.plot(history.history['val_loss'], label='Validação', linewidth=2)
plt.xlabel("Épocas", fontsize=12)
plt.ylabel("Perda", fontsize=12)
plt.legend(fontsize=12)
plt.title("Curva de Perda", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()

# ======================================================
# 7. Matriz de confusão
# ======================================================
y_pred = model.predict(X_val, verbose=0).argmax(axis=1)

print("\n📊 Relatório de Classificação:")
print(classification_report(y_val, y_pred, digits=4))

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 12},
            xticklabels=[f"Atividade {i+1}" for i in range(9)],
            yticklabels=[f"Atividade {i+1}" for i in range(9)])
plt.xlabel("Predito", fontsize=12)
plt.ylabel("Real", fontsize=12)
plt.title("Matriz de Confusão", fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()