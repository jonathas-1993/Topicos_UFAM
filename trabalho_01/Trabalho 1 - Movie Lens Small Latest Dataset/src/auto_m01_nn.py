##CÓDIGO NÃO NORMALIZADO##

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import matplotlib.pyplot as plt

# ========== 1. CARREGAR DADOS ==========
data_path = r"C:\Users\Jonathas\Documents\UFAM 2025_2\Topicos Avancados em Aprendizado de Maquina e Otimizacao\Trabalho 1 - Movie Lens Small Latest Dataset\archive"
ratings_file = os.path.join(data_path, "ratings.csv")
movies_file = os.path.join(data_path, "movies.csv")

ratings = pd.read_csv(ratings_file)
movies = pd.read_csv(movies_file)

# Normalizar ratings para [0,1]
print("Normalizando ratings para [0,1]...")
ratings['rating'] = (ratings['rating'] - 0.5) / 4.5  # De [0.5,5] para [0,1]

movie_ids_in_ratings = sorted(ratings['movieId'].unique())
n_movies = len(movie_ids_in_ratings)
movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_ids_in_ratings)}
idx_to_movie_id = {idx: mid for mid, idx in movie_id_to_idx.items()}
movie_id_to_title = dict(zip(movies['movieId'], movies['title']))

user_ids = sorted(ratings['userId'].unique())
n_users = len(user_ids)
user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
idx_to_user_id = {idx: uid for uid, idx in user_id_to_idx.items()}

user_movie_matrix = np.zeros((n_users, n_movies), dtype=np.float32)
for r in ratings.itertuples(index=False):
    u = user_id_to_idx[r.userId]
    if r.movieId in movie_id_to_idx:
        m = movie_id_to_idx[r.movieId]
        user_movie_matrix[u, m] = r.rating

print(f"Matriz criada: {n_users} usuários x {n_movies} filmes")

# ========== 2. AUTOENCODER ==========
class AutoEncoder(nn.Module):
    def __init__(self, n_movies, k, dropout_p=0.1):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_movies, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, k)
        )
        self.decoder = nn.Sequential(
            nn.Linear(k, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, n_movies),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ========== 3. TREINAMENTO ==========
learning_rate = 0.001
lambda_reg = 0.0001
patience = 100
batch_size = 32
num_epochs = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando device: {device}")

user_movie_tensor = torch.from_numpy(user_movie_matrix).to(device)
train_size = int(0.8 * n_users)
train_tensor = user_movie_tensor[:train_size]
val_tensor = user_movie_tensor[train_size:]

def masked_mse_loss(pred, true, mask):
    diff = (pred - true) * mask
    denom = mask.sum()
    return torch.sum(diff ** 2) / denom if denom.item() > 0 else torch.tensor(0.0, device=pred.device)

def treinar_autoencoder(k, optimizer_type="RMSprop", lr=learning_rate, weight_decay=lambda_reg):
    model = AutoEncoder(n_movies, k).to(device)
    train_mask = (train_tensor > 0).float()
    val_mask = (val_tensor > 0).float()
    n_train = train_tensor.size(0)
    n_batches = max(1, n_train // batch_size)

    if optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8)
    elif optimizer_type == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, alpha=0.9, eps=1e-8)
    elif optimizer_type == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "LBFGS":
        def closure():
            optimizer.zero_grad()
            outputs = model(train_tensor)
            loss = masked_mse_loss(outputs, train_tensor, train_mask)
            loss.backward()
            return loss
        optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=20)

    if optimizer_type != "LBFGS":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)

    best_val_mse = float("inf")
    patience_counter = 0
    history = {"train_mse": [], "val_mse": []}
    best_epoch = 0
    best_model_state = None
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        if optimizer_type != "LBFGS":
            perm = torch.randperm(n_train)
            epoch_train_loss = 0.0
            for b in range(n_batches):
                idxs = perm[b*batch_size : (b+1)*batch_size]
                batch_x = train_tensor[idxs]
                batch_mask = train_mask[idxs]
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = masked_mse_loss(outputs, batch_x, batch_mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_train_loss += loss.item()
            epoch_train_loss /= n_batches
        else:
            optimizer.step(closure)
            with torch.no_grad():
                outputs = model(train_tensor)
                epoch_train_loss = masked_mse_loss(outputs, train_tensor, train_mask).item()

        model.eval()
        with torch.no_grad():
            val_outputs = model(val_tensor)
            val_loss = masked_mse_loss(val_outputs, val_tensor, val_mask).item()

        history["train_mse"].append(epoch_train_loss)
        history["val_mse"].append(val_loss)

        if optimizer_type != "LBFGS":
            scheduler.step(val_loss)

        if val_loss < best_val_mse:
            best_val_mse = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1

        if best_val_mse < 0.1:
            print(f"🎯 MSE < 0.1 atingido na época {epoch}!")
            break
        elif patience_counter >= patience:
            print(f"⏰ Early stopping na época {epoch}")
            break

    training_time = time.time() - start_time
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
    return model, best_val_mse, history, best_epoch, training_time

# ========== 4. EXPERIMENTOS ==========
k_values = [50, 75, 100]
optimizers = ["SGD", "Adam", "RMSprop", "Adagrad", "AdamW", "LBFGS"]

results = []
results_models = {}
results_history = {}
results_path = r"C:\Users\Jonathas\Documents\UFAM 2025_2\Topicos Avancados em Aprendizado de Maquina e Otimizacao\Trabalho 1 - Movie Lens Small Latest Dataset\results"
os.makedirs(results_path, exist_ok=True)

for k in k_values:
    for opt in optimizers:
        print(f"\nTreinando: k={k}, otimizador={opt}")
        model, val_mse, hist, best_epoch, ttime = treinar_autoencoder(k, opt)
        results.append({
            "k": k,
            "optimizer": opt,
            "final_val_mse": val_mse,
            "best_epoch": best_epoch,
            "train_time_s": ttime
        })
        results_models[(k, opt)] = model
        results_history[(k, opt)] = hist

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(results_path, "summary_results.csv"), index=False)

# ========== 5. RECOMENDAÇÕES ==========
def gerar_tabela_recomendacoes(model, user_movie_matrix, user_ids_especificos):
    model.eval()
    with torch.no_grad():
        all_predictions = model(user_movie_tensor).cpu().numpy()
    resultados = []
    for user_id in user_ids_especificos:
        if user_id in user_id_to_idx:
            user_idx = user_id_to_idx[user_id]
            user_original_ratings = user_movie_matrix[user_idx]
            user_predictions = all_predictions[user_idx]
            user_predictions = user_predictions * 4.5 + 0.5
            rated_movies = set(np.where(user_original_ratings > 0)[0])
            unrated_indices = [i for i in range(n_movies) if i not in rated_movies]
            if unrated_indices:
                unrated_predictions = user_predictions[unrated_indices]
                top5_indices = np.argsort(unrated_predictions)[::-1][:5]
                top5_movie_indices = [unrated_indices[i] for i in top5_indices]
                top5_movie_ids = [idx_to_movie_id[i] for i in top5_movie_indices]
                top5_titles = [movie_id_to_title[mid] if mid in movie_id_to_title else f"Movie_{mid}" for mid in top5_movie_ids]
            else:
                top5_titles = ["N/A"] * 5
            rated_indices = list(rated_movies)
            if rated_indices:
                true_ratings = user_original_ratings[rated_indices] * 4.5 + 0.5
                pred_ratings = user_predictions[rated_indices]
                mse = np.mean((true_ratings - pred_ratings) ** 2)
            else:
                mse = float('nan')
            resultados.append({
                'UserID': user_id,
                'Rec_1': top5_titles[0],
                'Rec_2': top5_titles[1],
                'Rec_3': top5_titles[2],
                'Rec_4': top5_titles[3],
                'Rec_5': top5_titles[4],
                'MSE': f"{mse:.4f}" if not np.isnan(mse) else "N/A"
            })
    return pd.DataFrame(resultados)

usuarios_especificos = [40, 92, 123, 245, 312, 460, 514, 590]
best_result = df_results.nsmallest(1, 'final_val_mse').iloc[0]
best_k = int(best_result['k'])
best_opt = best_result['optimizer']
best_model = results_models[(best_k, best_opt)]
tabela_final = gerar_tabela_recomendacoes(best_model, user_movie_matrix, usuarios_especificos)
print(tabela_final.to_string(index=False))
csv_filename = os.path.join(results_path, f"recomendacoes_MELHOR_k{best_k}_{best_opt}.csv")
tabela_final.to_csv(csv_filename, index=False)
print(f"Tabela final salva: {csv_filename}")
