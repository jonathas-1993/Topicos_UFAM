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

# Normalizar ratings para [0,1] para melhor convergência
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

# ========== 2. AUTOENCODER MELHORADO ==========
class AutoEncoder(nn.Module):
    def __init__(self, n_movies, k, dropout_p=0.1):
        super(AutoEncoder, self).__init__()
        # Arquitetura mais profunda para melhor capacidade
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
            nn.Sigmoid()  # Para saída [0,1]
        )
        
        # Inicialização Xavier melhorada
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ========== 3. TREINAMENTO OTIMIZADO ==========
# Hiperparâmetros ajustados para convergência mais rápida
learning_rate = 0.001  # Reduzido para estabilidade
lambda_reg = 0.0001    # Reduzido para menos regularização
patience = 100         # Mais paciência
batch_size = 32        # Batch menor para melhor convergência
num_epochs = 1000      # Mais épocas

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
    
    # Criar otimizador com configurações otimizadas
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
    
    # Learning rate scheduler
    if optimizer_type != "LBFGS":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, verbose=True
        )
    
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
                
                # Gradient clipping mais conservador
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            epoch_train_loss /= n_batches
        else:
            # LBFGS
            optimizer.step(closure)
            with torch.no_grad():
                outputs = model(train_tensor)
                epoch_train_loss = masked_mse_loss(outputs, train_tensor, train_mask).item()
        
        # Validação
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_tensor)
            val_loss = masked_mse_loss(val_outputs, val_tensor, val_mask).item()
        
        history["train_mse"].append(epoch_train_loss)
        history["val_mse"].append(val_loss)
        
        # Update scheduler
        if optimizer_type != "LBFGS":
            scheduler.step(val_loss)
        
        # Early stopping e melhor modelo
        if val_loss < best_val_mse:
            best_val_mse = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
        
        # Critério de parada: MSE < 0.1 ou paciência esgotada
        if best_val_mse < 0.1:
            print(f"🎯 MSE < 0.1 atingido na época {epoch}!")
            break
        elif patience_counter >= patience:
            print(f"⏰ Early stopping na época {epoch} (paciência esgotada)")
            break
        
        # Log a cada 50 épocas
        if epoch % 50 == 0:
            print(f"Época {epoch}: train_mse={epoch_train_loss:.4f}, val_mse={val_loss:.4f}, best_val_mse={best_val_mse:.4f}")
    
    training_time = time.time() - start_time
    
    # Carregar melhor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
    
    return model, best_val_mse, history, best_epoch, training_time

# ========== 4. EXPERIMENTOS ==========
k_values = [50, 75, 100]  # Conforme especificação
optimizers = ["SGD", "Adam", "RMSprop", "Adagrad", "AdamW", "LBFGS"]

results = []
results_models = {}
results_history = {}

results_path = r"C:\Users\Jonathas\Documents\UFAM 2025_2\Topicos Avancados em Aprendizado de Maquina e Otimizacao\Trabalho 1 - Movie Lens Small Latest Dataset\results"
os.makedirs(results_path, exist_ok=True)

for k in k_values:
    for opt in optimizers:
        print(f"\n{'='*60}")
        print(f"Treinando: k={k}, otimizador={opt}")
        print(f"{'='*60}")
        
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
        
        status = "🎯 MSE < 0.1 ATINGIDO!" if val_mse < 0.1 else "📊 Treinamento concluído"
        print(f"{status}: val_mse={val_mse:.4f}, best_epoch={best_epoch}, time={ttime:.1f}s")

# Salvar resultados
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(results_path, "summary_results.csv"), index=False)

# Mostrar tabela comparativa
print("\n" + "="*80)
print("=== TABELA COMPARATIVA DE OTIMIZADORES (MSE de Validação Final) ===")
print("="*80)
pivot_table = df_results.pivot(index="k", columns="optimizer", values="final_val_mse")
print(pivot_table.round(4))

# Destacar resultados que atingiram MSE < 0.1
successful_runs = df_results[df_results['final_val_mse'] < 0.1]
if not successful_runs.empty:
    print(f"\n🎯 CONFIGURAÇÕES QUE ATINGIRAM MSE < 0.1:")
    for _, row in successful_runs.iterrows():
        print(f"   k={row['k']}, {row['optimizer']}: MSE={row['final_val_mse']:.4f}")

# ========== 5. GRÁFICOS DE CONVERGÊNCIA ==========
print("\nGerando gráficos de convergência...")
for k in k_values:
    plt.figure(figsize=(14, 8))
    
    for opt in optimizers:
        key = (k, opt)
        if key in results_history:
            hist = results_history[key]
            epochs = list(range(1, len(hist["val_mse"]) + 1))
            val_mse_values = hist["val_mse"]
            
            plt.plot(epochs, val_mse_values, label=f"{opt} (final: {val_mse_values[-1]:.4f})", 
                    linewidth=2, marker='o', markersize=3)
    
    # Linha horizontal em MSE = 0.1
    plt.axhline(y=0.1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Critério MSE < 0.1')
    
    plt.title(f"Convergência do MSE de Validação — k = {k}", fontsize=16, fontweight='bold')
    plt.xlabel("Época", fontsize=12)
    plt.ylabel("MSE de Validação", fontsize=12)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.tight_layout()
    
    filename = f"convergence_k{k}.png"
    plt.savefig(os.path.join(results_path, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico salvo: {filename}")

# ========== 6. GERAR TABELA DE RECOMENDAÇÕES (MSE GLOBAL) ==========
def gerar_tabela_recomendacoes(model, user_movie_matrix, user_ids_especificos):
    """
    Gera a tabela com as 5 melhores recomendações para os usuários especificados
    usando o MSE global do modelo.
    """
    model.eval()
    with torch.no_grad():
        all_predictions = model(user_movie_tensor).cpu().numpy()
    
    # Máscara de filmes avaliados
    mask = user_movie_matrix > 0
    global_mse = ((all_predictions - user_movie_matrix) ** 2 * mask).sum() / mask.sum()
    global_mse = float(global_mse)
    
    resultados = []
    
    for user_id in user_ids_especificos:
        if user_id in user_id_to_idx:
            user_idx = user_id_to_idx[user_id]
            user_predictions = all_predictions[user_idx]
            
            # Desnormalizar predições para escala original [0.5, 5.0]
            user_predictions = user_predictions * 4.5 + 0.5
            
            # Encontrar filmes já avaliados pelo usuário
            rated_movies = set(np.where(user_movie_matrix[user_idx] > 0)[0])
            
            # Obter top 5 filmes não avaliados
            unrated_indices = [i for i in range(n_movies) if i not in rated_movies]
            if unrated_indices:
                unrated_predictions = user_predictions[unrated_indices]
                top5_indices = np.argsort(unrated_predictions)[::-1][:5]
                top5_movie_indices = [unrated_indices[i] for i in top5_indices]
                top5_movie_ids = [idx_to_movie_id[i] for i in top5_movie_indices]
                top5_titles = [movie_id_to_title.get(mid, f"Movie_{mid}") for mid in top5_movie_ids]
            else:
                top5_titles = ["N/A"] * 5
            
            resultados.append({
                'UserID': user_id,
                'Rec_1': top5_titles[0],
                'Rec_2': top5_titles[1],
                'Rec_3': top5_titles[2],
                'Rec_4': top5_titles[3],
                'Rec_5': top5_titles[4],
                'MSE': f"{global_mse:.4f}"
            })
        else:
            print(f"⚠️ UserID {user_id} não encontrado no dataset")
    
    return pd.DataFrame(resultados)

# Usuários específicos do trabalho
usuarios_especificos = [40, 92, 123, 245, 312, 460, 514, 590]

print("\n" + "="*80)
print("GERANDO TABELAS DE RECOMENDAÇÕES PARA OS USUÁRIOS ESPECIFICADOS")
print("="*80)

# Encontrar a MELHOR configuração geral (menor MSE)
best_result = df_results.nsmallest(1, 'final_val_mse').iloc[0]
best_k = int(best_result['k'])
best_opt = best_result['optimizer']
best_mse = best_result['final_val_mse']
best_model = results_models[(best_k, best_opt)]

print(f"\n🏆 MELHOR CONFIGURAÇÃO GERAL:")
print(f"   k={best_k}, {best_opt}: MSE={best_mse:.4f}")

if best_mse < 0.1:
    print("🎯 CRITÉRIO MSE < 0.1 ATINGIDO!")
else:
    print(f"📊 Melhor MSE alcançado: {best_mse:.4f} (não atingiu < 0.1)")

# Gerar tabela de recomendações da melhor configuração
print(f"\n--- RECOMENDAÇÕES DA MELHOR CONFIGURAÇÃO (k={best_k}, {best_opt}) ---")
tabela_final = gerar_tabela_recomendacoes(best_model, user_movie_matrix, usuarios_especificos)
print(tabela_final.to_string(index=False))

# Salvar tabela final
csv_filename = os.path.join(results_path, f"recomendacoes_MELHOR_k{best_k}_{best_opt}.csv")
tabela_final.to_csv(csv_filename, index=False)
print(f"\n✅ Tabela final salva: {csv_filename}")

# Gerar tabelas para cada k também
for k in k_values:
    df_k = df_results[df_results['k'] == k]
    if not df_k.empty:
        best_row = df_k.nsmallest(1, 'final_val_mse').iloc[0]
        best_opt_k = best_row['optimizer']
        best_model_k = results_models[(k, best_opt_k)]
        
        tabela_k = gerar_tabela_recomendacoes(best_model_k, user_movie_matrix, usuarios_especificos)
        csv_filename_k = os.path.join(results_path, f"recomendacoes_k{k}_{best_opt_k}.csv")
        tabela_k.to_csv(csv_filename_k, index=False)

print(f"\n🎯 Todas as tabelas de recomendação foram salvas em: {results_path}")
print("✅ Sistema executado! Verifique se alguma configuração atingiu MSE < 0.1")