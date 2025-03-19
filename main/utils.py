import torch
import torch.nn as nn
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import random
import os
import numpy as np
from geopy.distance import geodesic
import pandas as pd
import math 

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

import networkx as nx
from networkx.classes.function import degree
from networkx.linalg.laplacianmatrix import normalized_laplacian_matrix


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed):
    """It sets all the seeds for reproducibility.

    Args:
    ----------
    seed : int
        Seed for all the methods
    """
    print("Setting seeds")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize(inputs):
    normalized = []
    for eq in inputs:
        maks = np.max(np.abs(eq))
        if maks != 0:
            normalized.append(eq/maks)
        else:
            normalized.append(eq)
    return np.array(normalized)

def targets_to_list(targets):
    targets = targets.transpose(2,0,1)
    targetList = []
    for i in range(0, len(targets)):
        targetList.append(targets[i,:,:])
    return targetList

class CustomDataset(Dataset):
    def __init__(self, inputs, targets, graph_input, graph_features, metadata,transform=None):
        self.inputs = inputs
        self.targets = targets
        self.graph_input = graph_input
        self.graph_features = graph_features
        self.transform = transform
        self.metadata = metadata

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        target_data = self.targets[idx]
        graph_input = self.graph_input[idx]
        graph_features = self.graph_features[idx]
        metadata = self.metadata[idx]

        input_data = torch.tensor(input_data, dtype=torch.float)
        
        if self.transform:
            input_data = self.transform(input_data)
            target_data = self.transform(target_data)
            graph_input = self.transform(graph_input)
            graph_features = self.transform(graph_features)
            metadata = self.transform(metadata)

        return input_data, target_data, graph_input, graph_features, metadata
    
def calculate_metrics(outputs, targets):
    # print('outputs:',outputs, outputs.shape)
    # print('targets:',targets, targets.shape)
    mse = F.mse_loss(outputs, targets).item()
    #rmse = torch.sqrt(mse).item()
    mae = F.l1_loss(outputs, targets).item()
    return mse, mae #rmse


def compute_l2_regularization(model, lambda_l2=0.0001):
    l2_reg = torch.tensor(0.).to(device)
    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name:  # Considera solo i pesi dei layer di convoluzione
            l2_reg += torch.norm(param, p=2) ** 2
    return lambda_l2 * l2_reg


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, y_true, y_pred):
        # Calcola l'errore quadratico
        squared_error = (y_true - y_pred) ** 2
        weighted_error = torch.pow(squared_error, 1.5)

        return torch.mean(weighted_error)
    

def filtra_indici(array, lista_filtri):
    indici = [i for i, elemento in enumerate(array) if elemento in lista_filtri]
    return indici


class FocusedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(FocusedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_true, y_pred):
        # Calcolare l'errore
        error = y_true - y_pred
        abs_error = torch.abs(error)
        
        # Identificare i target piccoli e positivi
        small_target_mask = (y_true > 0) & (y_true < self.beta)
        
        # Peso maggiore per gli errori associati a target piccoli e positivi
        weighted_error = abs_error * (1 + self.alpha * small_target_mask.float())
        
        # Calcolo della loss come MSE ponderata
        loss = torch.mean(weighted_error ** 2)
        
        return loss
    
class CustomAttentionLoss(nn.Module):
    def __init__(self, epsilon=1e-1):
        super(CustomAttentionLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_true, y_pred):
        # Definisci una scala personalizzata per i pesi
        # I valori maggiori (in senso positivo) ottengono un peso maggiore
        weights = torch.clamp(y_true, min=0) + self.epsilon
        
        # Calcolo dell'errore quadratico medio ponderato (MSE ponderato)
        loss = torch.mean(weights * (y_true - y_pred) ** 2)
        
        return loss
    


def normalizza_pesi(distanze):
    # Trova il valore minimo e massimo nella lista delle distanze
    d_min = min(distanze)
    d_max = max(distanze)
    
    # Applica la formula della min-max normalization per assegnare i pesi
    pesi = [1 - (d - d_min) / (d_max - d_min) for d in distanze]
    
    return pesi

def distance_mse_loss(y_true, y_pred, pesi):
    # Calcola l'errore quadrato pesato
    error = np.array([(w * (yp - yt) ** 2) for yt, yp, w in zip(y_true, y_pred, pesi)])
    # Calcola la MSE pesata
    return np.mean(error)

def distance_mae_loss(y_true, y_pred, pesi):
    # Calcola l'errore quadrato pesato
    error = np.array([(w * np.abs(yp - yt)) for yt, yp, w in zip(y_true, y_pred, pesi)])
    # Calcola la MSE pesata
    return np.mean(error)

def calculate_all_first_arrival_time(df, metadata):
    latitudes = []
    longitudes = []
    hypocentral_distance = []

    for i in range(len(metadata)):
        # Filtra il DataFrame per il source_id corrente
        arrival_time = df[df['source_id'] == metadata[i][0]][['station_latitude_deg','station_longitude_deg','trace_P_arrival_time','path_hyp_distance_km']]
        
        # Controlla se arrival_time è vuoto
        if arrival_time.empty:
            # Aggiungi valori predefiniti se non ci sono arrivi
            latitudes.append(0)
            longitudes.append(0)
            hypocentral_distance.append(0)
        else:
            # Altrimenti, prendi i valori dal DataFrame
            arrival_time.sort_values(by='trace_P_arrival_time', inplace=True)
            latitudes.append(arrival_time.iloc[0, 0])
            longitudes.append(arrival_time.iloc[0, 1])
            hypocentral_distance.append(arrival_time.iloc[0, 3])

    # Crea un array numpy dallo stacked_tensor
    stacked_tensor = np.stack((latitudes, longitudes, hypocentral_distance), axis=1)

    return stacked_tensor

def preprocess_llm(data, file_path):
    with open(file_path, 'r') as file:
        # Leggi tutto il contenuto del file
        contenuto = file.read()
    contenuto = contenuto.split('\n')
    list_id = [float(elemento) for elemento in contenuto if elemento.strip()]
    id_mancanti = [i for i in list_id if i not in data['source_id'].values]

    # Se ci sono ID mancanti, crea un DataFrame con tutte le nuove righe
    if id_mancanti:
        # Crea una lista di dizionari per ogni nuovo ID
        nuove_righe = [{'source_id': i,
                        'lat_target': 0,
                        'lon_target': 0,
                        'depth_target': 0,
                        'mag_target': 0,
                        'first_p_target': 0,
                        'lat_pred': 0,
                        'lon_pred': 0,
                        'depth_pred': 0,
                        'mag_pred': 0,
                        'first_p_pred': 0} for i in id_mancanti]
        
        # Converti la lista di dizionari in un DataFrame
        nuove_righe_df = pd.DataFrame(nuove_righe)
        
        # Usa pd.concat per aggiungere tutte le nuove righe in una volta
        data = pd.concat([data, nuove_righe_df], ignore_index=True)

        return data
    

def filter_and_sort_dataframe(df, id_array, id_column = 'source_id', columns_to_select = ['lat_pred','lon_pred','depth_pred','mag_pred','first_p_pred']):
    """
    Filtra e ordina un DataFrame in base a una lista di ID e restituisce un array NumPy con colonne specifiche.

    Parameters:
    df (pd.DataFrame): Il DataFrame da cui estrarre i dati.
    id_array (array-like): L'array di ID in base a cui filtrare e ordinare.
    id_column (str): Il nome della colonna del DataFrame che contiene gli ID da confrontare.
    columns_to_select (list): Le colonne del DataFrame da selezionare per l'array finale.

    Returns:
    np.ndarray: Un array NumPy contenente i dati delle colonne selezionate, filtrati e ordinati.
    """
    
    # Filtra i dati in cui la colonna 'id_column' contiene valori presenti in 'id_array'
    filtered_data = df[df[id_column].isin(id_array)].copy()
    
    # Ordina i dati filtrati secondo l'ordine di 'id_array'
    filtered_data.loc[:, 'order'] = pd.Categorical(filtered_data[id_column], categories=id_array, ordered=True)
    sorted_filtered_data = filtered_data.sort_values(by='order').drop(columns='order')
    
    # Seleziona solo le colonne desiderate
    selected_columns_data = sorted_filtered_data[columns_to_select]
    
    # Converti i dati selezionati in un array NumPy
    return selected_columns_data.to_numpy()

def haversine(lat1, lon1, lat2, lon2):
    # Raggio della Terra in chilometri
    R = 6371.0

    # Converti le latitudini e longitudini da gradi a radianti
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Differenza di latitudine e longitudine
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Formula dell'Haversine
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Calcola la distanza
    distance = R * c

    return distance

def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0  # Raggio della Terra in km

    # Converti le latitudini e longitudini da gradi a radianti
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Differenza di latitudine e longitudine
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Formula dell'Haversine
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Distanza finale
    return R * c


