import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from geopy.distance import geodesic
import config_parser as cp
from utils import normalizza_pesi,distance_mse_loss,distance_mae_loss
from utils_plot import get_string_from_number,compute_loss,find_outliers,aggiorna_dataframe
import os

single_fold = False


# Funzione per ottenere tutti i nomi di cartelle nella directory 'Results'
def get_model_paths(results_path='../Results'):
    return [os.path.join(results_path, model_name) for model_name in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, model_name))]

# Itera su tutti i modelli nella cartella 'Results'
for model_path in get_model_paths():
    model_name = os.path.basename(model_path)
    print(f"Elaborazione del modello: {model_name}")

    # Inizializza una struttura per accumulare i risultati su tutti i fold
    folds_results = {target: {'MAE': [], 'MSE': [], 'dist_MAE': [], 'dist_MSE': [], 
                              'outlier_MAE': [], 'outlier_MSE': [], 
                              'outlier_dist_MAE': [], 'outlier_dist_MSE': [], 
                              'max_stations': [], 'min_stations': [], 'mean_stations': []} 
                     for target in range(5)}

    # Loop su tutti i fold e target
    for fold in range(5):
        # Percorsi dei file di input per il modello e il fold corrente
        target = np.load(os.path.join(model_path, f'predictions_fold_{fold}.npy'))
        coords = np.load('/media/work/danieletrappolini/GM_Instance/data/station_coords_INSTANCE.npy')
        df = pd.read_csv('/media/work/danieletrappolini/GM_Instance/data/metadata.csv')
        real = np.load(os.path.join(model_path, f'true_values_fold_{fold}.npy'))
        metadata = np.load(os.path.join(model_path, f'metadata_{fold}.npy'))
        metadata = list(metadata.astype(int)) 

        for features_target in range(5):  # Loop per ciascun target
            total_MAE, total_MSE, total_dist_MAE, total_dist_MSE, stations = [], [], [], [], []
            contatore = 0 
            
            for i in range(len(metadata)):
                loss = compute_loss(df, real, target, metadata, coords, i, features_target, sub_deg=1.0)
                if np.isnan(loss[0]):
                    contatore += 1
                else:
                    total_MAE.append(loss[0])
                    total_MSE.append(loss[1])
                    total_dist_MAE.append(loss[2])
                    total_dist_MSE.append(loss[3])
                    stations.append(loss[4])
            
            # Filtrare gli outliers
            outliers_MAE, outliers_id_MAE, _, _ = find_outliers(total_MAE, metadata)
            outliers_MSE, outliers_id_MSE, _, _ = find_outliers(total_MSE, metadata)
            outliers_dist_MSE, outliers_id_dist_MSE, _, _ = find_outliers(total_dist_MSE, metadata)
            outliers_dist_MAE, outliers_id_dist_MAE, _, _ = find_outliers(total_dist_MAE, metadata)

            # Risultati senza outliers
            result_MAE = [x for x in total_MAE if x not in outliers_MAE]
            result_MSE = [x for x in total_MSE if x not in outliers_MSE]
            result_dist_MSE = [x for x in total_dist_MSE if x not in outliers_dist_MSE]
            result_dist_MAE = [x for x in total_dist_MAE if x not in outliers_dist_MAE]

            # Calcolo dei risultati con e senza outliers
            mean_dist_MAE = sum(total_dist_MAE) / len(total_dist_MAE)
            mean_dist_MSE = sum(total_dist_MSE) / len(total_dist_MSE)
            mean_MAE = sum(total_MAE) / len(total_MAE)
            mean_MSE = sum(total_MSE) / len(total_MSE)

            result_outlier_dist_MSE = sum(result_dist_MSE) / (len(metadata) - contatore - len(outliers_dist_MSE))
            result_outlier_dist_MAE = sum(result_dist_MAE) / (len(metadata) - contatore - len(outliers_dist_MAE))
            result_outlier_MSE = sum(result_MAE) / (len(metadata) - contatore - len(outliers_MAE))
            result_outlier_MAE = sum(result_MSE) / (len(metadata) - contatore - len(outliers_MAE))

            max_stations = max(stations)
            min_stations = min(stations)
            mean_stations = int(round(sum(stations) / len(stations), 0))

            # Aggiungi i risultati del fold corrente al dizionario di accumulo
            folds_results[features_target]['MAE'].append(mean_MAE)
            folds_results[features_target]['MSE'].append(mean_MSE)
            folds_results[features_target]['dist_MAE'].append(mean_dist_MAE)
            folds_results[features_target]['dist_MSE'].append(mean_dist_MSE)
            folds_results[features_target]['outlier_MAE'].append(result_outlier_MAE)
            folds_results[features_target]['outlier_MSE'].append(result_outlier_MSE)
            folds_results[features_target]['outlier_dist_MAE'].append(result_outlier_dist_MAE)
            folds_results[features_target]['outlier_dist_MSE'].append(result_outlier_dist_MSE)
            folds_results[features_target]['max_stations'].append(max_stations)
            folds_results[features_target]['min_stations'].append(min_stations)
            folds_results[features_target]['mean_stations'].append(mean_stations)

    # Calcola la media su tutti i fold e aggiorna il DataFrame
    for features_target in range(5):
        TARGET = get_string_from_number(features_target)
        
        # Calcola la media dei risultati su tutti i fold per ciascuna metrica
        avg_results = {key: np.mean(folds_results[features_target][key]) for key in folds_results[features_target]}
        
        # Utilizza la funzione aggiorna_dataframe per salvare i risultati medi nel CSV
        df_aggiornato = aggiorna_dataframe(
            path_modello=model_name,
            mean_MAE=avg_results['MAE'],
            mean_MSE=avg_results['MSE'],
            mean_dist_MAE=avg_results['dist_MAE'],
            mean_dist_MSE=avg_results['dist_MSE'],
            mean_outlier_MAE=avg_results['outlier_MAE'],
            mean_outlier_MSE=avg_results['outlier_MSE'],
            mean_outlier_dist_MAE=avg_results['outlier_dist_MAE'],
            mean_outlier_dist_MSE=avg_results['outlier_dist_MSE'],
            max_stations=avg_results['max_stations'],
            min_stations=avg_results['min_stations'],
            mean_stations=avg_results['mean_stations'],
            csv_path=f'/media/work/danieletrappolini/GM_Instance_Dec/main/Table/results_{TARGET}.csv'
        )

        print(f"Risultati medi per {TARGET} del modello {model_name} su tutti i fold:\n", df_aggiornato)

###########################################################################################
if single_fold == True:
    args = cp.configure_args()   
    PATH = f'./Results/{args.result_name}/' 
    fold = 4
    target = np.load(PATH + f'predictions_fold_{fold}.npy')
    coords = np.load('/media/work/danieletrappolini/GM_Instance/data/station_coords_INSTANCE.npy')
    df = pd.read_csv('/media/work/danieletrappolini/GM_Instance/data/metadata.csv')
    real = np.load(PATH + f'true_values_fold_{fold}.npy')
    metadata = np.load(PATH + f'metadata_{fold}.npy')
    metadata = list(metadata.astype(int)) 

    for features_target in range(5):  # Loop attraverso i 5 target
        total_MAE = []
        total_MSE = []
        total_dist_MAE = []
        total_dist_MSE = []
        stations = []
        contatore = 0 
        
        for i in range(len(metadata)):
            loss = compute_loss(df, real, target, metadata, coords, i, features_target, sub_deg=1.0)
            if loss[0] is np.nan:
                contatore += 1
            else:
                total_MAE.append(loss[0])
                total_MSE.append(loss[1])
                total_dist_MAE.append(loss[2])
                total_dist_MSE.append(loss[3])
                stations.append(loss[4])
        
        TARGET = get_string_from_number(features_target)

        print(f'''
            {TARGET} With Outlier on testset:  
        
            DIST_MAE: {sum(total_dist_MAE) / len(total_dist_MAE)},
            DIST_MSE: {sum(total_dist_MSE) / len(total_dist_MSE)},
            MAE: {sum(total_MAE) / len(total_MAE)},
            MSE: {sum(total_MSE) / len(total_MSE)},

            Mean Stations Used: {int(round(sum(stations) / len(stations),0))}
        ''')

        # Trova gli outliers
        outliers_MAE, outliers_id_MAE, _, _ = find_outliers(total_MAE, metadata)
        outliers_MSE, outliers_id_MSE, _, _ = find_outliers(total_MSE, metadata)
        outliers_dist_MSE, outliers_id_dist_MSE, _, _ = find_outliers(total_dist_MSE, metadata)
        outliers_dist_MAE, outliers_id_dist_MAE, _, _ = find_outliers(total_dist_MAE, metadata)

        # Risultati senza outliers
        result_MAE = [x for x in total_MAE if x not in outliers_MAE]
        result_MSE = [x for x in total_MSE if x not in outliers_MSE]
        result_dist_MSE = [x for x in total_dist_MSE if x not in outliers_dist_MSE]
        result_dist_MAE = [x for x in total_dist_MAE if x not in outliers_dist_MAE]

        # Calcolo dei risultati con e senza outliers
        result_outlier_dist_MSE = sum(result_dist_MSE) / (len(metadata) - contatore - len(outliers_dist_MSE))
        result_outlier_dist_MAE = sum(result_dist_MAE) / (len(metadata) - contatore - len(outliers_dist_MAE))
        result_outlier_MSE = sum(result_MAE) / (len(metadata) - contatore - len(outliers_MAE))
        result_outlier_MAE = sum(result_MSE) / (len(metadata) - contatore - len(outliers_MAE))

        max_stations = max(stations)
        min_stations = min(stations)
        mean_stations = int(round(sum(stations) / len(stations), 0))

        # Esempio di utilizzo della funzione con i valori già calcolati
        path_modello = args.result_name

        # Valori pre-calcolati passati alla funzione
        df_aggiornato = aggiorna_dataframe(
            path_modello=path_modello,
            mean_MAE=sum(total_MAE) / len(total_MAE),
            mean_MSE=sum(total_MSE) / len(total_MSE),
            mean_dist_MAE=sum(total_dist_MAE) / len(total_dist_MAE),
            mean_dist_MSE=sum(total_dist_MSE) / len(total_dist_MSE),
            mean_outlier_MAE=result_outlier_MAE,
            mean_outlier_MSE=result_outlier_MSE,
            mean_outlier_dist_MAE=result_outlier_dist_MAE,
            mean_outlier_dist_MSE=result_outlier_dist_MSE,
            max_stations=max_stations,
            min_stations=min_stations,
            mean_stations=mean_stations,
            csv_path=f'./Table/results_{TARGET}.csv'
        )

        print(df_aggiornato)

