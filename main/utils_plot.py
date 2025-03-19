import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pdb
import os
from utils import normalizza_pesi,distance_mse_loss,distance_mae_loss
from geopy.distance import geodesic


def filter_data(df, metadata, i, predictions, targets, coords):
    
    filtered_df = df[df['source_id'] == metadata[i]]

    if filtered_df.empty:
        raise ValueError(f"No data found for source_id: {metadata[i]}")
    
    predictions = predictions[i,:,0]
    targets = targets[i,:,0]

    ID = metadata[i]
    # Estrai le coordinate di tutte le stazioni
    lat = coords[:,0]
    lon = coords[:,1]

    # Estrai le informazioni dalla sorgente specificata
    source_info = df[df['source_id'] == metadata[i]][['source_magnitude', 'source_latitude_deg', 'source_longitude_deg']].iloc[0]

    
    # Ottieni la magnitudo, la longitudine e la latitudine dalla sorgente
    magnitude = source_info.iloc[0]
    latitude_value = source_info.iloc[1]
    longitude_value = source_info.iloc[2]
    
    # Definisci i range di latitudine e longitudine
    lat_range_min = latitude_value - 1
    lat_range_max = latitude_value + 1
    lon_range_min = longitude_value - 1
    lon_range_max = longitude_value + 1
    
    # Trova gli indici delle stazioni all'interno del range specificato
    indices_within_range = np.where((lat >= lat_range_min) & (lat <= lat_range_max) &
                                    (lon >= lon_range_min) & (lon <= lon_range_max))[0]
    
    # Filtra le predizioni, i target, le latitudini e le longitudini
    filtered_predictions = predictions[indices_within_range]
    filtered_targets = targets[indices_within_range]
    filtered_lat = lat[indices_within_range]
    filtered_lon = lon[indices_within_range]

    arrival_time = df[df['source_id'] == metadata[i]][['station_latitude_deg','station_longitude_deg',
                                                   'trace_P_arrival_time','trace_S_arrival_time']]
    
    arrival_time.sort_values(by='trace_P_arrival_time',inplace=True)

    arrival_time['trace_P_arrival_time'] = pd.to_datetime(arrival_time['trace_P_arrival_time']).dt.tz_localize(None)
    arrival_time['trace_S_arrival_time'] = pd.to_datetime(arrival_time['trace_S_arrival_time']).dt.tz_localize(None)

    arrival_time['difference_P_arrival'] = (pd.to_datetime(arrival_time['trace_P_arrival_time']) - pd.to_datetime(arrival_time['trace_P_arrival_time'].iloc[0])).dt.total_seconds()
    arrival_time['difference_S_arrival'] = (pd.to_datetime(arrival_time['trace_S_arrival_time']) - pd.to_datetime(arrival_time['trace_P_arrival_time'].iloc[0])).dt.total_seconds()
    
    return filtered_predictions, filtered_targets, filtered_lat, filtered_lon, longitude_value, latitude_value, source_info, ID, magnitude, arrival_time


def plot_shakemaps_and_statistics(filtered_targets, filtered_predictions, filtered_lon, filtered_lat, longitude_value, latitude_value, ID, magnitude, arrival_time, model_name): 
    
    directory = f'/media/work/danieletrappolini/Results/{model_name}'
    
    # Verifica se la cartella esiste, se non esiste, creala
    if not os.path.exists(directory):
        os.makedirs(directory)

    save_path=f'{directory}/shakemaps_{ID}.png'

    # Calcolo di MAE e MSE
    mae = mean_absolute_error(filtered_targets, filtered_predictions)
    mse = mean_squared_error(filtered_targets, filtered_predictions)

    # Determine the common color scale limits
    vmin = min(filtered_targets.min(), filtered_predictions.min())
    vmax = max(filtered_targets.max(), filtered_predictions.max())

    # Creazione delle due shakemap affiancate
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 8), 
                             gridspec_kw={'width_ratios': [1, 1, 0.3]}, subplot_kw={'projection': ccrs.PlateCarree()})

    # Configurazione della proiezione e creazione della prima mappa (valori reali)
    ax1 = axes[0]
    ax1.set_extent([longitude_value-1, longitude_value+1, latitude_value-1, latitude_value+1], crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.LAND)
    ax1.add_feature(cfeature.OCEAN)
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    ax1.add_feature(cfeature.LAKES, alpha=0.5)
    ax1.add_feature(cfeature.RIVERS)
    scatter1 = ax1.scatter(filtered_lon, filtered_lat, c=filtered_targets, cmap='viridis', s=50, edgecolor='k', alpha=0.7, vmin=vmin, vmax=vmax)
    ax1.plot(longitude_value, latitude_value, 'r*', markersize=15)  # Add red star
    ax1.set_title('Shakemap - Real')

    # Annotate the points with their values
    for (x, y, val) in zip(filtered_lon, filtered_lat, filtered_targets):
        ax1.annotate(f"{val:.2f}", (x, y), textcoords="offset points", xytext=(0, 5), ha='center')

    # Configurazione della proiezione e creazione della seconda mappa (valori predetti)
    ax2 = axes[1]
    ax2.set_extent([longitude_value-1, longitude_value+1, latitude_value-1, latitude_value+1], crs=ccrs.PlateCarree())
    ax2.add_feature(cfeature.LAND)
    ax2.add_feature(cfeature.OCEAN)
    ax2.add_feature(cfeature.COASTLINE)
    ax2.add_feature(cfeature.BORDERS, linestyle=':')
    ax2.add_feature(cfeature.LAKES, alpha=0.5)
    ax2.add_feature(cfeature.RIVERS)
    scatter2 = ax2.scatter(filtered_lon, filtered_lat, c=filtered_predictions, cmap='viridis', s=50, edgecolor='k', alpha=0.7, vmin=vmin, vmax=vmax)
    ax2.plot(longitude_value, latitude_value, 'r*', markersize=15)  # Add red star
    ax2.set_title('Shakemap - Predicted')

    # Annotate the points with their values
    for (x, y, val) in zip(filtered_lon, filtered_lat, filtered_predictions):
        ax2.annotate(f"{val:.2f}", (x, y), textcoords="offset points", xytext=(0, 5), ha='center')


    # Adding conditional circles
    for idx, row in arrival_time.iterrows():
        if row['difference_P_arrival'] < 10.0:
            ax1.plot(row['station_longitude_deg'], row['station_latitude_deg'], 'o', markersize=10, markerfacecolor='none', markeredgecolor='red', markeredgewidth=2)
            ax2.plot(row['station_longitude_deg'], row['station_latitude_deg'], 'o', markersize=10, markerfacecolor='none', markeredgecolor='red', markeredgewidth=2)
            if row['difference_S_arrival'] < 10.0:
                ax1.plot(row['station_longitude_deg'], row['station_latitude_deg'], 'o', markersize=20, markerfacecolor='none', markeredgecolor='blue', markeredgewidth=2)
                ax2.plot(row['station_longitude_deg'], row['station_latitude_deg'], 'o', markersize=20, markerfacecolor='none', markeredgecolor='blue', markeredgewidth=2)

    # Creazione di un asse per la tabella
    ax_table = fig.add_subplot(133)
    ax_table.axis('off')

    # Creazione della tabella
    table_data = [
        ['Event Id', ID],
        ["Magnitude", magnitude],
        ['MAE', mae],
        ['MSE', mse]
    ]
    table = ax_table.table(cellText=table_data, colLabels=["Parameter", "Value"], cellLoc='center', loc='center')
    table.scale(2, 2)

    # Modifica della dimensione del font
    for key, cell in table.get_celld().items():
        cell.set_fontsize(14)

    # Adjust layout to make room for colorbar
    plt.tight_layout(rect=[0, 0, 0.9, 1])  

    # Salva il plot
    plt.savefig(save_path, bbox_inches='tight')

    print(f'Plot saved to {save_path}')

    # Chiudi la figura per liberare memoria
    plt.close(fig)



def get_string_from_number(number):
    mapping = {
        0: "PGA",
        1: "PGV",
        2: "SA03",
        3: "SA10",
        4: "SA30"
    }
    
    return mapping.get(number, "Valore non valido")



def compute_loss(df,real, predicted, metadata, coords, j, features_target=0, sub_deg=0.7):
    predictions = predicted[j,:,features_target]
    targets = real[j,:,features_target]
    lat = coords[:,0]
    lon = coords[:,1]
    num_stations = 565
    offset = 1e-6

    try:
        source_info = df[df['source_id'] == metadata[j]][['source_magnitude', 'source_latitude_deg', 'source_longitude_deg']].iloc[0]
        arrival_time = df[df['source_id'] == metadata[j]][['station_latitude_deg', 'station_longitude_deg',
                                                        'trace_P_arrival_time', 'trace_S_arrival_time', 'path_hyp_distance_km']]

        arrival_time = arrival_time.sort_values(by='trace_P_arrival_time')

        arrival_time['difference_P_arrival'] = (pd.to_datetime(arrival_time['trace_P_arrival_time']) - pd.to_datetime(arrival_time['trace_P_arrival_time'].iloc[0])).dt.total_seconds()
        arrival_time['difference_S_arrival'] = (pd.to_datetime(arrival_time['trace_S_arrival_time']) - pd.to_datetime(arrival_time['trace_P_arrival_time'].iloc[0])).dt.total_seconds()
        arrival_time['difference_S_arrival'] = arrival_time['difference_S_arrival'].fillna(100.0)

        source_info = df[df['source_id'] == metadata[j]][['source_magnitude', 'source_latitude_deg', 'source_longitude_deg']].iloc[0]

        # Usa il primo valore per la latitudine e la longitudine
        longitude_value = arrival_time['station_longitude_deg'].iloc[0]
        latitude_value = arrival_time['station_latitude_deg'].iloc[0]

        lat_range_min = latitude_value - sub_deg
        lat_range_max = latitude_value + sub_deg
        lon_range_min = longitude_value - sub_deg
        lon_range_max = longitude_value + sub_deg

        # Trova gli indici delle stazioni entro l'intervallo specificato
        indices_within_range = np.where((lat >= lat_range_min) & (lat <= lat_range_max) &
                                        (lon >= lon_range_min) & (lon <= lon_range_max))[0]

        # Filtra le previsioni, i target, le latitudini e le longitudini
        filtered_predictions = predictions[indices_within_range]
        filtered_targets = targets[indices_within_range]
        filtered_lat = lat[indices_within_range]
        filtered_lon = lon[indices_within_range]

        km_distance = []
        for i in zip(filtered_lat, filtered_lon):
            km_distance.append(geodesic((source_info['source_latitude_deg'], source_info['source_longitude_deg']), i).km)

        distance_weight = normalizza_pesi(km_distance)

        dist_mse = distance_mse_loss(filtered_targets, filtered_predictions, distance_weight)
        dist_mae = distance_mae_loss(filtered_targets, filtered_predictions, distance_weight)
        mae = mean_absolute_error(filtered_targets, filtered_predictions)
        mse = mean_squared_error(filtered_targets, filtered_predictions)

        # Filtrare il DataFrame per includere solo le righe con latitudini presenti nella lista
        filtered_df = arrival_time[arrival_time['station_latitude_deg'].isin(filtered_lat)]

        # Usa .loc per evitare il warning SettingWithCopyWarning
        filtered_df = filtered_df.copy()  # Effettua una copia esplicita per evitare problemi di copia

        # Creare una colonna che rappresenta l'ordine delle latitudini nella lista
        filtered_df.loc[:, 'lat_order'] = filtered_df['station_latitude_deg'].apply(lambda x: list(filtered_lat).index(x))

        # Ordinare il DataFrame in base alla colonna 'lat_order'
        filtered_hyp = filtered_df.sort_values(by='lat_order')

        # Rimuovere la colonna 'lat_order' se non è più necessaria
        filtered_hyp = filtered_hyp.drop(columns=['lat_order'])

        # Identifica l'indice corrispondente delle stazioni
        pred_targ_id = [i for i, valore in enumerate(list(filtered_lat)) if valore in list(filtered_hyp['station_latitude_deg'])]

    except IndexError:
        #print(f"Indice fuori dai limiti per source_id = {metadata[j]}, skip this entry.")
        mae = np.nan
        mse = np.nan
        dist_mae = np.nan
        dist_mse = np.nan
        filtered_df = []

    return mae, mse, dist_mae, dist_mse, len(filtered_df)


def find_outliers(data,metadata):
    # Calcolo dei quartili
    q10 = np.percentile(data, 10)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1  # Intervallo interquartile

    # Definizione dei limiti per gli outlier
    lower_bound = q10 
    upper_bound = q3 + 1.5 * iqr

    # Identificazione degli outlier
    #outliers = [x for x in data if x > upper_bound- 0.3] #- 0.3
    outliers = [x for x in data if x > upper_bound] #- 0.3
    outliers_id = [metadata[i] for i, x in enumerate(data) if x > upper_bound]

    outliers_low = [x for x in data if x < lower_bound] #- 0.3
    outliers_low_id = [metadata[i] for i, x in enumerate(data) if x < lower_bound]

    return outliers,outliers_id,outliers_low,outliers_low_id

def estrai_caratteristiche(nome_modello):
    # Divide il nome del modello in segmenti utilizzando il carattere di underscore come separatore
    segments = nome_modello.split('_')
    
    caratteristiche = {
        'arrival_time': 'no' if segments[0] == 'no' else 'yes',  # Primo valore del path
        'distance_matrix': 'no' if segments[1] == 'no' else 'yes',  # Secondo valore del path
        'max_values': 'no' if segments[2] == 'no' else 'yes'  # Terzo valore del path
    }
    return caratteristiche

def aggiorna_dataframe(path_modello, mean_MAE, mean_MSE, mean_dist_MAE, mean_dist_MSE,
                       mean_outlier_MAE, mean_outlier_MSE, mean_outlier_dist_MAE, mean_outlier_dist_MSE,
                       max_stations, min_stations, mean_stations,csv_path="risultati.csv"):
    # Estrai il nome del modello dal path del checkpoint
    nome_modello = os.path.basename(path_modello)
    
    # Ottieni le caratteristiche del modello
    caratteristiche = estrai_caratteristiche(nome_modello)
    
    # Crea un dizionario con i dati per questa esecuzione
    dati_nuovi = {
        'nome modello': nome_modello,
        'features arrival time': caratteristiche['arrival_time'],
        'features distance matrix': caratteristiche['distance_matrix'],
        'features max values': caratteristiche['max_values'],
        'MAE': mean_MAE,
        'MSE': mean_MSE,
        'MAE dist': mean_dist_MAE,
        'MSE dist': mean_dist_MSE,
        'MAE without outlier': mean_outlier_MAE,
        'MSE without outlier': mean_outlier_MSE,
        'MAE dist without outlier': mean_outlier_dist_MAE,
        'MSE dist without outlier': mean_outlier_dist_MSE,
        'max stations': max_stations,
        'min stations': min_stations,
        'mean stations': mean_stations
    }
    
    # Verifica se il file CSV esiste già
    if os.path.exists(csv_path):
        # Carica il CSV esistente
        df = pd.read_csv(csv_path)
        
        # Controlla se il modello è già presente nel DataFrame
        if not df[df['nome modello'] == path_modello].empty:
            # Se esiste, rimuovi la riga con il modello esistente
            df = df[df['nome modello'] != path_modello]
            print(f"Sovrascrivo i dati per il modello {path_modello} nel CSV.")
        
        # Aggiungi i nuovi dati
        df = pd.concat([df, pd.DataFrame([dati_nuovi])], ignore_index=True)
    else:
        # Se il CSV non esiste, crea un nuovo DataFrame
        df = pd.DataFrame([dati_nuovi])
    
    # Salva il DataFrame aggiornato nel CSV
    df.to_csv(csv_path, index=False)
    return df