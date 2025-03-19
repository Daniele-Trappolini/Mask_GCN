import os
import pandas as pd
import numpy as np

def conta_presenze(metadata, real_metadata, num_iter=195):
    presenti = 0
    non_presenti = 0
    ID_NON_PRESENTI = []
    ID_PRESENTI = []
    index_id_presenti = []
    
    for j in range(num_iter):
        metadata_value_str = str(int(metadata[j]))

        if any(str(int(source_id)).startswith(metadata_value_str) for source_id in real_metadata):
            presenti += 1
            ID_PRESENTI.append(metadata[j])
            index_id_presenti.append(j)
        else:
            non_presenti += 1
            ID_NON_PRESENTI.append(metadata[j])
    
    return {
        "presenti": presenti,
        "non_presenti": non_presenti,
        "ID_PRESENTI": ID_PRESENTI,
        "ID_NON_PRESENTI": ID_NON_PRESENTI,
        "index_id_presenti": index_id_presenti
    }

def process_event_data(df, mask, metadata, index_id_presenti, coords, target, real, fold, output_folder):
    data = []

    for j in index_id_presenti:
        source_info = df[df['source_id'] == metadata[j]][['source_magnitude', 'source_latitude_deg', 'source_longitude_deg']].iloc[0]
        source_magnitude = source_info['source_magnitude']
        event_latitude = source_info['source_latitude_deg']
        event_longitude = source_info['source_longitude_deg']
        recording_station = mask['source_id'] == metadata[j]
        num_stations = recording_station.sum()

        longitude_value = source_info.iloc[2]
        latitude_value = source_info.iloc[1]

        sub_deg = 0.7
        lat_range_min = latitude_value - sub_deg
        lat_range_max = latitude_value + sub_deg
        lon_range_min = longitude_value - sub_deg
        lon_range_max = longitude_value + sub_deg

        lat = coords[:, 0]
        lon = coords[:, 1]
        
        indices_within_range = np.where((lat >= lat_range_min) & (lat <= lat_range_max) &
                                        (lon >= lon_range_min) & (lon <= lon_range_max))[0]
        
        filtered_lat = lat[indices_within_range]
        filtered_lon = lon[indices_within_range]
        
        predictions_pga = target[j, :, 0][indices_within_range]
        targets_pga = real[j, :, 0][indices_within_range]
        predictions_pgv = target[j, :, 1][indices_within_range]
        targets_pgv = real[j, :, 1][indices_within_range]
        predictions_sa03 = target[j, :, 2][indices_within_range]
        targets_sa03 = real[j, :, 2][indices_within_range]
        predictions_sa10 = target[j, :, 3][indices_within_range]
        targets_sa10 = real[j, :, 3][indices_within_range]
        predictions_sa30 = target[j, :, 4][indices_within_range]
        targets_sa30 = real[j, :, 4][indices_within_range]

        for i in range(len(indices_within_range)):
            data.append({
                'ID': metadata[j],
                'Magnitude': source_magnitude,
                'Event_Latitude': event_latitude,
                'Event_Longitude': event_longitude,
                'Station_Latitude': filtered_lat[i],
                'Station_Longitude': filtered_lon[i],
                'Target_PGA': targets_pga[i] - 6,
                'Prediction_PGA': predictions_pga[i] - 6,
                'Target_PGV': targets_pgv[i] - 6,
                'Prediction_PGV': predictions_pgv[i] - 6,
                'Target_SA03': targets_sa03[i] - 6,
                'Prediction_SA03': predictions_sa03[i] - 6,
                'Target_SA10': targets_sa10[i] - 6,
                'Prediction_SA10': predictions_sa10[i] - 6,
                'Target_SA30': targets_sa30[i] - 6,
                'Prediction_SA30': predictions_sa30[i] - 6,
                'Recording_Station': num_stations
            })


    os.makedirs(output_folder, exist_ok=True)
    print(f"Created/checked folder: {output_folder}")

    output_path = os.path.join(output_folder, f'event_results_{fold}.csv')

    if len(data) == 0:
        print(f"WARNING: No data to save for {output_path}!")
    else:
        output_df = pd.DataFrame(data)
        output_df.to_csv(output_path, index=False)
        print(f"Saved {len(data)} records to {output_path}")

    return output_df if len(data) > 0 else None

if __name__ == "__main__":
    BASE_PATH = '/media/work/danieletrappolini/GM_Instance_Dec/Results/'
    OUTPUT_BASE_PATH = '/media/work/danieletrappolini/GM_Instance_Dec/Event_data/'
    
    coords = np.load('/media/work/danieletrappolini/GM_Instance_Dec/data/station_coords_INSTANCE.npy')
    df = pd.read_csv('/media/work/danieletrappolini/GM_Instance_Dec/data/metadata.csv')
    mask = pd.read_csv('../data/mask.csv')
    
    for folder in os.listdir(BASE_PATH):
        folder_path = os.path.join(BASE_PATH, folder)
        output_folder = os.path.join(OUTPUT_BASE_PATH, folder)
        
        if os.path.isdir(folder_path):
            for fold in range(5):
                try:
                    target = np.load(os.path.join(folder_path, f'predictions_fold_{fold}.npy'))
                    real = np.load(os.path.join(folder_path, f'true_values_fold_{fold}.npy'))
                    metadata = np.load(os.path.join(folder_path, f'metadata_{fold}.npy'))
                    metadata = list(metadata.astype(int))
                    real_metadata = set(df['source_id'].astype(int))
                    
                    risultato = conta_presenze(metadata, real_metadata)
                    print(f'Folder {folder} - Fold {fold} - Presenti: {risultato["presenti"]}, Non presenti: {risultato["non_presenti"]}')
                    
                    process_event_data(df, mask, metadata, risultato["index_id_presenti"], coords, target, real, fold, output_folder)
                except FileNotFoundError:
                    print(f'Missing data for folder {folder}, fold {fold}. Skipping.')
