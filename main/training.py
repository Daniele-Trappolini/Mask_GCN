import os
from itertools import product
from main_gcn import main  # Assumendo che main_gcn.py accetti args
from config_parser import configure_args  # Importa la funzione di configurazione
import torch

# Definisci i possibili valori per i flag
arrival_time_options = [True, False]
distance_matrix_options = [True, False]
max_values_third_channel_options = [True, False]

# Genera tutte le combinazioni di flag
combinations = product(arrival_time_options, distance_matrix_options, max_values_third_channel_options)

# Loop su tutte le combinazioni
for arrival_time_flag, distance_matrix_flag, max_values_third_channel_flag in combinations:
    # Crea il nome del checkpoint basato sui valori dei flag
    checkpoint_name = f"{'yes' if arrival_time_flag else 'no'}_" \
                      f"{'yes' if distance_matrix_flag else 'no'}_" \
                      f"{'yes' if max_values_third_channel_flag else 'no'}_mask"
    
    # Configura gli argomenti
    args = configure_args()
    args.checkpoint_name = checkpoint_name
    args.result_name = checkpoint_name  # Usa lo stesso nome per il risultato
    args.arrival_time_flag = arrival_time_flag
    args.distance_matrix_flag = distance_matrix_flag
    args.max_values_third_channel_flag = max_values_third_channel_flag
    args.mask = True
    # Log della configurazione corrente
    print(f"Running with configuration: "
          f"checkpoint_name={checkpoint_name}, "
          f"arrival_time_flag={arrival_time_flag}, "
          f"distance_matrix_flag={distance_matrix_flag}, "
          f"max_values_third_channel_flag={max_values_third_channel_flag}")
    
    # Esegui il main con gli argomenti configurati
    main(args)