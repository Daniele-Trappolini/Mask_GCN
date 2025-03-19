import os
from main_gcn import main  # Assumendo che main_gcn.py accetti args
from config_parser import configure_args  # Importa la funzione di configurazione
import torch

# Fissa la configurazione per arrival_time_flag, distance_matrix_flag e max_values_third_channel_flag
arrival_time_flag = False  
distance_matrix_flag = True  
max_values_third_channel_flag = True  

# Definisci i possibili valori per trace_length
trace_length_options = [500]  # Puoi modificare i valori secondo necessità

# Loop su tutti i valori di trace_length
for trace_length in trace_length_options:
    # Crea il nome del checkpoint basato sui valori della configurazione e trace_length
    checkpoint_name = f"no_no_yes_trace_{trace_length}"
    
    # Configura gli argomenti
    args = configure_args()
    args.checkpoint_name = checkpoint_name
    args.result_name = checkpoint_name  # Usa lo stesso nome per il risultato
    args.arrival_time_flag = arrival_time_flag
    args.distance_matrix_flag = distance_matrix_flag
    args.max_values_third_channel_flag = max_values_third_channel_flag
    args.trace_length = trace_length  # Aggiungi il parametro trace_length dinamicamente

    # Log della configurazione corrente
    print(f"Running with configuration: "
          f"checkpoint_name={checkpoint_name}, "
          f"trace_length={trace_length}, "
          f"arrival_time_flag={arrival_time_flag}, "
          f"distance_matrix_flag={distance_matrix_flag}, "
          f"max_values_third_channel_flag={max_values_third_channel_flag}")
    
    # Esegui il main con gli argomenti configurati
    main(args)
