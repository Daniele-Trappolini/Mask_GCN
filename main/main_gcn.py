import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.nn as nn
import utils as u
import config_parser as cp
import h5py
import pandas as pd
from utils import normalize, calculate_all_first_arrival_time
from model import OriginalModel_gcn_3
import os
import time
import pdb

# Caricamento e filtraggio dei dati se necessario

def main(args):
    checkpoint_dir = f"../Checkpoint/{args.checkpoint_name}"
    result_dir = f"../Results/{args.result_name}"
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Checkpoint directory: {result_dir}")
    ours = pd.read_csv('../data/metadata.csv')
    args = cp.configure_args()
    seeds = [1] 

    filtered = []
    if args.location == 'Central Italy':
        conditions = (
            (ours['source_depth_km'] < 30) &
            (ours['source_magnitude'] > 2.9) &
            (ours['source_latitude_deg'] > 42) & (ours['source_latitude_deg'] < 43.75) &
            (ours['source_longitude_deg'] > 12.3) & (ours['source_longitude_deg'] < 14)
        )
        filtered = ours.loc[conditions, 'source_id'].tolist()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    map_location = None if torch.cuda.is_available() else 'cpu'

    for seed in seeds:
        print(f"Using seed: {seed}")
        u.seed_everything(seed)

        if args.network_choice == 'INSTANCE':
            with h5py.File(args.data_path, 'r') as file:
                print("Keys:", list(file.keys()))
                
                sorted_traces = file['sorted_traces'][:] if 'sorted_traces' in file else None
                target = file['target'][:] if 'target' in file else None
                metadata = file['event_ID'][:] if 'event_ID' in file else None
                mask = file['mask'][:] if 'mask' in file else None

            test_set_size = args.test_percentage
            original_inputs = sorted_traces.reshape(-1, args.stations, args.original_trace_length, args.channels)

            # Estraggo il canale 0 per calcoli futuri
            third_channel_all = original_inputs[:, :, :args.trace_length, :]
            max_values_third_all_channel = np.max(third_channel_all, axis=2)
            max_values_third_channel_all = np.max(max_values_third_all_channel, axis=2)

            # Normalizzazione e reshaping
            inputs = normalize(original_inputs[:, :, :args.trace_length, :])
            targets = target.reshape(-1, args.stations, args.num_target)
            metadata = metadata.reshape(-1, args.stations)

            if args.mask:
                mask = mask.reshape(-1, args.stations)
            else:
                mask = np.ones((metadata.shape[0], metadata.shape[1]))

            # Filtro i dati se siamo in "Central Italy"
            if args.location == 'Central Italy':
                index = u.filtra_indici(list(metadata[:, 0]), set(filtered))
                inputs = inputs[index, :, :, :]
                targets = targets[index, :, :]
                max_values_third_channel_all = max_values_third_channel_all[index, :]
                metadata = metadata[index, :]
                mask = mask[index, :]

            # Creazione loss_mask
            linear_threshold = 0
            loss_mask = torch.tensor(targets >= linear_threshold).float()

            # Log scaling dei target
            offset = 1e-6
            targets = np.log10(targets / 100 + offset)
            min_value = np.min(targets)
            targets = targets + np.abs(min_value)

            # Carico input e features del grafo
            graph_input = np.load(args.graph_input_path, allow_pickle=True)
            graph_input = np.array([graph_input] * inputs.shape[0])
            graph_features = np.load(args.graph_features_path, allow_pickle=True)
            graph_features = np.array([graph_features] * inputs.shape[0])

        # Shuffle degli indici
        np.random.seed(args.random_state)
        idx_dset = np.arange(len(inputs))
        np.random.shuffle(idx_dset)
        idx_train_set = idx_dset[:len(idx_dset) - int(len(idx_dset) * test_set_size)]
        idx_test_set = idx_dset[len(idx_dset) - int(len(idx_dset) * test_set_size):]

        print("len(idx_dset)", len(idx_dset),
            "len(idx_train_set)", len(idx_train_set),
            "len(idx_test_set)", len(idx_test_set))

        # Suddivisione dati train e test
        train_features = graph_features[idx_train_set]
        test_features = graph_features[idx_test_set]
        train_graph_input = graph_input[idx_train_set]
        test_graph_input = graph_input[idx_test_set]
        train_inputs = inputs[idx_train_set]
        test_inputs = inputs[idx_test_set]
        train_targets = targets[idx_train_set]
        test_targets = targets[idx_test_set]
        train_metadata = metadata[idx_train_set].astype(int)
        test_metadata = metadata[idx_test_set].astype(int)
        train_mask = mask[idx_train_set]
        test_mask = mask[idx_test_set]
        train_loss_mask = loss_mask[idx_train_set]
        test_loss_mask = loss_mask[idx_test_set]
        train_max_values_third_channel_all = max_values_third_channel_all[idx_train_set]
        test_max_values_third_channel_all = max_values_third_channel_all[idx_test_set]

        # Calcolo first arrival times
        train_first_arrival = calculate_all_first_arrival_time(ours, train_metadata)[:, :2]
        test_first_arrival = calculate_all_first_arrival_time(ours, test_metadata)[:, :2]

        # Normalizzo nuovamente gli input per sicurezza
        trace_len = args.trace_length
        train_inputs = u.normalize(train_inputs[:, :, :trace_len, :])
        test_inputs = u.normalize(test_inputs[:, :, :trace_len, :])

        # Calcolo distanze
        train_distance = u.haversine_vectorized(train_features[:, :, 0], train_features[:, :, 1],
                                                train_first_arrival[:, 0].reshape(-1, 1),
                                                train_first_arrival[:, 1].reshape(-1, 1))
        test_distance = u.haversine_vectorized(test_features[:, :, 0], test_features[:, :, 1],
                                            test_first_arrival[:, 0].reshape(-1, 1),
                                            test_first_arrival[:, 1].reshape(-1, 1))

        # Crea TensorDataset per train e test
        train_dataset = TensorDataset(
            torch.tensor(train_inputs, dtype=torch.float32),
            torch.tensor(train_targets, dtype=torch.float32),
            torch.tensor(train_features, dtype=torch.float32),
            torch.tensor(train_graph_input, dtype=torch.float32),
            torch.tensor(train_metadata, dtype=torch.int32),
            torch.tensor(train_mask, dtype=torch.float32),
            torch.tensor(train_loss_mask, dtype=torch.float32),
            torch.tensor(train_first_arrival, dtype=torch.float32),
            torch.tensor(train_max_values_third_channel_all, dtype=torch.float32),
            torch.tensor(train_distance, dtype=torch.float32)
        )

        test_dataset = TensorDataset(
            torch.tensor(test_inputs, dtype=torch.float32),
            torch.tensor(test_targets, dtype=torch.float32),
            torch.tensor(test_features, dtype=torch.float32),
            torch.tensor(test_graph_input, dtype=torch.float32),
            torch.tensor(test_metadata, dtype=torch.int32),
            torch.tensor(test_mask, dtype=torch.float32),
            torch.tensor(test_loss_mask, dtype=torch.float32),
            torch.tensor(test_first_arrival, dtype=torch.float32),
            torch.tensor(test_max_values_third_channel_all, dtype=torch.float32),
            torch.tensor(test_distance, dtype=torch.float32)
        )

        # Carico i DataLoader senza scartare l'ultimo batch
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        batch_size = args.batch_size
        num_epochs = args.num_epochs
        patience = 50

        best_losses_per_fold = []

        # Conversione in torch tensor dei dati per KFold
        X_data = torch.tensor(train_inputs, dtype=torch.float32)
        Y_data = torch.tensor(train_targets, dtype=torch.float32)
        feat_data = torch.tensor(train_features, dtype=torch.float32)
        graph_input_data = torch.tensor(train_graph_input, dtype=torch.float32)
        mask_input_data = torch.tensor(train_mask, dtype=torch.float32)
        loss_mask_data = torch.tensor(train_loss_mask, dtype=torch.float32)
        arrival_time_data = torch.tensor(train_first_arrival, dtype=torch.float32)
        max_values_third_channel_all_data = torch.tensor(train_max_values_third_channel_all, dtype=torch.float32)
        distance_data = torch.tensor(train_distance, dtype=torch.float32)

        if args.training:
            train_avg_losses = []
            val_avg_losses = []
            train_losses_dict = {'pga': [], 'pgv': [], 'sa03': [], 'sa10': [], 'sa30': []}
            val_losses_dict = {'pga': [], 'pgv': [], 'sa03': [], 'sa10': [], 'sa30': []}

            for fold, (train_index, val_index) in enumerate(kf.split(idx_train_set)):
                model = OriginalModel_gcn_3(
                    arrival_time_flag=args.arrival_time_flag,
                    distance_matrix_flag=args.distance_matrix_flag,
                    max_values_third_channel_flag=args.max_values_third_channel_flag
                ).to(device)

                best_model = type(model)().to(device)
                optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-4)
                loss_function = nn.MSELoss()

                print(f"Starting fold {fold+1}/{n_splits}")

                best_val_loss = float('inf')
                epochs_without_improvement = 0

                # Split per fold
                fold_train_inputs, fold_val_inputs = X_data[train_index], X_data[val_index]
                fold_train_targets, fold_val_targets = Y_data[train_index], Y_data[val_index]
                fold_train_features, fold_val_features = feat_data[train_index], feat_data[val_index]
                fold_train_graph_input, fold_val_graph_input = graph_input_data[train_index], graph_input_data[val_index]
                fold_train_mask, fold_val_mask = mask_input_data[train_index], mask_input_data[val_index]
                fold_train_loss_mask, fold_val_loss_mask = loss_mask_data[train_index], loss_mask_data[val_index]
                fold_train_first_arrival, fold_val_first_arrival = arrival_time_data[train_index], arrival_time_data[val_index]
                fold_train_max_values_third_channel_all, fold_val_max_values_third_channel_all = max_values_third_channel_all_data[train_index], max_values_third_channel_all_data[val_index]
                fold_train_distance, fold_val_distance = distance_data[train_index], distance_data[val_index]

                fold_train_dataset = TensorDataset(
                    fold_train_inputs, fold_train_targets, fold_train_features, fold_train_graph_input,
                    fold_train_mask, fold_train_loss_mask, fold_train_first_arrival,
                    fold_train_max_values_third_channel_all, fold_train_distance
                )

                fold_val_dataset = TensorDataset(
                    fold_val_inputs, fold_val_targets, fold_val_features, fold_val_graph_input,
                    fold_val_mask, fold_val_loss_mask, fold_val_first_arrival,
                    fold_val_max_values_third_channel_all, fold_val_distance
                )

                fold_train_loader = DataLoader(fold_train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
                fold_val_loader = DataLoader(fold_val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

                start_time = time.time()

                for epoch in range(num_epochs):

                    start_time = time.time()  # Inizio misurazione tempo per epoca

                    model.train()
                    epoch_train_losses = {'pga': [], 'pgv': [], 'sa03': [], 'sa10': [], 'sa30': []}
                    train_total_loss = 0

                    # Training loop
                    for (batch_inp, batch_tgt, batch_feat, batch_graph_inp, batch_mask,
                        batch_loss_mask, batch_arrival_time, batch_max_val_chan, batch_dist) in fold_train_loader:

                        batch_mask_reshaped = batch_mask.unsqueeze(-1).unsqueeze(-1)

                        if args.zeros_to_add != 0:
                            batch_inp = torch.nn.functional.pad(batch_inp,
                                                                pad=(0, 0, 0, args.zeros_to_add, 0, 0, 0, 0),
                                                                mode="constant",
                                                                value=0)

                        batch_inp = batch_inp * batch_mask_reshaped
                        batch_max_val_chan = batch_max_val_chan * batch_mask_reshaped.squeeze(-1).squeeze(-1)

                        optimizer.zero_grad()

                        arrival_time_input = batch_arrival_time.to(device) if args.arrival_time_flag else None
                        max_val_chan_input = batch_max_val_chan.to(device) if args.max_values_third_channel_flag else None
                        dist_input = batch_dist.to(device) if args.distance_matrix_flag else None

                        pga_output, pgv_output, sa03_output, sa10_output, sa30_output = model(
                            batch_inp.to(device),
                            batch_feat.to(device),
                            batch_graph_inp.to(device),
                            arrival_time_input,
                            max_val_chan_input,
                            dist_input
                        )

                        loss_mask_bool = batch_loss_mask.bool()[:, :, 0]
                        pga_loss = loss_function(pga_output[loss_mask_bool], batch_tgt[:, :, 0][loss_mask_bool].to(device))
                        pgv_loss = loss_function(pgv_output[loss_mask_bool], batch_tgt[:, :, 1][loss_mask_bool].to(device))
                        sa03_loss = loss_function(sa03_output[loss_mask_bool], batch_tgt[:, :, 2][loss_mask_bool].to(device))
                        sa10_loss = loss_function(sa10_output[loss_mask_bool], batch_tgt[:, :, 3][loss_mask_bool].to(device))
                        sa30_loss = loss_function(sa30_output[loss_mask_bool], batch_tgt[:, :, 4][loss_mask_bool].to(device))

                        loss = pga_loss + pgv_loss + sa03_loss + sa10_loss + sa30_loss
                        loss.backward()
                        optimizer.step()
                        train_total_loss += loss.item()

                        epoch_train_losses['pga'].append(pga_loss.item())
                        epoch_train_losses['pgv'].append(pgv_loss.item())
                        epoch_train_losses['sa03'].append(sa03_loss.item())
                        epoch_train_losses['sa10'].append(sa10_loss.item())
                        epoch_train_losses['sa30'].append(sa30_loss.item())

                    # Fine dell'epoca: calcola il tempo impiegato
                    end_time = time.time()
                    epoch_time = end_time - start_time  # Tempo totale per questa epoca

                    # Stampa il tempo impiegato per questa epoca
                    print(f"Epoch {epoch+1}/{num_epochs} completata in {epoch_time:.2f} secondi")

                    # Medie per epoch train
                    avg_train_loss = train_total_loss / len(fold_train_loader)
                    train_avg_losses.append(avg_train_loss)
                    for k in epoch_train_losses:
                        train_losses_dict[k].append(np.mean(epoch_train_losses[k]))

                    print(f"Fold {fold+1}, Epoch {epoch+1}, Train AVG Loss: {avg_train_loss:.4f}")

                    # Validation loop
                    model.eval()
                    epoch_val_losses = {'pga': [], 'pgv': [], 'sa03': [], 'sa10': [], 'sa30': []}
                    val_total_loss = 0

                    with torch.no_grad():
                        for (batch_inp, batch_tgt, batch_feat, batch_graph_inp, batch_mask,
                            batch_loss_mask, batch_arrival_time, batch_max_val_chan, batch_dist) in fold_val_loader:

                            batch_mask_reshaped = batch_mask.unsqueeze(-1).unsqueeze(-1)
                            if args.zeros_to_add != 0:
                                batch_inp = torch.nn.functional.pad(batch_inp,
                                                                    pad=(0, 0, 0, args.zeros_to_add, 0, 0, 0, 0),
                                                                    mode="constant",
                                                                    value=0)

                            batch_inp = batch_inp * batch_mask_reshaped
                            batch_max_val_chan = batch_max_val_chan * batch_mask_reshaped.squeeze(-1).squeeze(-1)

                            arrival_time_input = batch_arrival_time.to(device) if args.arrival_time_flag else None
                            max_val_chan_input = batch_max_val_chan.to(device) if args.max_values_third_channel_flag else None
                            dist_input = batch_dist.to(device) if args.distance_matrix_flag else None

                            pga_output, pgv_output, sa03_output, sa10_output, sa30_output = model(
                                batch_inp.to(device),
                                batch_feat.to(device),
                                batch_graph_inp.to(device),
                                arrival_time_input,
                                max_val_chan_input,
                                dist_input
                            )

                            loss_mask_bool = batch_loss_mask.bool()[:, :, 0]
                            pga_loss = loss_function(pga_output[loss_mask_bool], batch_tgt[:, :, 0][loss_mask_bool].to(device))
                            pgv_loss = loss_function(pgv_output[loss_mask_bool], batch_tgt[:, :, 1][loss_mask_bool].to(device))
                            sa03_loss = loss_function(sa03_output[loss_mask_bool], batch_tgt[:, :, 2][loss_mask_bool].to(device))
                            sa10_loss = loss_function(sa10_output[loss_mask_bool], batch_tgt[:, :, 3][loss_mask_bool].to(device))
                            sa30_loss = loss_function(sa30_output[loss_mask_bool], batch_tgt[:, :, 4][loss_mask_bool].to(device))

                            val_loss = pga_loss + pgv_loss + sa03_loss + sa10_loss + sa30_loss
                            val_total_loss += val_loss.item()

                            epoch_val_losses['pga'].append(pga_loss.item())
                            epoch_val_losses['pgv'].append(pgv_loss.item())
                            epoch_val_losses['sa03'].append(sa03_loss.item())
                            epoch_val_losses['sa10'].append(sa10_loss.item())
                            epoch_val_losses['sa30'].append(sa30_loss.item())

                        avg_val_loss = val_total_loss / len(fold_val_loader)
                        val_avg_losses.append(avg_val_loss)
                        for k in epoch_val_losses:
                            val_losses_dict[k].append(np.mean(epoch_val_losses[k]))

                        print(f"Fold {fold+1}, Epoch {epoch+1}, Val AVG Loss: {avg_val_loss:.4f}")

                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            best_model_path = f"{checkpoint_dir}/model_gcn_fold_{fold+1}_seed_{seed}.pth"

                            checkpoint = {
                                'model_state_dict': model.state_dict(),
                                'best_val_loss': best_val_loss,
                                'arrival_time_flag': model.arrival_time_flag,
                                'distance_matrix_flag': model.distance_matrix_flag,
                                'max_values_third_channel_flag': model.max_values_third_channel_flag,
                                'optimizer_state_dict': optimizer.state_dict()
                            }

                            torch.save(checkpoint, best_model_path)
                            print(f"Best model saved at {best_model_path} with val loss: {best_val_loss:.4f}")

                            del best_model
                            best_model = OriginalModel_gcn_3(
                                arrival_time_flag=model.arrival_time_flag,
                                distance_matrix_flag=model.distance_matrix_flag,
                                max_values_third_channel_flag=model.max_values_third_channel_flag
                            ).to(device)
                            best_model.load_state_dict(model.state_dict())
                            
                            epochs_without_improvement = 0
                        else:
                            epochs_without_improvement += 1
                        best_losses_per_fold.append(best_val_loss)

                        if epochs_without_improvement >= patience:
                            print(f"Early stopping after {patience} epochs without improvement.")
                            break


        # Se non stiamo allenando, passiamo alla fase di test
        if not args.training:
            with torch.no_grad():
                te_losses = []
                mse_lists = []
                rmse_lists = []
                mae_lists = []

                for fold in range(n_splits):
                    checkpoint_path = f"{checkpoint_dir}/model_gcn_fold_{fold+1}_seed_{seed}.pth"
                    print(f"Loading model from: {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))

                    model = OriginalModel_gcn_3(
                        arrival_time_flag=checkpoint.get('arrival_time_flag', args.arrival_time_flag),
                        distance_matrix_flag=checkpoint.get('distance_matrix_flag', args.distance_matrix_flag),
                        max_values_third_channel_flag=checkpoint.get('max_values_third_channel_flag', args.max_values_third_channel_flag)
                    ).to(device)
                    
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()

                    test_total_loss = 0
                    predictions = []
                    metadatas = []
                    target_list = []

                    for (inp, tgt, feat, g_inp, meta, msk, msk_loss,
                        arr_time, max_val_chan, dist) in tqdm(test_loader, total=len(test_loader)):

                        if args.zeros_to_add != 0:
                            inp = torch.nn.functional.pad(inp,
                                                        pad=(0, 0, 0, args.zeros_to_add, 0, 0, 0, 0),
                                                        mode="constant",
                                                        value=0)
                        msk_reshaped = msk.unsqueeze(-1).unsqueeze(-1)
                        inp = inp * msk_reshaped

                        arr_time_input = arr_time if model.arrival_time_flag else None
                        max_val_chan_input = max_val_chan if model.max_values_third_channel_flag else None
                        dist_input = dist if model.distance_matrix_flag else None

                        pga_output, pgv_output, sa03_output, sa10_output, sa30_output = model(
                            inp.to(device),
                            feat.to(device),
                            g_inp.to(device),
                            arr_time_input.to(device) if arr_time_input is not None else None,
                            max_val_chan_input.to(device) if max_val_chan_input is not None else None,
                            dist_input.to(device) if dist_input is not None else None
                        )

                        # Salva le predizioni
                        predictions.append(torch.stack([pga_output, pgv_output, sa03_output, sa10_output, sa30_output]))

                        # Calcola la loss su questo batch
                        batch_loss = (
                            nn.MSELoss()(pga_output, tgt[:, :, 0].to(device)) +
                            nn.MSELoss()(pgv_output, tgt[:, :, 1].to(device)) +
                            nn.MSELoss()(sa03_output, tgt[:, :, 2].to(device)) +
                            nn.MSELoss()(sa10_output, tgt[:, :, 3].to(device)) +
                            nn.MSELoss()(sa30_output, tgt[:, :, 4].to(device))
                        )
                        test_total_loss += batch_loss.item()

                        metadatas.append(meta[:, 0])
                        target_list.append(tgt)

                    avg_test_loss = test_total_loss / len(test_loader)
                    print(f"Fold {fold+1} Test Loss (media): {avg_test_loss:.4f}")

                    # Qui non scartiamo l'ultimo batch, quindi non facciamo slicing dei predictions
                    # ma salviamo tutto
                    predictions_tensor = torch.stack(predictions)  # [batch, 5, stations, 1]
                    predictions_tensor = predictions_tensor.permute(0, 2, 3, 1)  # [batch, stations, 1, 5]
                    predictions_tensor = predictions_tensor.reshape(195,565,5)
                    predictions_tensor = predictions_tensor.cpu().numpy()

                    stacked_target = torch.stack(target_list).numpy().reshape(-1, args.stations, 5)
                    stacked_metadata = torch.stack(metadatas).numpy().reshape(-1)

                    np.save(f'{result_dir}/predictions_fold_{fold}.npy', predictions_tensor)
                    np.save(f'{result_dir}/true_values_fold_{fold}.npy', stacked_target)
                    np.save(f'{result_dir}/metadata_{fold}.npy', stacked_metadata)

                    mse_list = []
                    rmse_list = []
                    mae_list = []

                    test_targets_aligned = test_targets[:predictions_tensor.shape[0]]
                    for i in range(5):
                        mse_val = mean_squared_error(test_targets_aligned[:, :, i], predictions_tensor[:, :, i])
                        rmse_val = mean_squared_error(test_targets_aligned[:, :, i], predictions_tensor[:, :, i], squared=False)
                        mae_val = mean_absolute_error(test_targets_aligned[:, :, i], predictions_tensor[:, :, i])
                        mse_list.append(mse_val)
                        rmse_list.append(rmse_val)
                        mae_list.append(mae_val)

                    print('MSE avg:', np.mean(mse_list))
                    print('RMSE avg:', np.mean(rmse_list))
                    print('MAE avg:', np.mean(mae_list))

                    mse_lists.append(mse_list)
                    rmse_lists.append(rmse_list)
                    mae_lists.append(mae_list)
                    te_losses.append(avg_test_loss)

                print("Test loss mean:", np.array(te_losses).mean())

                def print_stats(name, arr):
                    print(f"{name} mean: {arr.mean():.3f}",
                        "PGA", f"{np.array([i[0] for i in arr]).mean():.3f}", u"\u00B1", f"{np.array([i[0] for i in arr]).std():.3f}",
                        "PGV", f"{np.array([i[1] for i in arr]).mean():.3f}", u"\u00B1", f"{np.array([i[1] for i in arr]).std():.3f}",
                        "PSA03", f"{np.array([i[2] for i in arr]).mean():.3f}", u"\u00B1", f"{np.array([i[2] for i in arr]).std():.3f}",
                        "PSA1", f"{np.array([i[3] for i in arr]).mean():.3f}", u"\u00B1", f"{np.array([i[3] for i in arr]).std():.3f}",
                        "PSA3", f"{np.array([i[4] for i in arr]).mean():.3f}", u"\u00B1", f"{np.array([i[4] for i in arr]).std():.3f}")

                print_stats("MSE", np.array(mse_lists))
                print_stats("RMSE", np.array(rmse_lists))
                print_stats("MAE", np.array(mae_lists))


if __name__ == "__main__":
    from config_parser import configure_args
    args = cp.configure_args()  
    main(args)  