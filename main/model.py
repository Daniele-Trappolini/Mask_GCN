import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import config_parser as cp
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = cp.configure_args()    
    
#############################################################################################

class OriginalModel_gcn(nn.Module):
    def __init__(self):
        super(OriginalModel_gcn, self).__init__()
        self.model_chosen = 'nofeatures'
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 125), stride=(1, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 125), stride=(1, 2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 125), stride=(1, 2))

        # Dimensione di input per il GCN dopo i layer convoluzionali
        gcn_input_dim = 128 * ((args.stations // 8) // args.stations)
        self.gcn1 = GCNConv(gcn_input_dim + 2 if self.model_chosen == 'main' else gcn_input_dim, 64)
        self.gcn2 = GCNConv(64, 64)

        self.dropout = nn.Dropout(0.4)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * args.stations, 1280)
        self.pga = nn.Linear(1280, args.stations)
        self.pgv = nn.Linear(1280, args.stations)
        self.sa03 = nn.Linear(1280, args.stations)
        self.sa10 = nn.Linear(1280, args.stations)
        self.sa30 = nn.Linear(1280, args.stations)

    def forward(self, wav_input, features, graph_input):
        wav_input = wav_input.permute(0, 3, 1, 2)
        
        x = self.relu(self.conv1(wav_input))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        batch_size = x.shape[0]
        x = x.reshape(batch_size, args.stations, -1)
        
        edge_index, edge_weight = dense_to_sparse(torch.Tensor(graph_input[0]))
        edge_weight = edge_weight.float()

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        x = self.relu(self.gcn1(data.x, data.edge_index, data.edge_attr))
        x = torch.tanh(self.gcn2(x, data.edge_index, data.edge_attr))
        
        x = x.view(batch_size, -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))

        pga = self.pga(x)
        pgv = self.pgv(x)
        sa03 = self.sa03(x)
        sa10 = self.sa10(x)
        sa30 = self.sa30(x)
        
        return pga, pgv, sa03, sa10, sa30
    

#############################################################################################
class OriginalModel_gcn_2(nn.Module):
    def __init__(self):
        super(OriginalModel_gcn_2, self).__init__()
        self.model_chosen = 'nofeatures'
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 125), stride=(1, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 125), stride=(1, 2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 125), stride=(1, 2))

        # Dimensione di input per il GCN dopo i layer convoluzionali
        gcn_input_dim = 128 * ((args.stations // 8) // args.stations)
        self.gcn1 = GCNConv(gcn_input_dim, 64)
        self.gcn2 = GCNConv(64, 64)

        self.dropout = nn.Dropout(0.4)

        # Fully connected layers
        self.fc_extra = nn.Linear(1280 + 2 + 565 + 565, 1280 + 2 + 565 + 565) 
        self.fc1 = nn.Linear(64 * args.stations, 1280)
        self.pga = nn.Linear(1280 + 2 + 565 + 565, args.stations)
        self.pgv = nn.Linear(1280 + 2 + 565 + 565, args.stations)
        self.sa03 = nn.Linear(1280 + 2 + 565 + 565, args.stations)
        self.sa10 = nn.Linear(1280 + 2 + 565 + 565, args.stations)
        self.sa30 = nn.Linear(1280 + 2 + 565 + 565, args.stations)

    def forward(self, wav_input, features, graph_input,arrival_time, max_values_third_channel_all, distance_matrix, edge_weight=None):
        wav_input = wav_input.permute(0, 3, 1, 2)
        
        x = self.relu(self.conv1(wav_input))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        batch_size = x.shape[0]
        x = x.reshape(batch_size, args.stations, -1)
        
        if isinstance(graph_input, torch.Tensor) and graph_input.dim() == 2 and graph_input.shape[0] == 2:
            edge_index = graph_input  # Usa `graph_input` come `edge_index` se è già nel formato corretto
            # Usa `edge_weight` se è fornito, altrimenti impostalo su None (o usa un valore predefinito)
            edge_weight = edge_weight if edge_weight is not None else torch.ones(edge_index.size(1)).float()
        else:
            # Conversione `dense_to_sparse` se `graph_input` non è `edge_index`
            edge_index, edge_weight = dense_to_sparse(torch.Tensor(graph_input[0]))
            edge_weight = edge_weight.float()

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        x = self.relu(self.gcn1(data.x, data.edge_index, data.edge_attr))
        x = torch.tanh(self.gcn2(x, data.edge_index, data.edge_attr))
        
        x = x.view(batch_size, -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))

         # Espandi arrival_time per ogni batch
        arrival_time = arrival_time.view(batch_size, 2)
        max_values_third = max_values_third_channel_all.view(batch_size, 565)
        distance_matrix = distance_matrix.view(batch_size, 565)

        x_with_extra = torch.cat((x, arrival_time, max_values_third, distance_matrix), dim=1)

        x_with_extra = self.relu(self.fc_extra(x_with_extra))

        # Passare attraverso i vari layer di predizione
        pga = self.pga(x_with_extra)
        pgv = self.pgv(x_with_extra)
        sa03 = self.sa03(x_with_extra)
        sa10 = self.sa10(x_with_extra)
        sa30 = self.sa30(x_with_extra)
        
        return pga, pgv, sa03, sa10, sa30

#############################################################################################

class OriginalModel_gcn_3(nn.Module):
    def __init__(self, arrival_time_flag=False, distance_matrix_flag=False, max_values_third_channel_flag=False):
        super(OriginalModel_gcn_3, self).__init__()
        
        # Flags per controllare l'inclusione delle feature
        self.arrival_time_flag = arrival_time_flag
        self.distance_matrix_flag = distance_matrix_flag
        self.max_values_third_channel_flag = max_values_third_channel_flag
        
        self.relu = nn.ReLU()

        # Layer convoluzionali
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 125), stride=(1, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 125), stride=(1, 2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 125), stride=(1, 2))

        # Dimensione di input per il GCN dopo i layer convoluzionali
        gcn_input_dim = 128 * ((args.stations // 8) // args.stations)
        self.gcn1 = GCNConv(gcn_input_dim, 64)
        self.gcn2 = GCNConv(64, 64)

        self.dropout = nn.Dropout(0.4)

        # Definisci le dimensioni aggiuntive in base ai flag
        extra_dim = 0
        if self.arrival_time_flag:
            extra_dim += 2
        if self.max_values_third_channel_flag:
            extra_dim += 565
        if self.distance_matrix_flag:
            extra_dim += 565

        # Fully connected layers con condizione sui flag
        self.fc_extra = nn.Linear(1280 + extra_dim, 1280 + extra_dim)
        self.fc1 = nn.Linear(64 * args.stations, 1280)
        
        self.pga = nn.Linear(1280 + extra_dim, args.stations)
        self.pgv = nn.Linear(1280 + extra_dim, args.stations)
        self.sa03 = nn.Linear(1280 + extra_dim, args.stations)
        self.sa10 = nn.Linear(1280 + extra_dim, args.stations)
        self.sa30 = nn.Linear(1280 + extra_dim, args.stations)

    def forward(self, wav_input, features, graph_input, arrival_time, max_values_third_channel_all, distance_matrix,edge_weight=None):
        if wav_input.dim() == 2:
            wav_input = wav_input.view(565,1000,3)
            wav_input = wav_input.unsqueeze(0)
        
        wav_input = wav_input.permute(0, 3, 1, 2)
        
        # Passaggio convoluzionale
        x = self.relu(self.conv1(wav_input))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        batch_size = x.shape[0]
        x = x.reshape(batch_size, args.stations, -1)
        
        if graph_input.dim() < 3:
            edge_index = graph_input  # Usa `graph_input` come `edge_index` se è già nel formato corretto
            edge_weight = edge_weight.float() 
        else:
            edge_index, edge_weight = dense_to_sparse(torch.Tensor(graph_input[0]))
            edge_weight = edge_weight.float()

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        x = self.relu(self.gcn1(data.x, data.edge_index, data.edge_attr))
        x = torch.tanh(self.gcn2(x, data.edge_index, data.edge_attr))
        
        x = x.view(batch_size, -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        # Gestione di arrival_time
        if self.arrival_time_flag:
            arrival_time = arrival_time.view(batch_size, 2)
        else:
            arrival_time = torch.empty(batch_size, 0).to(x.device)

        # Gestione di max_values_third_channel_all
        if self.max_values_third_channel_flag:
            max_values_third = max_values_third_channel_all.view(batch_size, 565)
        else:
            max_values_third = torch.empty(batch_size, 0).to(x.device)

        # Gestione di distance_matrix
        if self.distance_matrix_flag:
            distance_matrix = distance_matrix.view(batch_size, 565)
        else:
            distance_matrix = torch.empty(batch_size, 0).to(x.device)

        # Concatena tutte le feature extra in base ai flag
        x_with_extra = torch.cat((x, arrival_time, max_values_third, distance_matrix), dim=1)
        x_with_extra = self.relu(self.fc_extra(x_with_extra))

        # Passaggio attraverso i vari layer di predizione
        pga = self.pga(x_with_extra)
        pgv = self.pgv(x_with_extra)
        sa03 = self.sa03(x_with_extra)
        sa10 = self.sa10(x_with_extra)
        sa30 = self.sa30(x_with_extra)
        
        return pga, pgv, sa03, sa10, sa30


#############################################################################################

class llm_gcn_3(nn.Module):
    def __init__(self):
        super(OriginalModel_gcn_3, self).__init__()
        self.model_chosen = 'nofeatures'
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 125), stride=(1, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 125), stride=(1, 2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 125), stride=(1, 2))

        # Dimensione di input per il GCN dopo i layer convoluzionali
        gcn_input_dim = 128 * ((args.stations // 8) // args.stations)
        self.gcn1 = GCNConv(gcn_input_dim + 2 if self.model_chosen == 'main' else gcn_input_dim, 64)
        self.gcn2 = GCNConv(64, 64)

        self.dropout = nn.Dropout(0.4)

        # Fully connected layers
        self.fc_extra = nn.Linear(565, 565) 
        self.fc1 = nn.Linear(64 * args.stations, 1280)
        self.pga = nn.Linear(1280 + 5 + 565, args.stations)
        self.pgv = nn.Linear(1280 + 5 + 565, args.stations)
        self.sa03 = nn.Linear(1280 + 5 + 565, args.stations)
        self.sa10 = nn.Linear(1280 + 5 + 565, args.stations)
        self.sa30 = nn.Linear(1280 + 5 + 565, args.stations)

    def forward(self, wav_input, features, graph_input,llm_metadata, max_values_third_channel_all):
        wav_input = wav_input.permute(0, 3, 1, 2)
        
        x = self.relu(self.conv1(wav_input))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        batch_size = x.shape[0]
        x = x.reshape(batch_size, args.stations, -1)
        
        edge_index, edge_weight = dense_to_sparse(torch.Tensor(graph_input[0]))
        edge_weight = edge_weight.float()

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        x = self.relu(self.gcn1(data.x, data.edge_index, data.edge_attr))
        x = torch.tanh(self.gcn2(x, data.edge_index, data.edge_attr))
        
        x = x.view(batch_size, -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))

         # Espandi arrival_time per ogni batch
        llm_metadata = llm_metadata.view(batch_size, 5)
        max_values_third = max_values_third_channel_all.view(batch_size, 565)
        extra_features = self.relu(self.fc_extra(max_values_third))

        
        # Concatenare l'output con arrival_time
        x_with_extra = torch.cat((x, llm_metadata, extra_features), dim=1)

        # Passare attraverso i vari layer di predizione
        pga = self.pga(x_with_extra)
        pgv = self.pgv(x_with_extra)
        sa03 = self.sa03(x_with_extra)
        sa10 = self.sa10(x_with_extra)
        sa30 = self.sa30(x_with_extra)
        
        return pga, pgv, sa03, sa10, sa30