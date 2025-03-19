import argparse

def configure_args():
    parser = argparse.ArgumentParser()

    # Dataset type
    parser.add_argument("--network_choice",
                        default='INSTANCE',
                        type=str,
                        help="Choose the dataset type. Options: [INSTANCE,CI,CW]")
    
    parser.add_argument("--data_path",
                        default='../data/output_data.h5', # data
                        type=str,
                        help="")
    
    parser.add_argument("--graph_input_path",
                        default='../data/adjacency_matrix.npy',
                        type=str,
                        help="")
    
    parser.add_argument("--graph_features_path",
                        default='../data/station_coords_INSTANCE.npy',
                        type=str,
                        help="")
    
    parser.add_argument("--checkpoint_name",
                        default='',
                        type=str,
                        help="")  

    parser.add_argument("--result_name",
                        default='',
                        type=str,
                        help="") 

    parser.add_argument("--run_name",
                        default='llm',
                        type=str,
                        help="")
    
    
    parser.add_argument("--model_chosen",
                        default='',
                        type=str,
                        help="")

    parser.add_argument("--random_state",
                        default=42,
                        type=int,
                        help="")
    
    parser.add_argument("--stations",
                        default=565,
                        type=int,
                        help="")

    parser.add_argument("--mask",
                        default=False,
                        type=bool,
                        help="")
        
    parser.add_argument("--num_folds",
                        default=5,
                        type=int,
                        help="")
    
    parser.add_argument("--device",
                        default=1,
                        type=int,
                        help="")
    
    parser.add_argument("--trace_length",
                        default=1000,
                        type=int,
                        help="")
    
    parser.add_argument("--original_trace_length",
                        default=2000,
                        type=int,
                        help="")
    
    parser.add_argument("--zeros_to_add",
                        default=0,
                        type=int,
                        help="")
    
    parser.add_argument("--num_epochs",
                        default=100,
                        type=int,
                        help="")
      
    parser.add_argument("--batch_size",
                        default=16,
                        type=int,
                        help="")
    
    parser.add_argument("--num_target",
                        default=5,
                        type=int,
                        help="")
    
        
    parser.add_argument("--test_percentage",
                        default=0.2,
                        type=float,
                        help="")
    
    parser.add_argument("--location",
                        default='Central Italy',
                        type=str,
                        help="")
    

    parser.add_argument("--arrival_time_flag",
                        default=False,
                        type=bool,
                        help="")


    parser.add_argument("--distance_matrix_flag",
                        default=False,
                        type=bool,
                        help="")
    

    parser.add_argument("--max_values_third_channel_flag",
                        default=True,
                        type=bool,
                        help="")

    parser.add_argument("--training",
                        default=False,
                        type=bool,
                        help="")
    
    parser.add_argument("--explainability",
                        default = False,
                        type = bool)
    
    parser.add_argument("--wandb",
                        default=False,
                        type=bool,
                        help="")
    
    parser.add_argument("--lr",
                    default=0.0001,
                    type=float,
                    help="Learning rate for the optimizer")
    
    parser.add_argument("--channels",
                        default=3,
                        type=int,
                        help="")
     
    args = parser.parse_args()
    
    return args