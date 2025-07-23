import pandas as pd
import torch

def input_filter(input: pd.DataFrame, filter: pd.DataFrame, filter_id_col: str):
    """
    Filters an input dataframe using data specified in the filter dataframe at the filter_id_col column

    Parameters:
        input (pd.DataFrame): DataFrame with input data to be filtered
        filter (pd.DataFrame): DataFrame containing the data used for filtering the input
        filter_id_col (str): column name for the filter

    Returns:
        filtered_input (pd.DataFrame): filtered input DataFrame 
    """
    
    input_nodes = input.index.values.astype(str).tolist()
    filter_nodes = filter[filter_id_col].tolist()

    filtered_input_nodes_indices = [int(node) for node in input_nodes if node in filter_nodes]

    return input.loc[filtered_input_nodes_indices]

tumor_df = pd.read_csv('data/test_tumor_samples.tsv', sep='\t')
nodes_df = pd.read_csv('data/metapathway_nodes_2025.tsv', sep='\t')
edges_df = pd.read_csv('data/metapathway_edges_simplified_2025.tsv', sep='\t')
pathway_df = pd.read_csv('data/metapathway_nodes_to_pathways_2025.tsv', sep='\t')

tumor_df = input_filter(tumor_df, nodes_df, '#Id')

tumor_ids = tumor_df.index.astype(str).to_series() # Input nodes
nodes_ids = nodes_df['#Id'] # Metapathway nodes
nodes_source_ids = edges_df['#Source'] # Metapathway source nodes (2nd layer)
nodes_target_ids = edges_df['Target'] # Metapathway target nodes (2nd layer)
pathway_nodes_source_ids = pathway_df['NodeId'] # Metapathway source nodes to pathways (3rd layer)
pathway_nodes_target_ids = pathway_df['#PathwayId'] # Pathway target nodes (3rd layer)

input_data = tumor_df.T.values

# input -> metapathway total (15859 -> 20507)
node_index = {node_id: i for i, node_id in enumerate(nodes_ids.tolist())}
idx_in = torch.tensor([node_index[id] for id in tumor_ids.tolist()], dtype=torch.long)

# metapathway total -> metapathway source (20507 -> 13186)
node_index = {node_id: i for i, node_id in enumerate(nodes_ids.tolist())}
idx_src = torch.tensor([node_index[id] for id in nodes_source_ids.drop_duplicates().tolist()], dtype=torch.long)

# metapathway target -> metapathway total (18625 -> 20507)
node_index = {node_id: i for i, node_id in enumerate(nodes_ids.tolist())}
idx_tgt = torch.tensor([node_index[id] for id in nodes_target_ids.drop_duplicates().tolist()], dtype=torch.long)

# metapathway total -> pathway source (20507 -> 15825)
node_index = {node_id: i for i, node_id in enumerate(nodes_ids.tolist())}
idx_pathway = torch.tensor([node_index[id] for id in pathway_nodes_source_ids.drop_duplicates().tolist()], dtype=torch.long)