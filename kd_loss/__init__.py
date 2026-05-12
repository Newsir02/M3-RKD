"""
Knowledge Distillation Loss Functions for Graph Neural Networks

This package provides utilities and loss functions for knowledge distillation
based on random walk similarity matrices.
"""

from .util import (
    k_step_random_walk,
    n_times_k_step_random_walks,
    batch_random_walks_for_all_nodes,
    get_random_walk_nodes,
    sample_random_walk_subgraph,
    compute_random_walk_statistics,
    build_adj_list,
    batch_generate_walks,
    AddLaplacianPE,
    AddShortestPathDistance,
    AddTopologicalCore,
    AddBRICSFragments,
    AddSpectralClusters

)

from .loss_func import (
    cosine_similarity_matrix,
    compute_batch_random_walk_similarity,
    random_walk_distillation_loss,
    kl_divergence_loss,
    nce_criterion,
    nce_criterion_grouped,
    graph_topology_attention_loss,
    compute_core_view_graph_emb,
    rkd_distance_loss,
    subgraph_distillation_loss,
    topk_ranking_distillation_loss,
    lsp_distillation_loss,
    sa_mlp_node_structure_loss,
    sa_mlp_graph_structure_loss,
    sa_mlp_feature_alignment_loss,
    sa_mlp_distillation_loss,
    sale_deepwalk_loss,
    sale_mlp_distillation_loss
)

__all__ = [
    # Utility functions
    'k_step_random_walk',
    'n_times_k_step_random_walks', 
    'batch_random_walks_for_all_nodes',
    'get_random_walk_nodes',
    'sample_random_walk_subgraph',
    'compute_random_walk_statistics',
    'build_adj_list',
    'batch_generate_walks',
    'AddLaplacianPE',
    'AddShortestPathDistance',
    'AddTopologicalCore',
    'AddBRICSFragments',
    'AddSpectralClusters',

    
    # Loss functions
    'cosine_similarity_matrix',
    'compute_batch_random_walk_similarity',
    'random_walk_distillation_loss',
    'kl_divergence_loss',
    'nce_criterion',
    'nce_criterion_grouped',
    'graph_topology_attention_loss',
    'compute_core_view_graph_emb',
    'rkd_distance_loss',
    'subgraph_distillation_loss',
    'topk_ranking_distillation_loss',
    'lsp_distillation_loss',
    'sa_mlp_node_structure_loss',
    'sa_mlp_graph_structure_loss',
    'sa_mlp_feature_alignment_loss',
    'sa_mlp_distillation_loss',
    'sale_deepwalk_loss',
    'sale_mlp_distillation_loss'
]
