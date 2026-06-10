import numpy as np
from collections import Counter
import networkx as nx
import scipy.sparse as sp

class DRG:
    def __init__(self):
        self.reduced_species = []
        self.reached_species_indices = []
        self.reduced_rxns = []
        self.reduced_rxns_indices = []
    
    def build_A_mat(self, R_mat,eps = 0.1):
        '''Builds adjacency matrix from R_mat'''
        A_mat = R_mat.copy()
        A_mat.data = (np.abs(A_mat.data)>= eps).astype(int)
        A_mat.eliminate_zeros()
        return A_mat
    
    def dfs(self, A_mat, source_indices: list):
        '''Conducts depth first search of directed graph, given source term'''
        reached_species_indices = set()
        G = nx.from_scipy_sparse_array(A_mat, create_using=nx.DiGraph)

        for s in source_indices:
            reached_species = list(nx.dfs_preorder_nodes(G,s))
            reached_species_indices.update(reached_species)

        reached_species_indices = list(reached_species_indices)
        return reached_species_indices
    
    def reduce_net(self, reactions: list, 
                   species_map: dict, 
                   k: list, 
                   y, 
                   source_indices:list,
                   dropped = None ,
                   eps = 0.1):
        
        '''Reduces reaction network with DRG method'''

        reached_species_indices = set()
        t_steps = int(y[0].shape[0])

        for t in range(t_steps):
            concs = y[:,t]
            R_mat = self._point_build_R_mat(reactions, species_map, k,concs, dropped)
            A_mat = self.build_A_mat(R_mat, eps )
            reached_species_idx = self.dfs(A_mat, source_indices)
            reached_species_indices.update(reached_species_idx)

        index_to_species = {idx:species for species,idx in species_map.items()}
        reached_species = [index_to_species[i] for i in reached_species_indices]

        reduced_rxns = []
        reduced_rxns_indices = []

        for idx, reaction in enumerate(reactions):
            reactants = reaction.get("reactants", [])
            products = reaction.get("products", [])
            if any(s in reached_species for s in reactants) or any(s in reached_species for s in products):
                reduced_rxns.append(reaction)
                reduced_rxns_indices.append(idx)

        self.reduced_species = reached_species
        self.reduced_rxns = reduced_rxns
        self.reduced_rxns_indices = reduced_rxns_indices
        return self.reduced_rxns
    
    def _point_build_R_mat(self, reactions: list, 
                    species_map: dict, 
                    k: list, 
                    concs, 
                    dropped = None):
        
        '''Builds coefficient matrix at a single state'''
        if dropped is None:
            dropped = []

        excluded_rate = ["Photon", "CR", "CRP"]
        dropped = excluded_rate + dropped

        n_species = len(species_map)
        den_vec = np.zeros(n_species)
        num_mat = sp.dok_matrix((n_species,n_species), dtype=float)

        for i, reaction in enumerate(reactions):
            reactant_counts = Counter(reaction["reactants"])
            product_counts = Counter(reaction["products"])

            wi = k[i]

            for reactant, order in reactant_counts.items():
                if reactant in excluded_rate:
                    continue
                wi *= concs[species_map[reactant]] ** order
            
            if wi == 0: 
                continue 

            reaction_species = set(reactant_counts)
            reaction_species.update(product_counts)

            stoic = {}

            for species in reaction_species:
                if species in dropped:
                    continue
                idx = species_map[species]

                stoic[idx] = (
                    reactant_counts[species]
                    - product_counts[species]
                )

            for idx_a in stoic:
                rate_prod = np.abs(stoic[idx_a] * wi)

                den_vec[idx_a] += rate_prod

                for idx_b in stoic:
                    if idx_a == idx_b:
                        continue

                    num_mat[idx_a, idx_b] += rate_prod
        
        num_mat  = num_mat.tocsr()
        R_mat = num_mat.copy()

        nonzero, _ = R_mat.nonzero()
        R_mat.data /= den_vec[nonzero]

        return R_mat
