import numpy as np
from collections import Counter
import networkx as nx
import scipy.sparse as sp

class DRG:
    def __init__(self):
        self.reduced_species = []
        self.reduced_rates = []
        self.reduced_rxns = []
        self.reduced_rxns_indices = []
    
    def reduce_net(self, reactions: list, 
                   species_map: dict, 
                   k: list, 
                   y, 
                   sources:list,
                   dropped = None ,
                   eps = 0.1):
        
        '''Reduces reaction network with DRG method'''
        
        source_indices = [species_map[s] for s in sources]

        reactions = self._get_stoich(reactions, species_map, dropped)
        
        idx_to_species = {idx:species for species,idx in species_map.items()}
        
        reached_species_indices = set()
        t_steps = int(y[0].shape[0])

        for t in range(t_steps):
            concs = y[:,t]
            R_mat = self._point_build_R_mat(reactions, species_map, k,concs, dropped)
            A_mat = self._build_A_mat(R_mat, eps )
            reached_species_idx = self._dfs(A_mat, source_indices)
            reached_species_indices.update(reached_species_idx)

        reduced_rxns = []
        reduced_rxns_indices = []
        reduced_species = set()
        reduced_rates = []

        for i, rxn in enumerate(reactions):
            found_idx = set(rxn["stoichiometric"].keys())

            if found_idx.issubset(reached_species_indices):
                reduced_rxns.append(rxn)
                reduced_rates.append(k[i])
                reduced_rxns_indices.append(i)
                species = [idx_to_species[i] for i in found_idx]
                reduced_species.update(species)

        self.reduced_rates = reduced_rates
        self.reduced_species = sorted(reduced_species)
        self.reduced_rxns = reduced_rxns
        self.reduced_rxns_indices = reduced_rxns_indices
        return self.reduced_rxns
    
    #HELPER FUNCTIONS

    def _get_stoich(self, reactions, species_map, dropped):

        "adds stoichiometric coefficient dictionary to reactions"

        excluded_rate = ["Photon", "CR", "CRP"]
        dropped = set(excluded_rate) | set(dropped or [])

        for rxn in reactions:

            reactant_counts = Counter(rxn["reactants"])
            product_counts = Counter(rxn["products"])

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

            rxn["stoichiometric"] = stoic

        return reactions

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

        rows = []
        col = []
        data = []

        for i,rxn in enumerate(reactions):

            wi = k[i]

            for reactant in rxn["reactants"]:
                if reactant in excluded_rate:
                    continue

                wi *= concs[species_map[reactant]]
            
            if wi == 0: 
                continue 

            stoic = rxn["stoichiometric"]

            for idx_a in stoic:
                rate_prod = np.abs(stoic[idx_a] * wi)

                den_vec[idx_a] += rate_prod

                for idx_b in stoic:
                    if idx_a == idx_b:
                        continue

                    rows.append(idx_a)
                    col.append(idx_b)
                    data.append(rate_prod)
        
        num_mat  = sp.coo_matrix((data,(rows,col)), shape = (n_species,n_species), dtype = np.float64)
        num_mat.sum_duplicates()

        R_mat = num_mat.copy()
        R_mat.data /= den_vec[R_mat.row]
        return R_mat
    
    def _build_A_mat(self, R_mat,eps = 0.1):
        '''Builds adjacency matrix from R_mat'''
        A_mat = R_mat.copy()
        A_mat.data = (np.abs(A_mat.data)>= eps).astype(int)
        A_mat.eliminate_zeros()
        return A_mat
    
    def _dfs(self, A_mat, source_indices: list):
        '''Conducts depth first search of directed graph, given source terms'''

        found_species_indices = set()

        G = nx.from_scipy_sparse_array(A_mat, create_using=nx.DiGraph)

        for s in source_indices:
            if s in found_species_indices:
                continue

            new_species = list(nx.dfs_preorder_nodes(G,s))
            found_species_indices.update(new_species)

        found_species_indices = list(found_species_indices)
        return found_species_indices