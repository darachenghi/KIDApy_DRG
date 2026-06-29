import numpy as np
import networkx as nx
import scipy.sparse as sp
from collections import Counter
from parser import Network, load_abundances


class DRG_c:
    def __init__(self):
        self.reduced_species = []
        self.reduced_rxns = []

    def reduce_net(self, net, cluster, sources, dropped = None, eps = 0.1):


        reduced_rxns = set()
        reduced_species = set()


        if cluster.ndim == 1:
            n_cluster = 1
        else:
            n_cluster = int(cluster.shape[0])

        for i in range(n_cluster):

            if n_cluster == 1:
                data_row = cluster
            else:
                data_row = cluster[i,:]

            reached_species_indices = set()

            reactions, species_map = self._get_reactions(net,data_row, dropped )
            source_indices = [species_map[s] for s in sources]
            
            env = self._get_env(data_row)
            reaction_rates = net.reaction_rates(reactions, env) #function to get list of reaction rates)
            state_data = self._get_state_data(data_row, species_map, dropped)

            R_mat = self._point_build_R_mat(reactions, species_map, reaction_rates, state_data, dropped)
            A_mat = self._build_A_mat(R_mat, eps )
            reached_species_idx = self._dfs(A_mat, source_indices)
            reached_species_indices.update(reached_species_idx)

            idx_to_species = {idx:species for species,idx in species_map.items()}

            for i, rxn in enumerate(reactions):
                found_idx = set(rxn["stoichiometric"].keys())

                if found_idx.issubset(reached_species_indices):
                    reduced_rxns.update(rxn)
                    species = [idx_to_species[i] for i in found_idx]
                    reduced_species.update(species)

        self.reduced_species = sorted(reduced_species)
        self.reduced_rxns = reduced_rxns
        return self.reduced_rxns

#HELPER FUNCTIONS

    def _get_env(self,data_row):
        "Gets environment from row"
        env_data = data_row[2:7]
        env_splice = np.delete(env_data, 2)
        env_var = [ 'nH', 'T', 'Av','uv_flux']
        env = {}
        for i, var in enumerate(env_var):
            env[var] = env_splice[i]
        return env
    
    def _get_state_data(self,data_row, species_map, dropped):
        "Removes dropped species and environment entries"
        state_data = data_row[7:]
        dropped_idx = [species_map[i] for i in dropped]
        for i in dropped_idx:
            state_data = np.delete(state_data, i)
        return state_data.T
    
    def _get_reactions(self,net, sample_row, dropped):
        "Assumes that the reaction network is the same for a states"
        sample_env = self._get_env(sample_row)
        reactions = net._select_multirange_entries(net.reactions, sample_env["T"]) 
        species_map = net.species_map 
        reactions = self._get_stoich(reactions, species_map, dropped)
        return reactions, species_map
    
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