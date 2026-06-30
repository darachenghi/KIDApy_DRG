import numpy as np
import networkx as nx
import scipy.sparse as sp
import json 

from collections import Counter
from parser import Network


class DRG_c:
    def __init__(self):
        self.reduced_species = []
        self.reduced_rxns = []

    def reduce_net(self, net, cluster, sources, eps:list, dropped=None, savedir=None):
        scalar_eps = np.isscalar(eps)
        eps_list = [eps] if scalar_eps else list(eps)

        results = {e: {"seen": set(), "rxns": [], "species": set()} for e in eps_list}

        if cluster.ndim == 1:
            n_cluster = 1
        else:
            n_cluster = int(cluster.shape[0])

        for j in range(n_cluster):

            if n_cluster == 1:
                data_row = cluster
            else:
                data_row = cluster[j,:]

            reactions, species_map = self._get_reactions(net, data_row, dropped)
            source_indices = [species_map[s] for s in sources]

            env = self._get_env(data_row)
            reaction_rates = net.reaction_rates(reactions, env)
            state_data = self._get_state_data(data_row, species_map, dropped)

            R_mat = self._point_build_R_mat(reactions, species_map, reaction_rates, state_data, dropped)

            idx_to_species = {idx: species for species, idx in species_map.items()}

            for e in eps_list:
                A_mat = self._build_A_mat(R_mat, e)
                reached = set(self._dfs(A_mat, source_indices))
                r = results[e]

                for rxn in reactions:
                    rxn_id = rxn["id"]
                    if rxn_id in r["seen"]:
                        continue
                    found_idx = set(rxn["stoichiometric"].keys())
                    if found_idx.issubset(reached):
                        r["seen"].add(rxn_id)
                        r["rxns"].append(rxn)
                        r["species"].update(idx_to_species[idx] for idx in found_idx)

        if scalar_eps:
            e = eps_list[0]
            self.reduced_rxns = results[e]["rxns"]
            self.reduced_species = sorted(results[e]["species"])
            if savedir is not None:
                self._save_reduce_net(savedir, e, self.reduced_rxns, self.reduced_species)
            return self.reduced_rxns

        self.reduced_rxns = {e: results[e]["rxns"] for e in eps_list}
        self.reduced_species = {e: sorted(results[e]["species"]) for e in eps_list}
        if savedir is not None:
            for e in eps_list:
                self._save_reduce_net(savedir, e, self.reduced_rxns[e], self.reduced_species[e])

        for e in eps_list:
            print(f'Epsilon: {e}')
            print(f'Number of reactions in reduced network: {len(self.reduced_rxns[e])}')
            print(f'Number of species in reduced network: {len(self.reduced_species[e])}\n')
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
        if dropped == None:
            dropped = []
        state_data = data_row[7:]
        dropped_idx = [species_map[i] for i in dropped]
        state_data = np.delete(state_data, sorted(dropped_idx))
        return state_data.T
    
    def _get_reactions(self,net, data_row, dropped):
        env = self._get_env(data_row)
        reactions = net._select_multirange_entries(net.reactions, env["T"]) 
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

    def _save_reduce_net(self, savedir, eps, reduced_reactions, species):
        file_name = f"reduced_net_eps{eps}.json"
        file_path = savedir/file_name

        data = {"epsilon": eps, "reactions": reduced_reactions,
                "species": species}
            
        with open(file_path, "w") as f:
            json.dump(data, f, indent = 2)