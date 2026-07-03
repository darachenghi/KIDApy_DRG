import numpy as np
import networkx as nx
import scipy.sparse as sp
import json 
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time

class DRG_p:
    def __init__(self):
        self.reduced_species = []
        self.reduced_rxns = []
        self.results = {}

    def reduce_net(self,
                   net,
                   cluster,
                   sources,
                   eps,
                   dropped=None,
                   savedir=None,):

        ''' Iterates through each state in cluster and returns reduced network''

            Inputs:
                net: parsed reactions (dropped species removed and stoichiometric values added)
                cluster: array of states: idx | t | nH | T | Tgrain | Av | uv_flux | species 0 | ... | species 577
                sources: list of source species
                eps: list of epsilon values
                dropped: list of removed species
                savedir: directory to save reduced networks of form reduced_net_eps{eps}.json

            Outputs:
                reduced_rxns: list of reactions in reduced network for each epsilon value
                reduced_species: list of species in reduced network for each epsilon value '''

        if savedir is not None and not savedir.exists():
            raise FileNotFoundError(f"Save directory {savedir} does not exist.")

        scalar_eps = np.isscalar(eps)
        eps_list = [eps] if scalar_eps else list(eps)

        species_map = net.species_map

        with ProcessPoolExecutor() as executor:
            func = partial(self._state_reduce, net =net,species_map = species_map,sources =sources, eps_list = eps_list, dropped = dropped)
            out = executor.map(func, cluster)

            t1 = time.perf_counter()
            out = list(out)
            t2 = time.perf_counter()
            print(f"Unpack: {t2 -t1} seconds")
            found_ids = {e: set() for e in eps_list}

            start = time.perf_counter()
            for state_result in out:
                for e in eps_list:
                    found_ids[e].update(state_result[e])
            end = time.perf_counter()
            print(f"Merge: {end-start} seconds")

        idx_to_species = {idx: species for species, idx in species_map.items()}
        id_to_rxn = {}
        for rxn in net.reactions:
            id_to_rxn.setdefault(rxn["id"], rxn)

        self.results = {}
        for e in eps_list:
            rxns = [id_to_rxn[rxn_id] for rxn_id in found_ids[e]]
            species = set()
            for rxn in rxns:
                species.update(idx_to_species[idx] for idx in rxn["stoichiometric"].keys())
            self.results[e] = {"rxns": rxns, "species": species}

        if scalar_eps:
            e = eps_list[0]
            self.reduced_rxns = self.results[e]["rxns"]
            self.reduced_species = sorted(self.results[e]["species"])

            if savedir is not None:
                self._save_reduce_net(savedir, e, self.reduced_rxns, self.reduced_species)
            return self.reduced_rxns

        self.reduced_rxns = {e: self.results[e]["rxns"] for e in eps_list}
        self.reduced_species = {e: sorted(self.results[e]["species"]) for e in eps_list}

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
        '''Gets environment from row'''
        env_data = np.delete(data_row[2:7],2)
        env_var = [ 'nH', 'T', 'Av','uv_flux']
        env = {}
        for i, var in enumerate(env_var):
            env[var] = env_data[i]
        return env
    
    def _get_state_data(self,data_row, species_map, dropped):
        '''Removes dropped species and environment entries from data row'''
        if dropped == None:
            dropped = []
        state_data = data_row[7:]
        dropped_idx = [species_map[i] for i in dropped]
        state_data = np.delete(state_data, sorted(dropped_idx))
        return state_data.T
    
    def _get_reactions(self,net, data_row):
        '''Selects reactions and returns species map based on environment'''
        env = self._get_env(data_row)
        reactions = net._select_multirange_entries(net.reactions, env["T"]) 
        return reactions, env
    
    def _rxn_rate(self, net, rxn, env: dict) -> float:
        '''Calculates rxn rate for given rxn and environment'''
        T = float(env["T"])
        nH = float(env["nH"])
        Av = float(env["Av"])
        uv_flux = float(env["uv_flux"])
        Tcap_2body = bool(env.get("Tcap_2body", True))
        return net._calculate_rate(rxn, T, nH, Av, uv_flux, Tcap_2body) 

    def _point_build_R_mat(self, 
                    net, 
                    reactions: list, 
                    species_map: dict, 
                    env: dict, 
                    concs):
        
        '''Builds coefficient matrix at a single state'''

        excluded_rate = set(["Photon", "CR", "CRP"])

        n_species = len(species_map)
        den_vec = np.zeros(n_species)

        rows = []
        col = []
        data = []

        for rxn in reactions:

            wi = self._rxn_rate(net, rxn, env)

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

    def _state_reduce(self, data_row, net, species_map,sources, eps_list, dropped):
        '''Computes reaction ids reached for a single state, for each epsilon.'''

        reactions, env = self._get_reactions(net, data_row)
        source_indices = [species_map[s] for s in sources]

        state_data = self._get_state_data(data_row, species_map, dropped)
        R_mat = self._point_build_R_mat(net, reactions, species_map,env ,state_data)

        state_results = {}
        for e in eps_list:
            A_mat = self._build_A_mat(R_mat, e)
            reached = set(self._dfs(A_mat, source_indices))

            found_ids = {rxn["id"] for rxn in reactions
                         if set(rxn["stoichiometric"].keys()).issubset(reached)}

            state_results[e] = found_ids

        return state_results