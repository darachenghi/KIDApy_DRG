
import os
import warnings
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from typing import Dict


# ---------------------------------------------------------------------------
# Chemistry constants
# ---------------------------------------------------------------------------

#: Grain number density = grain_gas_ratio * nH
grain_gas_ratio: float = 1.8e-12

#: Cosmic-ray ionisation rate [s⁻¹]
zeta_cr: float = 1.3e-17



_PSEUDO_GRAIN_SPECIES = frozenset({"XH", "GRAIN-", "GRAIN0"})


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class assembly:

    def __init__(self, grains: bool = False):

        self.grains: bool = grains

        # Fields treated as external forcing (not ODE variables)
        self.external_fields = frozenset({"Photon", "CR", "CRP"})

        self.max_order: int = 0
        self._warned_high_order: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_operators(self,reactions, species_map, rates):

        N = len(species_map)
        A_rows, A_cols, A_data = [], [], []
        B_rows, B_cols, B_data = [], [], []

        for i,rxn in enumerate(reactions):
            r_idxs = [species_map[r] for r in rxn["reactants"]
                      if r in species_map]
            p_idxs = [species_map[p] for p in rxn["products"]
                      if p in species_map]

            order = len(r_idxs)
            if order == 0:
                continue

            k = rates[i]
            if k == 0.0:
                continue

            if order == 1:
                r1 = r_idxs[0]
                A_rows.append(r1); A_cols.append(r1); A_data.append(-k)
                for p in p_idxs:
                    A_rows.append(p); A_cols.append(r1); A_data.append(k)

            elif order == 2:
                r1, r2 = r_idxs
                col = r1 * N + r2
                B_rows.append(r1); B_cols.append(col); B_data.append(-k)
                B_rows.append(r2); B_cols.append(col); B_data.append(-k)
                for p in p_idxs:
                    B_rows.append(p); B_cols.append(col); B_data.append(k)

            else:
                if not self._warned_high_order:
                    warnings.warn(
                        "Reactions with >2 active reactants detected and "
                        "skipped (no third-order tensor is built).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._warned_high_order = True

        def _build(rows, cols, data, shape):
            if not data:
                return sp.csr_matrix(shape, dtype=np.float64)
            mat = sp.coo_matrix(
                (np.asarray(data, dtype=np.float64),
                 (np.asarray(rows, dtype=np.int64),
                  np.asarray(cols, dtype=np.int64))),
                shape=shape,
            )
            mat.sum_duplicates()
            return mat.tocsr()

        A = _build(A_rows, A_cols, A_data, (N, N))
        B = _build(B_rows, B_cols, B_data, (N, N * N))
        return A, B
    