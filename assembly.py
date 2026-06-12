
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

    def __init__(self, reactions,species,ks, grains: bool = False):

        self.grains: bool = grains

        self.species: list = []
        self.species_map: dict = {}
        self.reactions = reactions
        self.rates = ks

        # Fields treated as external forcing (not ODE variables)
        self.external_fields = frozenset({"Photon", "CR", "CRP"})

        self.max_order: int = 0
        self._warned_high_order: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_operators(self, env: dict):
        """
        Build the sparse ODE tensors ``A`` and ``B`` for the current
        environment.

        Parameters
        ----------
        env : dict
            Required keys: T, nH, Av, uv_flux.
            Optional: Tcap_2body (bool, default True).

        Returns
        -------
        A : scipy.sparse.csr_matrix, shape (N, N)
        B : scipy.sparse.csr_matrix, shape (N, N*N)
        """
        T = float(env["T"])
        nH = float(env["nH"])
        Av = float(env["Av"])
        uv_flux = float(env["uv_flux"])
        Tcap_2body = bool(env.get("Tcap_2body", True))

        N = len(self.species)
        A_rows, A_cols, A_data = [], [], []
        B_rows, B_cols, B_data = [], [], []

        reactions = self.reactions

        for rxn in reactions:
            r_idxs = [self.species_map[r] for r in rxn["reactants"]
                      if r in self.species_map]
            p_idxs = [self.species_map[p] for p in rxn["products"]
                      if p in self.species_map]

            order = len(r_idxs)
            if order == 0:
                continue

            k = self._calculate_rate(rxn, T, nH, Av, uv_flux, Tcap_2body)
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
    