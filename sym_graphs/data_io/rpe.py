from typing import Literal

import torch
from torch_geometric.data.data import Data
from torch_geometric.utils import to_networkx

from sym_graphs.correlators.k_correlators import Correlator
from sym_graphs.utils.utils import yaml_to_dotdict


class MyDataProcessor:
    """Computes the dynamics of the correlation matrix of the input data point"""

    def __init__(
        self,
        cfg_file: str,
        size: int,
        filter_type: Literal["exact", "upto"] = "exact",
        emulator: Literal["MPS", "SV"] = "SV",
        verbose: bool = False,
    ):
        """Initiate the data filer and transform

        Args:
            cfg_file (str): path to the configuration file for the correlation dynamics
            size (int): threshold size
            filter_type (str): the type of filtration
        """
        self.cfg_file = cfg_file
        self.size = size
        self.filter_type = filter_type
        self.emulator = emulator
        self.verbose = verbose

    def compute_correlation(self, data):
        cfg = yaml_to_dotdict(self.cfg_file)
        G = to_networkx(data, to_undirected=True)
        Corr = Correlator(G, cfg)
        res = Corr.emulate_non_UD(self.emulator, return_state=False, verbose=self.verbose)
        corr_dyn = torch.stack(res.correlation_matrix).real
        T, N, _ = corr_dyn.shape
        data.corr_dyn = corr_dyn.permute(1, 2, 0).reshape(N * N, T)
        return data

    def filter_out(self, data: Data) -> bool:
        if self.filter_type == "exact":
            return data.num_nodes == self.size
        if self.filter_type == "upto":
            return data.num_nodes <= self.size
        raise ValueError("Filter type is not recognized")
