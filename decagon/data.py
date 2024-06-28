import os
from os import path as osp
from typing import NamedTuple

import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch
import torch.nn.functional as F
from dgl import DGLGraph, heterograph, load_graphs, save_graphs
from dgl.data import DGLDataset
from dgl.data.utils import download, extract_archive
from torch import Tensor

# I promise to not use chained assignments
pd.options.mode.chained_assignment = None


# NOTE: The complexity of the processing functions below is largely due
# to trying to prevent data leakage between the train, validation, and test sets.
class UnipartiteResult(NamedTuple):
    src: Tensor
    dst: Tensor
    neg_src: Tensor
    neg_dst: Tensor
    train_mask: Tensor
    val_mask: Tensor
    test_mask: Tensor


def process_unipartite(
    src: Tensor,
    dst: Tensor,
    src_size: int,
    dst_size: int,
    train_ratio: float,
    val_ratio: float,
) -> UnipartiteResult:
    assert train_ratio + val_ratio < 1
    assert src.ndim == dst.ndim == 1
    assert src.size(0) == dst.size(0)

    # *Process adjacency matrix*
    adj = sps.coo_array(
        (torch.ones_like(src, dtype=torch.bool), (src, dst)),
        shape=(src_size, dst_size),
    )  # [N N] E

    # Remove self-loops
    adj.data[adj.row == adj.col] = 0
    adj.eliminate_zeros()

    # Realize an edge regardless of its direction
    adj_undir: sps.coo_array = (adj + adj.transpose()).tocoo()  # [N N] 2E
    adj = sps.triu(adj_undir).tocoo()  # [N N] E

    # *Generate negative samples*
    src = torch.from_numpy(adj_undir.row)  # [2E]
    dst = torch.from_numpy(adj_undir.col)  # [2E]
    neg_src = torch.clone(src)  # [2E]
    neg_dst = torch.randint(0, dst_size, (len(neg_src),))  # [2E]

    # *Create train, val, and test edge masks*
    n_edges: int = adj.data.size
    n_edges_train = int(train_ratio * n_edges)
    n_edges_val = int(val_ratio * n_edges)

    edge_indices = torch.randperm(n_edges)  # [E]
    train_indices = edge_indices[:n_edges_train]  # [E_train]
    val_indices = edge_indices[n_edges_train : n_edges_train + n_edges_val]  # [E_val]
    test_indices = edge_indices[n_edges_train + n_edges_val :]  # [E_test]

    # Mask both directions of an edge if either direction is masked
    train_adj = sps.coo_array(
        (adj.data[train_indices], (adj.row[train_indices], adj.col[train_indices])),
        shape=adj.shape,
    )  # [N N] E_train
    val_adj = sps.coo_array(
        (adj.data[val_indices], (adj.row[val_indices], adj.col[val_indices])),
        shape=adj.shape,
    )  # [N N] E_val
    test_adj = sps.coo_array(
        (adj.data[test_indices], (adj.row[test_indices], adj.col[test_indices])),
        shape=adj.shape,
    )  # [N N] E_test

    train_adj_undir = (train_adj + train_adj.transpose()).tocoo()  # [N N] 2E_train
    val_adj_undir = (val_adj + val_adj.transpose()).tocoo()  # [N N] 2E_val
    test_adj_undir = (test_adj + test_adj.transpose()).tocoo()  # [N N] 2E_test

    # Compute the masks
    # TODO: Make this vectorized?
    all_edges = np.stack(
        [adj_undir.row, adj_undir.col],
        axis=1,
    )  # [2E 2]
    train_edges = np.stack(
        [train_adj_undir.row, train_adj_undir.col],
        axis=1,
    )  # [2E_train 2]
    val_edges = np.stack(
        [val_adj_undir.row, val_adj_undir.col],
        axis=1,
    )  # [2E_val 2]
    test_edges = np.stack(
        [test_adj_undir.row, test_adj_undir.col],
        axis=1,
    )  # [2E_test 2]

    train_edges_set = set(map(tuple, train_edges))
    val_edges_set = set(map(tuple, val_edges))
    test_edges_set = set(map(tuple, test_edges))

    train_mask = torch.from_numpy(
        np.array([(edge[0], edge[1]) in train_edges_set for edge in all_edges])
    ).to(torch.bool)  # [2E]
    val_mask = torch.from_numpy(
        np.array([(edge[0], edge[1]) in val_edges_set for edge in all_edges])
    ).to(torch.bool)  # [2E]
    test_mask = torch.from_numpy(
        np.array([(edge[0], edge[1]) in test_edges_set for edge in all_edges])
    ).to(torch.bool)  # [2E]

    return UnipartiteResult(
        src=src,
        dst=dst,
        neg_src=neg_src,
        neg_dst=neg_dst,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


class BipartiteResult(NamedTuple):
    src: Tensor
    dst: Tensor
    neg_src: Tensor
    neg_dst: Tensor
    train_mask: Tensor
    val_mask: Tensor
    test_mask: Tensor
    src_inv: Tensor
    dst_inv: Tensor
    neg_src_inv: Tensor
    neg_dst_inv: Tensor
    train_mask_inv: Tensor
    val_mask_inv: Tensor
    test_mask_inv: Tensor


def process_bipartite(
    src: Tensor,
    dst: Tensor,
    src_size: int,
    dst_size: int,
    train_ratio: float,
    val_ratio: float,
) -> BipartiteResult:
    assert train_ratio + val_ratio < 1
    assert src.ndim == dst.ndim == 1

    # *Generate negative samples*
    neg_src = torch.clone(src)  # [E]
    neg_dst = torch.randint(0, dst_size, (len(neg_src),))  # [E]

    src_inv = torch.clone(dst)
    dst_inv = torch.clone(src)
    neg_src_inv = torch.clone(src_inv)  # [E]
    neg_dst_inv = torch.randint(0, src_size, (len(neg_src_inv),))  # [E]

    # *Create train, val, and test edge masks*
    n_edges: int = len(src)
    n_edges_train = int(train_ratio * n_edges)
    n_edges_val = int(val_ratio * n_edges)

    edge_indices = torch.randperm(n_edges)  # [E]
    train_indices = edge_indices[:n_edges_train]  # [E_train]
    val_indices = edge_indices[n_edges_train : n_edges_train + n_edges_val]  # [E_val]
    test_indices = edge_indices[n_edges_train + n_edges_val :]  # [E_test]

    train_mask = torch.zeros(n_edges, dtype=torch.bool)  # [E]
    val_mask = torch.zeros(n_edges, dtype=torch.bool)  # [E]
    test_mask = torch.zeros(n_edges, dtype=torch.bool)  # [E]
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    train_mask_inv = torch.clone(train_mask)  # [E]
    val_mask_inv = torch.clone(val_mask)  # [E]
    test_mask_inv = torch.clone(test_mask)  # [E]

    return BipartiteResult(
        src=src,
        dst=dst,
        neg_src=neg_src,
        neg_dst=neg_dst,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        src_inv=src_inv,
        dst_inv=dst_inv,
        neg_src_inv=neg_src_inv,
        neg_dst_inv=neg_dst_inv,
        train_mask_inv=train_mask_inv,
        val_mask_inv=val_mask_inv,
        test_mask_inv=test_mask_inv,
    )


def process(
    dsi_df: pd.DataFrame,
    ppi_df: pd.DataFrame,
    dpi_df: pd.DataFrame,
    ddi_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> tuple[DGLGraph, DGLGraph]:
    assert train_ratio + val_ratio < 1

    # *Drop side-effects that occur in fewer than 500 drug pairs*
    counts = ddi_df["Polypharmacy Side Effect"].value_counts()
    mask = ddi_df["Polypharmacy Side Effect"].isin(counts[counts >= 500].index)
    ddi_df = ddi_df[mask]

    # *Map drugs, proteins, mono side-effects, and combo side-effects to indices*
    all_drugs = pd.concat([ddi_df["STITCH 1"], ddi_df["STITCH 2"]], axis=0)
    all_proteins = pd.concat(
        [ppi_df["Gene 1"], ppi_df["Gene 2"], dpi_df["Gene"]], axis=0
    )
    all_mono_se = dsi_df["Individual Side Effect"]
    all_combo_se = ddi_df["Polypharmacy Side Effect"]

    unique_drugs: list[str] = list(np.sort(all_drugs.unique()))
    unique_proteins: list[str] = list(np.sort(all_proteins.unique()))
    unique_mono_ses: list[str] = list(np.sort(all_mono_se.unique()))
    unique_combo_ses: list[str] = list(np.sort(all_combo_se.unique()))

    drug2id = {d: i for i, d in enumerate(unique_drugs)}
    protein2id = {p: i for i, p in enumerate(unique_proteins)}
    mono_se2id = {mse: i for i, mse in enumerate(unique_mono_ses)}
    combo_se2id = {cse: i for i, cse in enumerate(unique_combo_ses)}

    dsi_df["STITCH"] = dsi_df["STITCH"].map(drug2id)
    dsi_df["Individual Side Effect"] = dsi_df["Individual Side Effect"].map(mono_se2id)
    ppi_df["Gene 1"] = ppi_df["Gene 1"].map(protein2id)
    ppi_df["Gene 2"] = ppi_df["Gene 2"].map(protein2id)
    dpi_df["STITCH"] = dpi_df["STITCH"].map(drug2id)
    dpi_df["Gene"] = dpi_df["Gene"].map(protein2id)
    ddi_df["STITCH 1"] = ddi_df["STITCH 1"].map(drug2id)
    ddi_df["STITCH 2"] = ddi_df["STITCH 2"].map(drug2id)
    ddi_df["Polypharmacy Side Effect"] = ddi_df["Polypharmacy Side Effect"].map(
        combo_se2id
    )

    # *Construct normalized drug features based on one-hot encoded mono side-effects*
    drug_feat = torch.zeros((len(drug2id), len(mono_se2id)), dtype=torch.float)
    drug_indices = torch.from_numpy(dsi_df["STITCH"].to_numpy())
    mse_indices = torch.from_numpy(dsi_df["Individual Side Effect"].to_numpy())
    drug_feat[drug_indices, mse_indices] = 1  # [N_drug F_drug]
    drug_feat = F.normalize(drug_feat, p=2, dim=1)  # [N_drug F_drug]

    # *Construct positive and negative heterographs*
    # (protein, ppi, protein)
    ppi_src = torch.from_numpy(ppi_df["Gene 1"].to_numpy())  # [N_ppi]
    ppi_dst = torch.from_numpy(ppi_df["Gene 2"].to_numpy())  # [N_ppi]
    ppi_src_size = len(protein2id)
    ppi_dst_size = len(protein2id)
    ppi_result = process_unipartite(
        src=ppi_src,
        dst=ppi_dst,
        src_size=ppi_src_size,
        dst_size=ppi_dst_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    # (drug, dpi, protein) and (protein, pdi, drug)
    dpi_src = torch.from_numpy(dpi_df["STITCH"].to_numpy())  # [N_dpi]
    dpi_dst = torch.from_numpy(dpi_df["Gene"].to_numpy())  # [N_dpi]
    dpi_src_size = len(drug2id)
    dpi_dst_size = len(protein2id)
    dpi_result = process_bipartite(
        src=dpi_src,
        dst=dpi_dst,
        src_size=dpi_src_size,
        dst_size=dpi_dst_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    # (drug, ddi_cse, drug)
    cse2ddi_result: dict[str, UnipartiteResult] = {}
    for cse, cse_id in combo_se2id.items():
        # Filter the ddi dataframe for this combo side-effect
        filtered_ddi_df = ddi_df[ddi_df["Polypharmacy Side Effect"] == cse_id]

        # Proceed as usual
        ddi_src = torch.from_numpy(
            filtered_ddi_df["STITCH 1"].to_numpy()
        )  # [N_ddi_cse]
        ddi_dst = torch.from_numpy(
            filtered_ddi_df["STITCH 2"].to_numpy()
        )  # [N_ddi_cse]
        ddi_src_size = len(drug2id)
        ddi_dst_size = len(drug2id)
        cse2ddi_result[cse] = process_unipartite(
            src=ddi_src,
            dst=ddi_dst,
            src_size=ddi_src_size,
            dst_size=ddi_dst_size,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

    n_nodes_dict = {"drug": len(drug2id), "protein": len(protein2id)}
    data_dict: dict[tuple[str, str, str], tuple[Tensor, Tensor]] = {}
    neg_data_dict: dict[tuple[str, str, str], tuple[Tensor, Tensor]] = {}

    data_dict[("protein", "ppi", "protein")] = (ppi_result.src, ppi_result.dst)
    neg_data_dict[("protein", "ppi", "protein")] = (
        ppi_result.neg_src,
        ppi_result.neg_dst,
    )

    data_dict[("drug", "dpi", "protein")] = (dpi_result.src, dpi_result.dst)
    neg_data_dict[("drug", "dpi", "protein")] = (
        dpi_result.neg_src,
        dpi_result.neg_dst,
    )

    data_dict[("protein", "pdi", "drug")] = (dpi_result.src_inv, dpi_result.dst_inv)
    neg_data_dict[("protein", "pdi", "drug")] = (
        dpi_result.neg_src_inv,
        dpi_result.neg_dst_inv,
    )

    for cse in combo_se2id.keys():
        ddi_result = cse2ddi_result[cse]
        data_dict[("drug", f"ddi_{cse}", "drug")] = (ddi_result.src, ddi_result.dst)
        neg_data_dict[("drug", f"ddi_{cse}", "drug")] = (
            ddi_result.neg_src,
            ddi_result.neg_dst,
        )

    # Initialize graph objects
    g: DGLGraph = heterograph(data_dict, n_nodes_dict)  # type: ignore
    neg_g: DGLGraph = heterograph(neg_data_dict, n_nodes_dict)  # type: ignore

    # Positive Node features and identities
    g.nodes["drug"].data["feat"] = drug_feat  # [N_drug F_drug]
    g.nodes["drug"].data["identity"] = torch.arange(g.num_nodes("drug"))  # [N_drug]
    g.nodes["protein"].data["identity"] = torch.arange(
        g.num_nodes("protein")
    )  # [N_protein]

    # Negative node features and identities
    neg_g.nodes["drug"].data["feat"] = drug_feat  # [N_drug F_drug]
    neg_g.nodes["drug"].data["identity"] = torch.arange(g.num_nodes("drug"))  # [N_drug]
    neg_g.nodes["protein"].data["identity"] = torch.arange(
        g.num_nodes("protein")
    )  # [N_protein]

    # Positive ppi masks
    g.edges["ppi"].data["train_mask"] = ppi_result.train_mask
    g.edges["ppi"].data["val_mask"] = ppi_result.val_mask
    g.edges["ppi"].data["test_mask"] = ppi_result.test_mask

    # Negative ppi masks
    neg_g.edges["ppi"].data["train_mask"] = ppi_result.train_mask
    neg_g.edges["ppi"].data["val_mask"] = ppi_result.val_mask
    neg_g.edges["ppi"].data["test_mask"] = ppi_result.test_mask

    # Positive dpi masks
    g.edges["dpi"].data["train_mask"] = dpi_result.train_mask
    g.edges["dpi"].data["val_mask"] = dpi_result.val_mask
    g.edges["dpi"].data["test_mask"] = dpi_result.test_mask

    # Negative dpi masks
    neg_g.edges["dpi"].data["train_mask"] = dpi_result.train_mask
    neg_g.edges["dpi"].data["val_mask"] = dpi_result.val_mask
    neg_g.edges["dpi"].data["test_mask"] = dpi_result.test_mask

    # Positive pdi masks
    g.edges["pdi"].data["train_mask"] = dpi_result.train_mask_inv
    g.edges["pdi"].data["val_mask"] = dpi_result.val_mask_inv
    g.edges["pdi"].data["test_mask"] = dpi_result.test_mask_inv

    # Negative pdi masks
    neg_g.edges["pdi"].data["train_mask"] = dpi_result.train_mask_inv
    neg_g.edges["pdi"].data["val_mask"] = dpi_result.val_mask_inv
    neg_g.edges["pdi"].data["test_mask"] = dpi_result.test_mask_inv

    # ddi masks
    for cse in combo_se2id.keys():
        ddi_result = cse2ddi_result[cse]

        # Positive ddi masks
        g.edges[f"ddi_{cse}"].data["train_mask"] = ddi_result.train_mask
        g.edges[f"ddi_{cse}"].data["val_mask"] = ddi_result.val_mask
        g.edges[f"ddi_{cse}"].data["test_mask"] = ddi_result.test_mask

        # Negative ddi masks
        neg_g.edges[f"ddi_{cse}"].data["train_mask"] = ddi_result.train_mask
        neg_g.edges[f"ddi_{cse}"].data["val_mask"] = ddi_result.val_mask
        neg_g.edges[f"ddi_{cse}"].data["test_mask"] = ddi_result.test_mask

    return g, neg_g


def split_graph(g: DGLGraph) -> tuple[DGLGraph, DGLGraph, DGLGraph]:
    assert "train_mask" in g.edata
    assert "val_mask" in g.edata
    assert "test_mask" in g.edata

    etype2train_eids: dict[str, Tensor] = {}
    etype2val_eids: dict[str, Tensor] = {}
    etype2test_eids: dict[str, Tensor] = {}

    for etype in g.etypes:
        train_mask: Tensor = g.edges[etype].data["train_mask"].to(torch.bool)  # [E]
        val_mask: Tensor = g.edges[etype].data["val_mask"].to(torch.bool)  # [E]
        test_mask: Tensor = g.edges[etype].data["test_mask"].to(torch.bool)  # [E]

        etype2train_eids[etype] = g.edges(form="eid", etype=etype)[
            train_mask
        ]  # [E_train]
        etype2val_eids[etype] = g.edges(form="eid", etype=etype)[val_mask]  # [E_val]
        etype2test_eids[etype] = g.edges(form="eid", etype=etype)[test_mask]  # [E_test]

    train_g = dgl.edge_subgraph(g, etype2train_eids, relabel_nodes=False)
    val_g = dgl.edge_subgraph(g, etype2val_eids, relabel_nodes=False)
    test_g = dgl.edge_subgraph(g, etype2test_eids, relabel_nodes=False)

    return train_g, val_g, test_g


class DecagonDataset(DGLDataset):
    def __init__(
        self,
        root_dir: str,
        train_ratio: float,
        val_ratio: float,
        force_reload: bool = False,
    ) -> None:
        assert train_ratio + val_ratio < 1

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        # Hack. First element is the positive graph, second the negative.
        self.glist: list[DGLGraph] = []

        super().__init__(
            name="decagon",
            url="https://snap.stanford.edu/decagon/",
            raw_dir=osp.join(root_dir, "raw"),
            save_dir=osp.join(root_dir, "processed"),
            force_reload=force_reload,
        )

    def download(self) -> None:
        print("Downloading...")

        filenames = [
            "bio-decagon-ppi",
            "bio-decagon-targets",
            "bio-decagon-combo",
            "bio-decagon-mono",
        ]
        for filename in filenames:
            full_name = osp.join(self.raw_dir, filename)

            # Check if the file already exists
            if osp.exists(full_name + ".csv"):
                print(f"File {full_name}.csv already exists. Skipping download.")
                continue

            # Download and extract
            full_url: str = osp.join(self.url, filename + ".tar.gz")  # type: ignore
            download(url=full_url, path=self.raw_dir)
            extract_archive(full_name + ".tar.gz", target_dir=self.raw_dir)

            # Clean up
            os.remove(osp.join(self.raw_dir, "._" + filename + ".csv"))
            os.remove(osp.join(self.raw_dir, filename + ".tar.gz"))

    def process(self) -> None:
        print("Processing...")

        # *Construct positive and negative graph objects from the raw data*
        dsi_df = pd.read_csv(osp.join(self.raw_dir, "bio-decagon-mono.csv"))
        ppi_df = pd.read_csv(osp.join(self.raw_dir, "bio-decagon-ppi.csv"))
        dpi_df = pd.read_csv(osp.join(self.raw_dir, "bio-decagon-targets.csv"))
        ddi_df = pd.read_csv(osp.join(self.raw_dir, "bio-decagon-combo.csv"))

        self.g, self.neg_g = process(
            dsi_df=dsi_df.astype(str),
            ppi_df=ppi_df.astype(str),
            dpi_df=dpi_df.astype(str),
            ddi_df=ddi_df.astype(str),
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
        )
        self.glist = [self.g, self.neg_g]

    def __getitem__(self, idx: int) -> DGLGraph:  # type: ignore
        return self.glist[idx]

    def __len__(self) -> int:  # type: ignore
        return len(self.glist)

    def save(self) -> None:
        print(f"Saving processed data to {self.save_path}.")
        save_graphs(self.save_path, self.glist)

    def load(self) -> None:
        print(f"Loading processed data from {self.save_path}.")
        glist, _ = load_graphs(self.save_path)
        self.glist = glist

    def has_cache(self) -> bool:  # type: ignore
        result = osp.exists(self.save_path)
        if not result:
            print(f"Processed data not found in {self.save_path}.")
        else:
            print(f"Processed data found in {self.save_path}.")

        return result
