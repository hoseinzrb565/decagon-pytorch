import os.path as osp
import re
from typing import NamedTuple

import pandas as pd
import pytest
import torch
from decagon import data
from dgl import DGLGraph
from scipy import sparse as sps


class GraphsResult(NamedTuple):
    train_pos_g: DGLGraph
    train_neg_g: DGLGraph
    val_pos_g: DGLGraph
    val_neg_g: DGLGraph
    test_pos_g: DGLGraph
    test_neg_g: DGLGraph


@pytest.fixture(scope="module")
def graphs() -> GraphsResult:
    dsi_df = pd.read_csv(osp.join("./data/raw", "bio-decagon-mono.csv"))
    ppi_df = pd.read_csv(osp.join("./data/raw", "bio-decagon-ppi.csv"))
    dpi_df = pd.read_csv(osp.join("./data/raw", "bio-decagon-targets.csv"))
    ddi_df = pd.read_csv(osp.join("./data/raw", "bio-decagon-combo.csv"))

    pos_g, neg_g = data.process(
        dsi_df=dsi_df.astype(str),
        ppi_df=ppi_df.astype(str),
        dpi_df=dpi_df.astype(str),
        ddi_df=ddi_df.astype(str),
        train_ratio=0.8,
        val_ratio=0.1,
    )

    train_pos_g, val_pos_g, test_pos_g = data.split_graph(pos_g)
    train_neg_g, val_neg_g, test_neg_g = data.split_graph(neg_g)

    return GraphsResult(
        train_pos_g=train_pos_g,
        train_neg_g=train_neg_g,
        val_pos_g=val_pos_g,
        val_neg_g=val_neg_g,
        test_pos_g=test_pos_g,
        test_neg_g=test_neg_g,
    )


def test_nodes(graphs: GraphsResult) -> None:
    for graph in graphs:
        ntypes = graph.ntypes

        assert set(ntypes) == {"drug", "protein"}
        assert graph.num_nodes("drug") == 645
        assert graph.num_nodes("protein") == 19089


def test_edge_types(graphs: GraphsResult) -> None:
    assert (
        graphs.train_pos_g.canonical_etypes
        == graphs.train_neg_g.canonical_etypes
        == graphs.val_pos_g.canonical_etypes
        == graphs.val_neg_g.canonical_etypes
        == graphs.test_pos_g.canonical_etypes
        == graphs.test_neg_g.canonical_etypes
    )
    c_etypes: list[tuple[str, str, str]] = graphs.train_pos_g.canonical_etypes  # type: ignore

    ddi_c_etypes = [
        c_etype for c_etype in c_etypes if c_etype[0] == c_etype[2] == "drug"
    ]
    dpi_c_etypes = [
        c_etype
        for c_etype in c_etypes
        if c_etype[0] == "drug" and c_etype[2] == "protein"
    ]
    pdi_c_etypes = [
        c_etype
        for c_etype in c_etypes
        if c_etype[0] == "protein" and c_etype[2] == "drug"
    ]
    ppi_c_etypes = [
        c_etype for c_etype in c_etypes if c_etype[0] == c_etype[2] == "protein"
    ]

    assert len(c_etypes) == (
        len(ddi_c_etypes) + len(dpi_c_etypes) + len(pdi_c_etypes) + len(ppi_c_etypes)
    )
    assert len(ddi_c_etypes) == 963
    assert all(re.match(r"ddi_C\d{7}", c_etype[1]) for c_etype in ddi_c_etypes)
    assert dpi_c_etypes == [("drug", "dpi", "protein")]
    assert pdi_c_etypes == [("protein", "pdi", "drug")]
    assert ppi_c_etypes == [("protein", "ppi", "protein")]


def test_no_split_leakage(graphs: GraphsResult) -> None:
    c_etypes: list[tuple[str, str, str]] = graphs.train_pos_g.canonical_etypes  # type: ignore
    for g in [graphs.train_pos_g, graphs.val_pos_g, graphs.test_pos_g]:
        for c_etype in c_etypes:
            if c_etype[0] != c_etype[2]:
                continue

            shape = g.adj(etype=c_etype).shape
            row, col = g.adj(etype=c_etype).indices()
            adj = sps.coo_array((torch.ones_like(row), (row, col)), shape)

            assert (adj != adj.transpose()).nnz == 0

    for g in [graphs.train_pos_g, graphs.val_pos_g, graphs.test_pos_g]:
        dpi_shape = g.adj(etype="dpi").shape
        dpi_row, dpi_col = g.adj(etype="dpi").indices()
        dpi_adj = sps.coo_array(
            (torch.ones_like(dpi_row), (dpi_row, dpi_col)),
            dpi_shape,
        )

        pdi_shape = g.adj(etype="pdi").shape
        pdi_row, pdi_col = g.adj(etype="pdi").indices()
        pdi_adj = sps.coo_array(
            (torch.ones_like(pdi_row), (pdi_row, pdi_col)),
            pdi_shape,
        )

        assert (dpi_adj != pdi_adj.transpose()).nnz == 0


def test_negative_samples(graphs: GraphsResult) -> None:
    c_etypes: list[tuple[str, str, str]] = graphs.train_pos_g.canonical_etypes  # type: ignore
    for c_etype in c_etypes:
        train_pos_src = graphs.train_pos_g.adj(c_etype).indices()[0]
        train_neg_src = graphs.train_neg_g.adj(c_etype).indices()[0]
        val_pos_src = graphs.val_pos_g.adj(c_etype).indices()[0]
        val_neg_src = graphs.val_neg_g.adj(c_etype).indices()[0]
        test_pos_src = graphs.test_pos_g.adj(c_etype).indices()[0]
        test_neg_src = graphs.test_neg_g.adj(c_etype).indices()[0]

        assert torch.equal(train_pos_src, train_neg_src)
        assert torch.equal(val_pos_src, val_neg_src)
        assert torch.equal(test_pos_src, test_neg_src)


def test_no_self_loops(graphs: GraphsResult) -> None:
    c_etypes: list[tuple[str, str, str]] = graphs.train_pos_g.canonical_etypes  # type: ignore
    for g in [graphs.train_pos_g, graphs.val_pos_g, graphs.test_pos_g]:
        for c_etype in c_etypes:
            if c_etype[0] != c_etype[2]:
                continue

            row, col = g.adj(etype=c_etype).indices()
            assert (row == col).sum().item() == 0
