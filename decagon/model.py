# NOTE: The initializations are kaiming, not xavier (which is used in the paper)
from typing import Literal

import torch
from dgl import DGLGraph
from dgl import function as fn
from dgl.udf import EdgeBatch
from torch import Tensor, nn
from torch.nn import functional as F


class HeteroRGCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout_p: float,
        ntypes: list[str],
        etypes: list[str],
    ) -> None:
        super().__init__()

        self.etype2w = nn.ModuleDict(
            {
                etype: nn.Sequential(
                    nn.Dropout(dropout_p),
                    nn.Linear(in_dim, out_dim, bias=True),
                )
                for etype in etypes
            }
        )
        self.ntype2selfloop_w = nn.ModuleDict(
            {
                ntype: nn.Sequential(
                    nn.Dropout(dropout_p),
                    nn.Linear(in_dim, out_dim, bias=True),
                )
                for ntype in ntypes
            }
        )

    def forward(self, g: DGLGraph, ntype2h: dict[str, Tensor]) -> dict[str, Tensor]:
        with g.local_scope():
            etype2funcs: dict[str, tuple[fn.BuiltinFunction, fn.BuiltinFunction]] = {}
            for c_etype in g.canonical_etypes:  # type: ignore
                srctype: str = c_etype[0]
                etype: str = c_etype[1]

                h = ntype2h[srctype]  # [N_srctype in_dim]
                h_etype: Tensor = self.etype2w[etype].forward(h)  # [N_srctype out_dim]
                g.nodes[srctype].data[f"h_{etype}"] = h_etype

                etype2funcs[etype] = (
                    fn.copy_u(f"h_{etype}", "m"),
                    fn.mean("m", "h_neigh"),  # type: ignore
                )

            # Message passing
            g.multi_update_all(etype2funcs, "mean")

            # Self-loop
            for ntype, h in ntype2h.items():
                selfloop_w = self.ntype2selfloop_w[ntype]

                h_neigh: Tensor = g.nodes[ntype].data["h_neigh"]  # [N_ntype out_dim]
                h_selfloop: Tensor = selfloop_w.forward(h)  # [N_ntype out_dim]
                ntype2h[ntype] = h_neigh + h_selfloop  # [N_ntype out_dim]

            return ntype2h


class DecagonEncoder(nn.Module):
    def __init__(
        self,
        g: DGLGraph,
        dims: list[int],
        dropout_ps: list[float],
    ) -> None:
        assert len(dims) == len(dropout_ps)

        super().__init__()

        self.in_dim, self.out_dim = dims[0], dims[-1]
        self.ntype2identity_w = nn.ModuleDict(
            {
                ntype: nn.Embedding(
                    g.nodes[ntype].data["identity"].size(0),
                    self.in_dim,
                )
                for ntype in g.ntypes
                if "identity" in g.nodes[ntype].data
            }
        )
        self.ntype2feat_w = nn.ModuleDict(
            {
                ntype: nn.Sequential(
                    nn.Dropout(dropout_ps[0]),
                    nn.Linear(
                        g.nodes[ntype].data["feat"].size(1),
                        self.in_dim,
                    ),
                )
                for ntype in g.ntypes
                if "feat" in g.nodes[ntype].data
            }
        )
        self.convs = nn.ModuleList(
            HeteroRGCN(
                in_dim=in_dim,
                out_dim=out_dim,
                dropout_p=dropout_p,
                ntypes=g.ntypes,
                etypes=g.etypes,
            )
            for in_dim, out_dim, dropout_p in zip(dims[:-1], dims[1:], dropout_ps[1:])
        )

    def forward(self, g: DGLGraph) -> dict[str, Tensor]:
        ntype2h: dict[str, Tensor] = {}
        for ntype in g.ntypes:
            ndata: dict[str, Tensor] = g.nodes[ntype].data
            num_nodes: int = g.num_nodes(ntype)

            assert "feat" in ndata or "identity" in ndata
            assert ntype in self.ntype2identity_w or ntype in self.ntype2feat_w

            ntype2h[ntype] = torch.zeros(num_nodes, self.in_dim, device=g.device)
            if "identity" in ndata:
                identity_w = self.ntype2identity_w[ntype]
                identity: Tensor = ndata["identity"]  # [N_ntype 1]
                identity = identity_w.forward(identity)  # [N_ntype in_dim]
                ntype2h[ntype] += identity
            if "feat" in ndata:
                feat_w = self.ntype2feat_w[ntype]
                feat: Tensor = ndata["feat"]  # [N_ntype F_ntype]
                feat = feat_w.forward(feat)  # [N_ntype in_dim]
                ntype2h[ntype] += feat

        for conv in self.convs:
            ntype2h = conv.forward(g, ntype2h)
            ntype2h = {ntype: F.relu(h) for ntype, h in ntype2h.items()}

        return ntype2h


class DecagonDecoder(nn.Module):
    def __init__(
        self,
        drug_in_dim: int,
        protein_in_dim: int,
        drug_ntype: str,
        protein_ntype: str,
        ppi_etype: str,
        dpi_etype: str,
        pdi_etype: str,
        ddi_etypes: list[str],
    ) -> None:
        super().__init__()

        self.drug_ntype = drug_ntype
        self.protein_ntype = protein_ntype
        self.ppi_etype = ppi_etype
        self.dpi_etype = dpi_etype
        self.pdi_etype = pdi_etype

        self.ppi_weight = nn.Linear(
            protein_in_dim,
            protein_in_dim,
            bias=False,
        )
        self.ddi_weight = nn.Linear(
            drug_in_dim,
            drug_in_dim,
            bias=False,
        )
        self.dpi_weight = nn.Linear(
            drug_in_dim,
            protein_in_dim,
            bias=False,
        )

        ddi_etype2cse_w: dict[str, Tensor] = {}
        for ddi_etype in ddi_etypes:
            cse_w = torch.empty(1, drug_in_dim)
            nn.init.kaiming_uniform_(cse_w)
            ddi_etype2cse_w[ddi_etype] = nn.Parameter(cse_w)
        self.ddi_etype2cse_w = nn.ParameterDict(ddi_etype2cse_w)

    def apply_dpi_edges(self, edges: EdgeBatch) -> dict[Literal["score"], Tensor]:
        src_feat: Tensor = edges.src["h"]  # [E_dpi F_drug]
        dst_feat: Tensor = edges.dst["h"]  # [E_dpi F_protein]

        src_feat = self.dpi_weight.forward(src_feat)  # [E_dpi F_protein]
        score = torch.einsum("ij,ij->i", src_feat, dst_feat)  # [E_dpi]

        return {"score": score}

    def apply_pdi_edges(self, edges: EdgeBatch) -> dict[Literal["score"], Tensor]:
        src_feat: Tensor = edges.src["h"]  # [E_pdi F_protein]
        dst_feat: Tensor = edges.dst["h"]  # [E_pdi F_drug]

        dst_feat = self.dpi_weight.forward(dst_feat)  # [E_pdi F_protein]
        score = torch.einsum("ij,ij->i", src_feat, dst_feat)  # [E_pdi]

        return {"score": score}

    def apply_ppi_edges(self, edges: EdgeBatch) -> dict[Literal["score"], Tensor]:
        src_feat: Tensor = edges.src["h"]  # [E_ppi F_protein]
        dst_feat: Tensor = edges.dst["h"]  # [E_ppi F_protein]

        src_feat = self.ppi_weight.forward(src_feat)  # [E_ppi F_protein]
        score = torch.einsum("ij,ij->i", src_feat, dst_feat)  # [E_ppi]

        return {"score": score}

    def apply_ddi_edges(self, edges: EdgeBatch) -> dict[Literal["score"], Tensor]:
        src_feat: Tensor = edges.src["h"]  # [E_ddi F_drug]
        dst_feat: Tensor = edges.dst["h"]  # [E_ddi F_drug]

        src_feat = self.ddi_weight.forward(src_feat)  # [E_ddi F_drug]
        score = torch.einsum("ij,ij->i", src_feat, dst_feat)  # [E_ddi]

        return {"score": score}

    def forward(
        self,
        g: DGLGraph,
        ntype2h: dict[str, Tensor],
    ) -> dict[tuple[str, str, str], Tensor]:
        with g.local_scope():
            drug_ntype = self.drug_ntype
            protein_ntype = self.protein_ntype
            drug_h = ntype2h[drug_ntype]  # [N_drug F_drug]
            protein_h = ntype2h[protein_ntype]  # [N_protein F_protein]

            g.nodes[drug_ntype].data["h"] = drug_h
            g.nodes[protein_ntype].data["h"] = protein_h

            g.apply_edges(self.apply_ppi_edges, etype=self.ppi_etype)
            g.apply_edges(self.apply_dpi_edges, etype=self.dpi_etype)
            g.apply_edges(self.apply_pdi_edges, etype=self.pdi_etype)
            for ddi_etype, cse_w in self.ddi_etype2cse_w.items():
                g.nodes[drug_ntype].data["h"] = drug_h.mul(cse_w)  # [N_drug F_drug]
                g.apply_edges(self.apply_ddi_edges, etype=ddi_etype)

            c_etype2score: dict[tuple[str, str, str], Tensor] = g.edata["score"]

            return c_etype2score


class DecagonLinkPredictor(nn.Module):
    def __init__(
        self,
        g: DGLGraph,
        dims: list[int],
        dropout_ps: list[float],
    ) -> None:
        assert len(dims) == len(dropout_ps)

        super().__init__()

        self.encoder = DecagonEncoder(g=g, dims=dims, dropout_ps=dropout_ps)

        self.decoder = DecagonDecoder(
            drug_in_dim=dims[-1],
            protein_in_dim=dims[-1],
            drug_ntype="drug",
            protein_ntype="protein",
            ppi_etype="ppi",
            dpi_etype="dpi",
            pdi_etype="pdi",
            ddi_etypes=[
                c_etype[1]
                for c_etype in g.canonical_etypes  # type: ignore
                if c_etype[0] == c_etype[2] == "drug"
            ],
        )

    def forward(
        self,
        pos_g: DGLGraph,
        neg_g: DGLGraph,
    ) -> tuple[dict[tuple[str, str, str], Tensor], dict[tuple[str, str, str], Tensor]]:
        ntype2h = self.encoder.forward(g=pos_g)

        c_etype2pos_score = self.decoder.forward(g=pos_g, ntype2h=ntype2h)
        c_etype2neg_score = self.decoder.forward(g=neg_g, ntype2h=ntype2h)

        return c_etype2pos_score, c_etype2neg_score
