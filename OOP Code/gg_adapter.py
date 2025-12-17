# gg_adapter.py
import gbfs_globals as GG

def sync_context_to_GG(ctx) -> None:
    # tối thiểu newtry_ms + các hàm phía dưới đang cần
    GG.data = ctx.zData
    GG.label = ctx.label
    GG.featNum = ctx.zData.shape[1]

    GG.trIdx = ctx.trIdx
    GG.trData = ctx.trData
    GG.trLabel = ctx.trLabel
    GG.teData = ctx.teData
    GG.teLabel = ctx.teLabel

    GG.Zout = ctx.Zout
    GG.Weight = ctx.Weight
    GG.vWeight = ctx.vWeight
    GG.vWeight1 = ctx.vWeight1

    GG.neigh_list = ctx.neigh_list
    GG.kNeiZout = ctx.A_init
