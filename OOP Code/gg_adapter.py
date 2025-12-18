# gg_adapter.py
import gbfs_globals as GG

def sync_context_to_GG(ctx, cfg=None, kshell_seq_mode=None):
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

    GG.C_matrix = getattr(ctx, "C_matrix", None)

    if cfg is not None:
        GG.kshell_max_add = int(cfg.kshell_max_add)
        GG.rc_tau = float(cfg.kshell_rc_tau)

    if kshell_seq_mode is not None:
        GG.kshell_seq_mode = kshell_seq_mode
