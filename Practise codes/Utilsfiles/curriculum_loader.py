import os
import jax.numpy as jnp
from utils.data_utils_stable import load_and_normalize_stagewise_data_stable

# Ordered list of stage tags from first to last
ALL_STAGES_ORDERED = [
    "1.0", "1.0_1.2", "1.2_1.4", "1.4_1.6", "1.6_1.8",
    "1.8_2.0", "2.0_2.2", "2.2_2.4", "2.4_2.6", "2.6_2.8", "2.8_3.0"
]

def get_all_previous_stages_upto(stage_tag):
    idx = ALL_STAGES_ORDERED.index(stage_tag)
    return ALL_STAGES_ORDERED[:idx+1]

def get_previous_stage_tag(stage_tag):
    idx = ALL_STAGES_ORDERED.index(stage_tag)
    if idx == 0:  # first stage
        return None
    return ALL_STAGES_ORDERED[idx-1]

def load_curriculum_with_previous_weights(cfg):
    """
    Loads cumulative or stage-only dataset and automatically finds
    previous stage checkpoint for transfer learning.
    """
    stage_tag = str(cfg.stage_tag)

    if cfg.curriculum.cumulative_mode:
        stages_list = get_all_previous_stages_upto(stage_tag)
    else:
        stages_list = [stage_tag]

    # Load datasets for all chosen stages
    results = load_and_normalize_stagewise_data_stable(
        cfg.model_type, "datafiles", stages_list,
        seed=cfg.seed, scaling_mode=cfg.data.scaling_mode
    )

    if cfg.curriculum.cumulative_mode:
        # Merge all datasets into single arrays
        Xtr_all, Xv_all, Xt_all = [], [], []
        Ytr_all, Yv_all, Yt_all = [], [], []
        first_stage = stages_list[0]
        X_mean, X_std, Y_mean, Y_std = results[first_stage][6:10]  # keep first stage stats

        for st in stages_list:
            Xtr, Xv, Xt, Ytr, Yv, Yt, _, _, _, _ = results[st]
            Xtr_all.append(Xtr); Xv_all.append(Xv); Xt_all.append(Xt)
            Ytr_all.append(Ytr); Yv_all.append(Yv); Yt_all.append(Yt)

        Xtr = jnp.concatenate(Xtr_all)
        Xv  = jnp.concatenate(Xv_all)
        Xt  = jnp.concatenate(Xt_all)
        Ytr = jnp.concatenate(Ytr_all)
        Yv  = jnp.concatenate(Yv_all)
        Yt  = jnp.concatenate(Yt_all)
    else:
        # Stage-only
        Xtr, Xv, Xt, Ytr, Yv, Yt, X_mean, X_std, Y_mean, Y_std = results[stage_tag]

    # Automatic previous stage checkpoint
    prev_stage_tag = get_previous_stage_tag(stage_tag)
    if prev_stage_tag:
        transfer_ckpt_path = os.path.join(
            cfg.output_dir, f"{cfg.model_type}_stage_{prev_stage_tag}_lambda_{cfg.training.lambda_phys[0]}",
            "trained_params.msgpack"
        )
    else:
        transfer_ckpt_path = None  # first stage → no previous weights

    return Xtr, Xv, Xt, Ytr, Yv, Yt, X_mean, X_std, Y_mean, Y_std, transfer_ckpt_path