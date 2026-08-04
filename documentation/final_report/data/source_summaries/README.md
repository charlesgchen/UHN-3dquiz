# Source metric snapshots

These files are content-equivalent snapshots of the small `summary.json` outputs used to build the
normalized report tables. JSON values are unchanged; the snapshots have a conventional final
newline. They are stored here so the numerical evidence is retained with the documentation even if
large, untracked prediction directories are archived.

| Snapshot | Original repository path |
|---|---|
| `accepted_heldout.json` | `predictions/original_tta_frozen_encoder_mlp16_equal_geometric_heldout/summary.json` |
| `cross_attention_oof_no_tta.json` | `predictions/original_subtype_head_5fold_oof_best_cls_no_tta/summary.json` |
| `cross_attention_oof_8x_tta.json` | `predictions/original_subtype_head_5fold_oof_best_cls_tta/summary.json` |
| `cross_attention_heldout_no_tta.json` | `predictions/reset_head_5fold_val_case_embeddings_no_tta/summary.json` |
| `cross_attention_heldout_8x_tta.json` | `predictions/head_adamw_5fold_val_best_cls/summary.json` |
| `encoder_meanmax_mlp_heldout.json` | `predictions/frozen_embedding_mlp16_heldout/summary.json` |
| `gap_oof_no_tta.json` | `predictions/matched_gap_head_5fold_oof_best_cls_no_tta/summary.json` |
| `gap_oof_8x_tta.json` | `predictions/matched_gap_head_5fold_oof_best_cls_tta/summary.json` |
| `gap_heldout_no_tta.json` | `predictions/matched_gap_head_5fold_heldout_best_cls_no_tta/summary.json` |
| `gap_heldout_8x_tta.json` | `predictions/matched_gap_head_5fold_heldout_best_cls_tta/summary.json` |
| `continued_mild_da_oof.json` | `predictions/ce_head_only_mild_da_5fold_oof_best_cls_no_tta/summary.json` |
| `fresh_mild_da_oof.json` | `predictions/ce_fresh_head_5fold_oof_best_cls_no_tta/summary.json` |
| `feature_augmented_mlp_oof.json` | `predictions/paired_encoder_mlp16_meanmax_aug_5fold_oof/summary.json` |
| `feature_augmented_mlp_heldout.json` | `predictions/paired_encoder_mlp16_meanmax_aug_heldout_no_tta/summary.json` |

These snapshots contain metrics and confusion matrices, not images, patient metadata, weights, or
prediction volumes.
