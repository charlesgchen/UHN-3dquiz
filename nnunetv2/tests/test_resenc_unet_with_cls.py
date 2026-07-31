"""
Shape and behaviour tests for the multi-task network.

The 'real plans' tests build the network from the actual generated nnUNetResEncUNetMPlans for
Dataset001_PancreasQuiz and are skipped when that dataset has not been preprocessed yet.
"""

import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.resenc_unet_with_cls import (
    ClassificationHead,
    CrossAttentionPooling,
    ResEncUNetWithClassification,
)
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

PLANS_PATH = os.path.join(os.environ.get('nnUNet_preprocessed', ''),
                          'Dataset001_PancreasQuiz', 'nnUNetResEncUNetMPlans.json')
NUM_SUBTYPES = 3


def _load_plans_config():
    import json
    with open(PLANS_PATH) as f:
        return json.load(f)['configurations']['3d_fullres']


def _build(deep_supervision=True, **kwargs):
    cfg = _load_plans_config()
    arch = cfg['architecture']
    backbone = get_network_from_plans(
        arch['network_class_name'], arch['arch_kwargs'], arch['_kw_requires_import'],
        input_channels=1, output_channels=3, allow_init=True, deep_supervision=deep_supervision)
    return ResEncUNetWithClassification(backbone, num_subtypes=NUM_SUBTYPES, **kwargs), cfg


needs_plans = pytest.mark.skipif(not os.path.isfile(PLANS_PATH),
                                 reason=f'plans not found at {PLANS_PATH}; run plan_and_preprocess first')


# ----------------------------------------------------------------------------- pooling module

def test_cross_attention_pooling_output_shape():
    pool = CrossAttentionPooling(embed_dim=32, query_num=4, num_classes=3, num_heads=4)
    out = pool(torch.randn(2, 32, 4, 4, 6))
    assert out.shape == (2, 3)


def test_cross_attention_pooling_accepts_flattened_input():
    pool = CrossAttentionPooling(embed_dim=32, query_num=4, num_classes=3, num_heads=4)
    assert pool(torch.randn(2, 32, 96)).shape == (2, 3)


def test_cross_attention_pooling_is_permutation_invariant():
    """No positional encoding, so shuffling the spatial tokens must not change the logits."""
    pool = CrossAttentionPooling(embed_dim=16, query_num=2, num_classes=3, num_heads=4).eval()
    x = torch.randn(1, 16, 2, 2, 4)
    flat = x.flatten(2)
    shuffled = flat[:, :, torch.randperm(flat.shape[2])]
    with torch.no_grad():
        assert torch.allclose(pool(flat), pool(shuffled), atol=1e-5)


def test_cross_attention_pooling_rejects_indivisible_head_count():
    with pytest.raises(ValueError, match='divisible'):
        CrossAttentionPooling(embed_dim=30, query_num=2, num_classes=3, num_heads=4)


def test_classifier_input_width_is_query_num_times_embed_dim():
    """Queries are concatenated, not averaged, so each one contributes its own features."""
    pool = CrossAttentionPooling(embed_dim=32, query_num=4, num_classes=3)
    assert pool.classifier.in_features == 4 * 32


def test_gap_head_shape():
    head = ClassificationHead(embed_dim=32, num_classes=3, use_cross_attention=False)
    assert head(torch.randn(2, 32, 4, 4, 6)).shape == (2, 3)


def test_pooling_handles_variable_spatial_size():
    """Sliding-window inference can hand the head differently sized bottlenecks."""
    pool = CrossAttentionPooling(embed_dim=16, query_num=2, num_classes=3, num_heads=4).eval()
    with torch.no_grad():
        assert pool(torch.randn(1, 16, 2, 2, 2)).shape == (1, 3)
        assert pool(torch.randn(1, 16, 4, 5, 6)).shape == (1, 3)


# ----------------------------------------------------------------------------- full network

@needs_plans
def test_forward_shapes_with_deep_supervision():
    net, cfg = _build(deep_supervision=True)
    net.eval()
    patch = cfg['patch_size']
    with torch.no_grad():
        seg, logits = net(torch.randn(1, 1, *patch))
    assert isinstance(seg, (list, tuple)), 'deep supervision should yield a list of seg outputs'
    assert seg[0].shape == (1, 3, *patch)
    assert logits.shape == (1, NUM_SUBTYPES)


@needs_plans
def test_forward_shapes_without_deep_supervision():
    net, cfg = _build(deep_supervision=False)
    net.eval()
    patch = cfg['patch_size']
    with torch.no_grad():
        seg, logits = net(torch.randn(1, 1, *patch))
    assert torch.is_tensor(seg)
    assert seg.shape == (1, 3, *patch)
    assert logits.shape == (1, NUM_SUBTYPES)


@needs_plans
def test_bottleneck_token_count_matches_plans():
    """Documents the sequence length the attention actually runs over."""
    net, cfg = _build()
    net.eval()
    strides = np.array(cfg['architecture']['arch_kwargs']['strides'])
    expected = np.array(cfg['patch_size']) // strides.prod(axis=0)
    with torch.no_grad():
        skips = net.backbone.encoder(torch.randn(1, 1, *cfg['patch_size']))
    assert list(skips[-1].shape[2:]) == list(expected)
    assert skips[-1].shape[1] == cfg['architecture']['arch_kwargs']['features_per_stage'][-1]


@needs_plans
def test_classification_head_is_small_relative_to_backbone():
    net, _ = _build()
    head = sum(p.numel() for p in net.classification_head.parameters())
    backbone = sum(p.numel() for p in net.backbone.parameters())
    assert head / backbone < 0.05, f'head is {head/backbone:.1%} of the backbone, expected <5%'


@needs_plans
def test_gradients_reach_encoder_from_classification_loss_only():
    """The whole point of the shared encoder: classification loss must train encoder weights."""
    net, cfg = _build(deep_supervision=False)
    small = [s // 2 for s in cfg['patch_size']]
    _, logits = net(torch.randn(1, 1, *small))
    nn.functional.cross_entropy(logits, torch.tensor([1])).backward()

    stem_grad = next(p.grad for p in net.backbone.encoder.parameters() if p.grad is not None)
    assert stem_grad.abs().sum() > 0

    # The decoder's own layers must NOT have received gradient from the classification loss.
    # NOTE: UNetDecoder keeps a reference to the encoder (unet_decoder.py: self.encoder = encoder),
    # so decoder.parameters() *includes* every encoder parameter. Iterate the decoder's own
    # submodules instead. The same aliasing bites anyone splitting optimizer param groups by
    # encoder/decoder, so it is asserted here rather than just commented.
    decoder_own = [p for module in (net.decoder.stages, net.decoder.transpconvs, net.decoder.seg_layers)
                   for p in module.parameters()]
    assert decoder_own, 'expected to find decoder-owned parameters'
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in decoder_own)


@needs_plans
def test_decoder_parameters_alias_the_encoder():
    """Guards the aliasing noted above, so a future refactor that splits param groups notices."""
    net, _ = _build()
    decoder_ids = {id(p) for p in net.decoder.parameters()}
    encoder_ids = {id(p) for p in net.encoder.parameters()}
    assert encoder_ids <= decoder_ids, 'UNetDecoder no longer aliases the encoder; revisit param grouping'


@needs_plans
def test_forward_classification_only_matches_full_forward():
    net, cfg = _build(deep_supervision=False)
    net.eval()
    x = torch.randn(1, 1, *[s // 2 for s in cfg['patch_size']])
    with torch.no_grad():
        _, logits_full = net(x)
        logits_fast = net.forward_classification_only(x)
    assert torch.allclose(logits_full, logits_fast, atol=1e-5)


@needs_plans
def test_deep_supervision_toggle_is_visible_through_wrapper():
    """nnU-Net flips network.decoder.deep_supervision during validation; the wrapper must expose it."""
    net, cfg = _build(deep_supervision=True)
    net.eval()
    net.decoder.deep_supervision = False
    with torch.no_grad():
        seg, _ = net(torch.randn(1, 1, *[s // 2 for s in cfg['patch_size']]))
    assert torch.is_tensor(seg), 'toggling .decoder.deep_supervision should change the seg output type'


def test_wrapper_rejects_backbone_without_encoder():
    class NoEncoder(nn.Module):
        pass

    with pytest.raises(ValueError, match='encoder'):
        ResEncUNetWithClassification(NoEncoder(), num_subtypes=3)
