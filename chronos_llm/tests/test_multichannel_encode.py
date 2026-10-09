"""CPU unit tests for multichannel soft-prompt encoding (real chronos-2 + tiny LLM + synthetic multichannel context)."""
import torch

from chronos_llm.tests.test_pretrained_peft import _build_tiny_base


def test_group_attention_crosstalk():
    """Channels of the same group interact across channels inside chronos GroupSelfAttention:
    all-equal group_ids (one group of 3 channels) vs all-different (3 independent) -> patch encodings must differ."""
    model, _ = _build_tiny_base()
    torch.manual_seed(0)
    x = torch.randn(3, 64)  # 3 channels, 64 steps (one chunk)
    # encode returns (encoder_outputs, loc_scale, pfm, ncp); [0][0] = last_hidden_state
    grp = model.chronos.encode(context=x, num_output_patches=1,
                               group_ids=torch.zeros(3, dtype=torch.long))[0][0]
    ind = model.chronos.encode(context=x, num_output_patches=1,
                               group_ids=torch.arange(3))[0][0]
    assert not torch.allclose(grp, ind, atol=1e-4), "group_ids does not affect the output -> cross-channel interaction is not active"
    print("group attention cross-channel interaction OK")


def test_token_budget_grows_with_channels():
    """More channels C => more soft-prompt tokens (log-linear budget proportional to log2(P*C)).

    P must be large enough that the budget (rather than P itself) is the binding constraint: at
    tiny P, nw is capped by P (one window per patch) and the token count degenerates to G + P*k,
    independent of C -- that is the saturated region of the soft lower bound, outside this test's scope.
    """
    model, _ = _build_tiny_base()
    L = 3200  # P=ceil(3200/16)=200, far above nw_target for C=1, so P is not the cap
    ctx_mc = torch.randn(3, L)                       # 1 sample, 3 channels
    soft_mc = model._encode_history_to_soft_prompt(
        ctx_mc, torch.tensor([L, L, L]), torch.tensor([3]))
    ctx_sc = torch.randn(1, L)                       # 1 sample, 1 channel
    soft_sc = model._encode_history_to_soft_prompt(
        ctx_sc, torch.tensor([L]), torch.tensor([1]))
    assert len(soft_mc) == 1 and len(soft_sc) == 1
    n_mc, n_sc = soft_mc[0].shape[0], soft_sc[0].shape[0]
    assert n_mc > n_sc, f"C=3 token count {n_mc} does not exceed C=1's {n_sc} (C did not enter the budget)"
    assert torch.isfinite(soft_mc[0]).all()
    print(f"token count grows with the number of channels OK (C=1 -> {n_sc} tokens, C=3 -> {n_mc} tokens)")


def test_multi_sample_mixed_channels():
    """Mixed: sample 0 has C=3 length 50, sample 1 has C=1 length 30, folded into (sum_C=4, L), aggregated per sample."""
    model, _ = _build_tiny_base()
    L = 50
    ctx = torch.full((4, L), float("nan"))
    for r in range(3):
        ctx[r] = torch.randn(L)            # sample 0: three channels (full length)
    ctx[3, L - 30:] = torch.randn(30)      # sample 1: single channel (length 30, left NaN pad)
    soft = model._encode_history_to_soft_prompt(
        ctx, torch.tensor([50, 50, 50, 30]), torch.tensor([3, 1]))
    assert len(soft) == 2
    assert all(torch.isfinite(s).all() for s in soft)
    print(f"multi-sample mixed channel counts OK (soft lens={[s.shape[0] for s in soft]})")


def test_single_channel_default_none():
    """n_channels=None: every row is treated as a single-channel sample (backward compatible)."""
    model, _ = _build_tiny_base()
    ctx = torch.randn(2, 48)
    soft = model._encode_history_to_soft_prompt(ctx, torch.tensor([48, 48]), None)
    assert len(soft) == 2 and all(torch.isfinite(s).all() for s in soft)
    print("n_channels=None single-channel compatibility OK")


if __name__ == "__main__":
    test_group_attention_crosstalk()
    test_token_budget_grows_with_channels()
    test_multi_sample_mixed_channels()
    test_single_channel_default_none()
    print("ALL OK")
