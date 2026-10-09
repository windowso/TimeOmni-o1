"""CPU unit test for the 2-D sliding-window Q-former (multi-channel): each window contains all channels + channel-id / intra-window time embeddings."""
import torch

from chronos_llm.models.qformer import SlidingWindowQFormer


def _swq():
    return SlidingWindowQFormer(in_dim=32, out_dim=64, queries_per_window=4,
                                target_windows=8, min_windows=2, num_heads=4, num_layers=1,
                                max_channels=16, intra_window_pos=True)


def test_token_count_grows_with_channels():
    swq = _swq()
    P = 512  # large enough that a growing C shows up in the window count
    cs = []
    for C in (1, 4, 8, 16):  # must not exceed max_channels=16 of _swq
        out = swq(torch.randn(1, C, P, 32))
        n = swq.num_tokens_for_length(P, C)
        assert out.shape == (1, n, 64), (C, out.shape)
        assert torch.isfinite(out).all()
        cs.append(n)
    assert cs == sorted(cs), f"token count must be non-decreasing in the channel count: {cs}"
    assert cs[-1] > cs[0], f"more channels did not increase the token count: {cs}"
    print("token count grows with channel count OK")


def test_backward_compat_single_channel():
    """(B,P,d) and (B,1,P,d) go through the same forward and are equivalent (C=1 degenerate entry point)."""
    torch.manual_seed(0)
    swq = _swq().eval()
    src = torch.randn(1, 20, 32)
    o2d = swq(src)
    o3d = swq(src.unsqueeze(1))
    assert torch.allclose(o2d, o3d, atol=1e-6), (o2d - o3d).abs().max()
    print("C=1 compatibility entry point OK")


def test_channel_embedding_distinguishes():
    """Same patch content in a different channel order must give a different output (channel-id embedding effective)."""
    torch.manual_seed(0)
    swq = _swq().eval()
    x = torch.randn(1, 3, 16, 32)
    a = swq(x)
    b = swq(x.flip(dims=[1]))
    assert not torch.allclose(a, b, atol=1e-5), "channel order does not affect the output -> channel id not effective"
    print("channel-id embedding effective OK")


def test_long_window_no_index_error():
    """With a large P (hence a large window w) the intra-window time PE index must not go out of range."""
    swq = _swq()
    out = swq(torch.randn(1, 2, 5000, 32))
    assert torch.isfinite(out).all()
    print("large window intra-window time PE in range OK")


if __name__ == "__main__":
    test_token_count_grows_with_channels()
    test_backward_compat_single_channel()
    test_channel_embedding_distinguishes()
    test_long_window_no_index_error()
    print("ALL OK")
