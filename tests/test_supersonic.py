import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@pytest.fixture(scope="module", params=range(torch.cuda.device_count()) if torch and torch.cuda.is_available() else [0])
def device(request):
    if torch is None or not torch.cuda.is_available():
        pytest.skip("CUDA PyTorch required")
    if torch.cuda.get_device_capability(request.param) != (8, 9):
        pytest.skip("SM89 / L4 required")
    return torch.device("cuda", request.param)


@pytest.fixture(scope="module")
def kernels(device):
    import xcalibur
    return xcalibur


def reference(logits, K, softmax):
    x = logits.float()
    if softmax:
        a = (x - x.amax(-1, keepdim=True)).exp().bfloat16().float()
    else:
        a = ((-x).exp().bfloat16().float() + 1).bfloat16().float()
        a = a.reciprocal().bfloat16().float()
    ids = a.argsort(dim=-1, descending=True, stable=True)[:, :K]
    weights = a.gather(1, ids)
    if softmax:
        weights = weights / a.sum(-1, keepdim=True)
    return ids, weights.bfloat16()


def unpack_routes(routes):
    return 65535 - (routes & 65535), (routes >> 16).to(torch.int16).view(torch.bfloat16)


def swiglu(g, u, w):
    g, u = g.bfloat16(), u.bfloat16()
    e = (-g.abs().float()).exp().bfloat16()
    s = (e + 1).reciprocal()
    s = torch.where(g < 0, e * s, s)
    return ((g * s) * u) * w.bfloat16()


@pytest.mark.parametrize("softmax", [True, False])
@pytest.mark.parametrize("N,E,K", [(1, 1, 1), (7, 31, 5), (9, 257, 16), (32, 128, 8), (17, 512, 16)])
def test_topk(kernels, device, softmax, N, E, K):
    torch.manual_seed(7)
    logits = (torch.randint(-32, 33, (N, E)).float() / 4).bfloat16()
    logits[0] = 0
    if N > 1:
        logits[1] = 8
        logits[1, -1] = 16
        logits[-1] -= 100
    ids, weights = reference(logits, K, softmax)
    got_ids, got_weights = unpack_routes(kernels.topk(logits.to(device), K, softmax).cpu())
    torch.testing.assert_close(got_ids.long(), ids, rtol=0, atol=0)
    torch.testing.assert_close(got_weights.float(), weights.float(), rtol=0.008, atol=2e-5)


@pytest.mark.parametrize("N,E,K,H,I", [
    (1, 4, 1, 8, 8), (8, 4, 4, 1536, 192), (9, 4, 3, 136, 17),
    (17, 8, 2, 2048, 193), (64, 32, 4, 2048, 256), (769, 4, 2, 64, 8),
])
def test_xr38f1(kernels, device, N, E, K, H, I):
    torch.manual_seed(38)
    X = torch.randn(N, H).bfloat16()
    gate, up = [(torch.randn(E, I, H) / H**0.5).bfloat16() for _ in range(2)]
    logits = (torch.randint(-16, 17, (N, E)).float() / 4).bfloat16()
    logits[:, -1] = -80
    ids, weights = reference(logits, K, True)
    with torch.cuda.stream(torch.cuda.Stream(device=device)):
        routes = kernels.topk(logits.to(device), K)
        Y = kernels.xR38F1(kernels.pack_w13(gate, up).to(device), X.to(device), routes).cpu()
    Ib, S = (I + 7) // 8, 8 + 32 * ((I + 7) // 8)
    assert not Y[:, 1:8].count_nonzero()
    for e in range(E):
        tokens, rank = (ids == e).nonzero(as_tuple=True)
        count = (len(tokens) + 7) // 8
        assert Y[e, 0].item() == count
        if not count:
            continue
        records = Y[e, 8:8 + count * S].reshape(count, S)
        expected_ids = torch.full((count * 8,), -1, dtype=torch.int32)
        expected_ids[:len(tokens)] = tokens.int()
        torch.testing.assert_close(records[:, :8], expected_ids.reshape(count, 8), rtol=0, atol=0)
        got = records[:, 8:].contiguous().view(torch.bfloat16)
        got = got.reshape(count, Ib, 8, 8).permute(0, 2, 1, 3).reshape(count * 8, Ib * 8)
        g, u = [X[tokens].float() @ w[e].float().T for w in (gate, up)]
        want = torch.zeros(count * 8, Ib * 8)
        want[:len(tokens), :I] = swiglu(g, u, weights[tokens, rank, None])
        torch.testing.assert_close(got.float(), want.bfloat16().float(), rtol=0.02, atol=2e-4)
        assert not got[len(tokens):].count_nonzero() and not got[:, I:].count_nonzero()


@pytest.mark.parametrize("K", [0, 5, 17])
def test_bad_k(kernels, device, K):
    with pytest.raises(RuntimeError, match="1 <= K"):
        kernels.topk(torch.zeros(1, 4, device=device, dtype=torch.bfloat16), K)
