"""Checks of the SPR / SIGReg / InfoNCE building blocks. Run: PYTHONPATH=<repo> JAX_PLATFORMS=cpu python test_spr_components.py"""
import os
import numpy as np
import jax
import jax.numpy as jnp
from flax.core import unfreeze

from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit import windows, ssl_losses
from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.gae import gae_from_deltas
from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.spr import SPR, init_target_from_online, ema_update_target

rng = np.random.default_rng(0)


def test_agent_windows():
    T, E, K = 7, 3, 4
    actions = rng.normal(size=(T, E, 2)).astype(np.float32)
    next_states = rng.normal(size=(T, E, 5)).astype(np.float32)
    dones = np.zeros((T, E), np.float32)
    dones[2, 0] = 1; dones[5, 1] = 1; dones[0, 2] = 1; dones[1, 2] = 1
    flat = lambda x: jnp.asarray(x.reshape((T * E,) + x.shape[2:]))
    idx = jnp.arange(T * E)
    a, x, m = windows.agent_windows(idx, flat(actions), flat(next_states), flat(dones), E, K)
    for t in range(T):
        for e in range(E):
            alive = 1.0
            for k in range(K):
                ok = alive * (t + k < T)
                assert float(m[t * E + e, k]) == ok, (t, e, k)
                if ok:
                    np.testing.assert_allclose(a[t * E + e, k], actions[t + k, e])
                    np.testing.assert_allclose(x[t * E + e, k], next_states[t + k, e])
                if t + k < T:
                    alive *= 1.0 - dones[t + k, e]
    print("agent windows: ok")


def test_expert_windows():
    lengths = [3, 4, 2, 5]
    absorbing = np.concatenate([np.r_[np.zeros(n - 1), 1.0] for n in lengths]).astype(np.float32)
    n = len(absorbing)
    actions = rng.normal(size=(n, 2)).astype(np.float32)
    next_states = rng.normal(size=(n, 5)).astype(np.float32)
    end = windows.expert_episode_end(jnp.asarray(absorbing))
    starts = np.cumsum([0] + lengths[:-1])
    ref_end = np.concatenate([np.full(l, s + l - 1) for s, l in zip(starts, lengths)])
    np.testing.assert_array_equal(np.asarray(end), ref_end)
    K = 4
    a, x, m = windows.expert_windows(jnp.arange(n), jnp.asarray(actions), jnp.asarray(next_states), end, K)
    for i in range(n):
        for k in range(K):
            assert float(m[i, k]) == float(i + k <= ref_end[i]), (i, k)
            if i + k <= ref_end[i]:
                np.testing.assert_allclose(x[i, k], next_states[i + k])
    print("expert windows: ok")


PAPER_DATA_SHA256 = "d1374c0b605dbf22f2901e9846133a5864f850677770a4a90f8575ffdcf8fc9b"


def test_real_expert_data():
    """Windows / boundaries / continuity on the expert file (default: the 600-demo paper file, checked against its hash)."""
    default = os.path.join(os.path.dirname(__file__), "../../trirl_dataset/rl_expert/trirl_expert_data_100_episodes.npz")
    path = os.environ.get("PUSHT_DATA", default)
    if not os.path.exists(path):
        print("real expert data: skipped (not found)"); return
    is_paper = os.path.abspath(path) == os.path.abspath(default)
    if is_paper:
        from trust_region_irl.algorithms.trirl_ppo_spr.flax_full_jit.data_utils import verify_data_sha256
        verify_data_sha256(path, PAPER_DATA_SHA256)
    d = np.load(path)
    s, ns, a, ab = d["states"], d["next_states"], d["actions"], d["absorbing"].reshape(-1)
    contiguous = np.all(np.isclose(ns[:-1], s[1:], atol=1e-6), axis=1)
    assert np.array_equal(~contiguous, ab[:-1] > 0.5), "episodes are not stored contiguously, ending with absorbing=1"
    assert ab[-1] == 1
    end = np.asarray(windows.expert_episode_end(jnp.asarray(ab)))
    n_episodes = int(ab.sum())
    assert (np.diff(end) >= 0).all() and len(np.unique(end)) == n_episodes
    if is_paper:
        assert len(s) == 72789 and n_episodes == 600, (len(s), n_episodes)
        starts = np.r_[0, np.flatnonzero(ab[:-1] > 0.5) + 1]
        first = np.round(s[starts][:, [0, 1, 3, 6]], 3)
        uniq, counts = np.unique(first, axis=0, return_counts=True)
        assert len(uniq) == 6 and (counts == 100).all(), (len(uniq), counts)       # six starts x 100 demonstrations
    # no K-step window leaves its episode: masked-in entries are exactly the in-episode successors
    K = 5
    idx = jnp.arange(len(s))
    a_w, x_w, m_w = windows.expert_windows(idx, jnp.asarray(a), jnp.asarray(ns), jnp.asarray(end), K)
    a_w, x_w, m_w = np.asarray(a_w), np.asarray(x_w), np.asarray(m_w)
    for k in range(K):
        j = np.minimum(np.arange(len(s)) + k, len(s) - 1)
        in_episode = (np.arange(len(s)) + k <= end)
        np.testing.assert_array_equal(m_w[:, k] > 0.5, in_episode)
        assert (end[j][in_episode] == end[in_episode]).all()                      # same episode
        np.testing.assert_allclose(x_w[in_episode, k], ns[j][in_episode])
        np.testing.assert_allclose(a_w[in_episode, k], a[j][in_episode])
    print(f"real expert data: {n_episodes} contiguous episodes, {len(s)} transitions, K={K} windows stay inside episodes, ok")


def test_sigreg():
    key = jax.random.PRNGKey(0)
    N, D = 1024, 64
    z = jax.random.normal(jax.random.PRNGKey(1), (N, D))
    # reference: direct trapezoid integration over the full symmetric interval [-3, 3] for the same directions
    t_pos = np.linspace(0, 3.0, 17)
    t = np.concatenate([-t_pos[:0:-1], t_pos])
    A = np.asarray(jax.random.normal(key, (D, 256)))
    A = A / np.linalg.norm(A, axis=0, keepdims=True)
    proj = np.asarray(z) @ A
    ecf = np.exp(1j * proj[..., None] * t).mean(0)
    err = np.abs(ecf - np.exp(-0.5 * t ** 2)) ** 2 * np.exp(-0.5 * t ** 2)
    ref = (np.trapezoid(err, t, axis=-1) * N).mean()
    ours = float(ssl_losses.sigreg(z, key))
    assert abs(ours - ref) < 1e-3 * max(1.0, abs(ref)), (ours, ref)
    iso, scaled, aniso = float(ssl_losses.sigreg(z, key)), float(ssl_losses.sigreg(2.0 * z, key)), float(ssl_losses.sigreg(z * jnp.linspace(0.05, 1.5, D), key))
    collapsed = float(ssl_losses.sigreg(z[:, :1] * jnp.ones((1, D)), key))
    assert iso < scaled and iso < aniso and iso < collapsed, (iso, scaled, aniso, collapsed)
    g = jax.grad(lambda x: ssl_losses.sigreg(x, key))(2.0 * z)
    assert bool(jnp.all(jnp.isfinite(g)))
    print(f"sigreg: matches direct integration ({ours:.4f} vs {ref:.4f}); N(0,I) {iso:.3f} < scaled x2 {scaled:.1f}, anisotropic {aniso:.1f}, collapsed {collapsed:.1f}")


def test_infonce():
    B, K, P, T = 256, 3, 32, 0.1
    key = jax.random.PRNGKey(0)
    targets = jax.random.normal(key, (B, K, P))
    mask = jnp.ones((B, K))
    ids = jnp.arange(B)
    rand_preds = jax.random.normal(jax.random.PRNGKey(1), (B, K, P))
    loss_rand, _ = ssl_losses.infonce_terms(rand_preds, targets, mask, ids, ids, T, 0)
    # random unit vectors: cosines ~ N(0, 1/P), logits have std 1/(sqrt(P) T), so E[loss] ~ log B + sigma^2 / 2
    expected = np.log(B) + 0.5 / (P * T ** 2)
    assert abs(float(loss_rand.mean()) - expected) < 0.3, (float(loss_rand.mean()), expected)
    loss_pos, cos = ssl_losses.infonce_terms(targets, targets, mask, ids, ids, T, 0)
    assert float(loss_pos.mean()) < 0.2 and float(cos.mean()) > 0.999
    # duplicate sample (same id) with identical target must not count as a negative
    dup_targets = targets.at[1].set(targets[0])
    dup_ids = ids.at[1].set(0)
    masked, _ = ssl_losses.infonce_terms(dup_targets, dup_targets, mask, dup_ids, dup_ids, T, 0)
    unmasked, _ = ssl_losses.infonce_terms(dup_targets, dup_targets, mask, ids, ids, T, 0)
    assert float(masked[0].mean()) < 0.2 and float(unmasked[0].mean()) > 0.6, (float(masked[0].mean()), float(unmasked[0].mean()))
    # an invalid step contributes no loss and is not a negative
    m2 = mask.at[5, 1].set(0.0)
    l2, _ = ssl_losses.infonce_terms(targets, targets, m2, ids, ids, T, 0)
    assert float(l2[5, 1]) == 0.0
    print(f"infonce: random {float(loss_rand.mean()):.3f} (expected {expected:.3f}), perfect {float(loss_pos.mean()):.4f}, duplicate masked {float(masked[0].mean()):.3f} vs unmasked {float(unmasked[0].mean()):.3f}")


def test_byol_s_equivalence():
    from trust_region_irl.algorithms.trirl_ppo_byol_s.flax_full_jit.byol import BYOL
    B = 8
    x = jax.random.normal(jax.random.PRNGKey(0), (B, 24))
    xn = jax.random.normal(jax.random.PRNGKey(1), (B, 24))
    a = jax.random.normal(jax.random.PRNGKey(2), (B, 2))
    ref_model = BYOL()
    ref = ref_model.init(jax.random.PRNGKey(3), x, a, xn)["params"]
    # random theta so that the reward check is not trivially x*0.001
    ref = dict(ref); ref["theta"] = jax.random.normal(jax.random.PRNGKey(4), (64,))
    conv = lambda prefix: {"Dense_0": ref[f"{prefix}1"], "LayerNorm_0": ref[f"{prefix}_norm"], "Dense_1": ref[f"{prefix}2"]}
    enc = lambda kind: {"Dense_0": ref[f"{kind}_enc1"], "LayerNorm_0": ref[f"{kind}_enc_norm"], "Dense_1": ref[f"{kind}_enc2"], "Dense_2": ref[f"{kind}_enc3"]}
    proj = lambda kind: {"Dense_0": ref[f"{kind}_proj1"], "LayerNorm_0": ref[f"{kind}_proj_norm"], "Dense_1": ref[f"{kind}_proj2"]}
    mine = {"params": {
        "online_encoder": enc("online"), "online_projector": proj("online"),
        "target_encoder": enc("target"), "target_projector": proj("target"),
        "predictor": {"Dense_0": ref["online_pred1"], "LayerNorm_0": ref["online_pred_norm"], "Dense_1": ref["online_pred2"]},
        "theta": ref["theta"]}}
    model = SPR(predictor_mode="byol_s")
    z, p, y = model.apply(mine, x, a[:, None], xn[:, None])
    z_ref, p_ref, y_ref = ref_model.apply({"params": ref}, x, a, xn)
    np.testing.assert_allclose(z, z_ref, atol=1e-5); np.testing.assert_allclose(p[:, 0], p_ref, atol=1e-5); np.testing.assert_allclose(y[:, 0], y_ref, atol=1e-5)
    np.testing.assert_allclose(model.apply(mine, x, a, method=model._reward), ref_model.apply({"params": ref}, x, a, method=ref_model._reward), atol=1e-5)
    # init has the same parameter structure as the hand-mapped one
    init = model.init(jax.random.PRNGKey(0), x, a[:, None], xn[:, None])
    assert jax.tree_util.tree_structure(init) == jax.tree_util.tree_structure(mine)
    print("byol_s mode == teammate's BYOL (z, p, y, reward): ok")


def test_spr_mode():
    B, K = 8, 5
    x = jax.random.normal(jax.random.PRNGKey(0), (B, 24))
    a = jax.random.normal(jax.random.PRNGKey(1), (B, K, 2))
    xn = jax.random.normal(jax.random.PRNGKey(2), (B, K, 24))
    model = SPR(predictor_mode="spr")
    params = init_target_from_online(model.init(jax.random.PRNGKey(3), x, a, xn))
    z, p, y = model.apply(params, x, a, xn)
    assert z.shape == (B, 64) and p.shape == (B, K, 128) and y.shape == (B, K, 128)
    # the target network gets no gradient, the transition model and predictor do
    g = jax.grad(lambda pr: ssl_losses.cosine_terms(*model.apply(pr, x, a, xn)[1:], jnp.ones((B, K)))[0].sum())(params)["params"]
    assert all(float(jnp.abs(l).sum()) == 0 for l in jax.tree_util.tree_leaves({k: g[k] for k in ("target_encoder", "target_projector")}))
    assert all(float(jnp.abs(l).sum()) > 0 for l in jax.tree_util.tree_leaves({k: g[k] for k in ("transition", "predictor", "online_projector", "online_encoder")}))
    # the first prediction only depends on the first action, the second on the first two
    a2 = a.at[:, 1:].set(0.0)
    p2 = model.apply(params, x, a2, xn)[1]
    np.testing.assert_allclose(p[:, 0], p2[:, 0], atol=1e-5)
    assert float(jnp.abs(p[:, 1] - p2[:, 1]).max()) > 1e-4
    ema = ema_update_target(params, 0.99)["params"]
    np.testing.assert_allclose(ema["target_encoder"]["Dense_0"]["kernel"], 0.99 * params["params"]["target_encoder"]["Dense_0"]["kernel"] + 0.01 * params["params"]["online_encoder"]["Dense_0"]["kernel"], atol=1e-6)
    n = sum(int(l.size) for l in jax.tree_util.tree_leaves(params))
    n_b = sum(int(l.size) for l in jax.tree_util.tree_leaves(SPR(predictor_mode="byol_s").init(jax.random.PRNGKey(0), x, a[:, :1], xn[:, :1])))
    print(f"spr mode: shapes, gradients, action causality, EMA ok; parameters: spr {n}, byol_s {n_b}")


def _window(B=16, K=5, seed=0):
    x = jax.random.normal(jax.random.PRNGKey(seed), (B, 24))
    a = jax.random.normal(jax.random.PRNGKey(seed + 1), (B, K, 2))
    xn = jax.random.normal(jax.random.PRNGKey(seed + 2), (B, K, 24))
    return x, a, xn


def test_lejepa_mode():
    B, K = 16, 5
    x, a, xn = _window(B, K)
    model = SPR(predictor_mode="lejepa")
    params = unfreeze(model.init(jax.random.PRNGKey(3), x, a, xn))               # the trainer unfreezes as well
    names = set(params["params"].keys())
    assert names == {"online_encoder", "transition", "theta"}, names              # no projector, predictor, target network
    same = jax.tree_util.tree_structure(init_target_from_online(params)) == jax.tree_util.tree_structure(params)
    same &= jax.tree_util.tree_structure(ema_update_target(params, 0.99)) == jax.tree_util.tree_structure(params)
    assert same, "EMA helpers must be no-ops for modes without target networks"
    z, p, y = model.apply(params, x, a, xn)
    assert p.shape == (B, K, 64) and y.shape == (B, K, 64)
    np.testing.assert_allclose(y, model.apply(params, xn, method=model.encode), atol=1e-6)   # target = the online encoder

    mask = jnp.ones((B, K))
    def loss(pr, detach_target):
        _, pp, yy = model.apply(pr, x, a, xn)
        yy = jax.lax.stop_gradient(yy) if detach_target else yy
        return ssl_losses.mse_terms(pp, yy, mask)[0].sum(1).mean()
    assert abs(float(loss(params, True)) - float(loss(params, False))) < 1e-7
    g_full, g_sg = jax.grad(loss)(params, False), jax.grad(loss)(params, True)
    diff = sum(float(jnp.abs(u - v).sum()) for u, v in zip(jax.tree_util.tree_leaves(g_full), jax.tree_util.tree_leaves(g_sg)))
    assert diff > 1e-6, "the target side must receive gradient (no stop-gradient)"
    # mean over the latent dimensions, summed over K, masked
    per_step, _ = ssl_losses.mse_terms(jnp.ones((2, 3, 64)) + 1.0, jnp.ones((2, 3, 64)), jnp.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
    np.testing.assert_allclose(per_step, [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    # compute_ssl wiring, including SIGReg on z
    win = dict(states=x, a_seq=a, x_next=xn, mask=mask, episode_key=jnp.arange(B), time_key=jnp.zeros(B))
    ssl, sg, m = ssl_losses.compute_ssl(model, params, win, jax.random.PRNGKey(5), ssl_loss="mse", temperature=0.1, mask_radius=0,
                                        w_sigreg=0.05, sigreg_slices=64, sigreg_on="z")
    assert np.isfinite(float(ssl)) and float(sg) > 0
    print(f"lejepa mode: parameters only {sorted(names)}, target gradient flows (|dg|={diff:.2e}), mse/mask ok, compute_ssl ok")


def test_bilinear_mode():
    B, K = 16, 5
    x, a, xn = _window(B, K)
    model = SPR(predictor_mode="bilinear", num_steps=K)
    params = model.init(jax.random.PRNGKey(3), x, a, xn)
    names = set(params["params"].keys())
    assert names == {"online_encoder", "transition", "theta", "bilinear_w"}, names
    assert params["params"]["bilinear_w"].shape == (K, 64, 64) and float(jnp.abs(params["params"]["bilinear_w"]).sum()) == 0.0
    mask = jnp.ones((B, K)); ids = jnp.arange(B)
    win = dict(states=x, a_seq=a, x_next=xn, mask=mask, episode_key=ids, time_key=jnp.zeros(B))
    kw = dict(ssl_loss="bilinear_infonce", temperature=0.1, mask_radius=0, w_sigreg=0.0, sigreg_slices=64, sigreg_on="z")
    ssl, _, _ = ssl_losses.compute_ssl(model, params, win, jax.random.PRNGKey(5), **kw)
    # W_k = 0 -> all logits equal -> every step costs exactly log(#candidates); summed over K steps
    assert abs(float(ssl) - K * np.log(B)) < 1e-4, (float(ssl), K * np.log(B))
    # duplicates (same episode and time) are not negatives: sample 1 duplicates sample 0 -> 15 candidates for them
    dup = dict(win, episode_key=ids.at[1].set(0))
    ssl_dup, _, _ = ssl_losses.compute_ssl(model, params, dup, jax.random.PRNGKey(5), **kw)
    expected = K * (np.log(B) * (B - 2) + 2 * np.log(B - 1)) / B
    assert abs(float(ssl_dup) - expected) < 1e-4, (float(ssl_dup), expected)

    def loss(pr, detach):
        _, pp, yy = model.apply(pr, x, a, xn)
        if detach == "target":
            yy = jax.lax.stop_gradient(yy)
        if detach == "context":
            pp = jax.lax.stop_gradient(pp)
        logits = model.apply(pr, pp, yy, method=model.bilinear_logits)
        return ssl_losses.infonce_from_logits(logits, mask, ids, jnp.zeros(B), 0).sum(1).mean()
    g0 = jax.grad(loss)(params, None)["params"]
    assert all(float(jnp.abs(l).sum()) == 0 for l in jax.tree_util.tree_leaves({k: g0[k] for k in ("online_encoder", "transition")}))
    assert float(jnp.abs(g0["bilinear_w"]).sum()) > 0                       # at W=0 only W learns
    # after W moves, gradient reaches the encoder through BOTH the context side and the target side
    p2 = jax.tree_util.tree_map(lambda t: t, params)
    p2 = {"params": {**p2["params"], "bilinear_w": 0.1 * jax.random.normal(jax.random.PRNGKey(9), (K, 64, 64))}}
    dist = lambda ga, gb: sum(float(jnp.abs(u - v).sum()) for u, v in zip(jax.tree_util.tree_leaves(ga), jax.tree_util.tree_leaves(gb)))
    gf = jax.grad(loss)(p2, None)
    d_t, d_c = dist(gf, jax.grad(loss)(p2, "target")), dist(gf, jax.grad(loss)(p2, "context"))
    assert d_t > 1e-6 and d_c > 1e-6, (d_t, d_c)
    print(f"bilinear mode: params {sorted(names)}, loss at W=0 is K*log(B) ({float(ssl):.4f}), duplicates masked, "
          f"gradient through target side ({d_t:.2e}) and context side ({d_c:.2e}) once W != 0")


def test_gae_truncation():
    gamma, lam = 0.99, 0.95
    # one env: episode 1 = steps 0-1 (reward 1, then a time-out), episode 2 = steps 2-3 with reward 100; V = 0
    delta = jnp.array([[1.0], [1.0], [100.0], [100.0]])
    terminated = jnp.zeros((4, 1)); truncated = jnp.array([[0.0], [1.0], [0.0], [0.0]])
    original = gae_from_deltas(delta, terminated, gamma, lam)                 # cut at terminations only (old behaviour)
    fixed = gae_from_deltas(delta, jnp.maximum(terminated, truncated), gamma, lam)
    assert abs(float(original[1, 0]) - (1.0 + gamma * lam * 195.0 * 0 + gamma * lam * float(original[2, 0]))) < 1e-4
    assert float(original[1, 0]) > 90.0, float(original[1, 0])               # leak: step 1 absorbs episode 2
    assert abs(float(fixed[1, 0]) - 1.0) < 1e-6                              # fixed: the last step of episode 1 sees only its own delta
    assert abs(float(fixed[0, 0]) - (1.0 + gamma * lam * 1.0)) < 1e-6
    np.testing.assert_allclose(fixed[2:], original[2:])                      # the second episode is unchanged
    # no episode ends: identical to the plain recursion
    plain = delta[-1:]
    for t in range(2, -1, -1):
        plain = jnp.concatenate([delta[t:t + 1] + gamma * lam * plain[:1], plain])
    np.testing.assert_allclose(gae_from_deltas(delta, jnp.zeros((4, 1)), gamma, lam), plain[:4], rtol=1e-5)
    print(f"gae: time-out leak reproduced (old {float(original[1, 0]):.2f}) and fixed (new {float(fixed[1, 0]):.2f}); other cases unchanged")


if __name__ == "__main__":
    test_agent_windows(); test_expert_windows(); test_real_expert_data()
    test_sigreg(); test_infonce(); test_byol_s_equivalence(); test_spr_mode(); test_lejepa_mode(); test_bilinear_mode(); test_gae_truncation()
    print("ALL OK")
