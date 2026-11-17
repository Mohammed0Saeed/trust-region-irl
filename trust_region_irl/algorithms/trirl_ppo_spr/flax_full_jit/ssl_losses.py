"""Self-supervised losses on the K-step predictions of spr.SPR, and SIGReg.

cosine : SPR loss, sum over the K steps of the normalised-L2 loss 2 - 2 cos(pred_k, target_k), masked at episode
         boundaries, weight 1 (official SPR: model_spr_weight=5 on the *mean* of K=5 terms, t0 term off).
infonce: the "temporal contrastive loss with an explicit dynamics model" of SPR appendix B (similar to CPC|Action):
         cosine similarities / temperature, the targets of the other samples in the batch are the negatives.
         Per step this is InfoNCE of CPC (Oord et al. 2018, eq. 4): -log f(pos) / sum_{x_j in X} f(x_j),
         with X containing the positive and the negatives.
mse    : LeJEPA-style prediction loss, squared error averaged over the latent dimensions (official MINIMAL.md:
         `(proj.mean(0) - proj).square().mean()`), summed over the K steps, masked at episode boundaries.
bilinear_infonce: CPC-style InfoNCE with the unnormalised bilinear score f(s_j)^T W_k c_i (CPC eq. 3), no temperature.
sigreg : Sketched Isotropic Gaussian Regularisation of LeJEPA (Balestriero & LeCun, https://arxiv.org/abs/2511.08544),
         a port of the official implementation (github.com/rbalestr-lab/lejepa, MINIMAL.md / epps_pulley.py).
"""
import jax
import jax.numpy as jnp

HIGHEST = jax.lax.Precision.HIGHEST


def l2_normalize(x, eps=1e-12):
    return x * jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def cosine_terms(preds, targets, mask):
    """preds, targets: (B, K, P); mask: (B, K). Returns per-step loss (B, K) and the cosines (B, K)."""
    cos = jnp.sum(l2_normalize(preds) * l2_normalize(targets), axis=-1)
    return (2.0 - 2.0 * cos) * mask, cos


def mse_terms(preds, targets, mask):
    """preds, targets: (B, K, D); mask: (B, K). Squared error averaged over the D dimensions, masked. Returns the
    per-step loss (B, K) and the pred/target cosines (B, K) (diagnostic only)."""
    per_step = jnp.mean((preds - targets) ** 2, axis=-1) * mask
    cos = jnp.sum(l2_normalize(preds) * l2_normalize(targets), axis=-1)
    return per_step, cos


def infonce_from_logits(logits, mask, episode_key, time_key, mask_radius):
    """logits: (K, B, B), logits[k, i, j] = score of anchor i against candidate j at step k; mask: (B, K).

    The positive of sample i is candidate i; the other candidates j with a valid step k are the negatives. Samples of
    the same episode within `mask_radius` time steps (mask_radius=0: only exact duplicates, which the with-replacement
    expert sampling produces) are not used as negatives. CPC eq. 4: the denominator contains the positive.
    Returns the per-step loss (B, K)."""
    batch = logits.shape[-1]
    eye = jnp.eye(batch, dtype=bool)
    same = (episode_key[:, None] == episode_key[None, :]) & (jnp.abs(time_key[:, None] - time_key[None, :]) <= mask_radius)
    allowed = ((~(same & ~eye))[None] & (mask.T > 0.5)[:, None, :]) | eye[None]
    log_probs = jax.nn.log_softmax(jnp.where(allowed, logits, -1e9), axis=-1)
    nce = -jnp.diagonal(log_probs, axis1=-2, axis2=-1)  # (K, B)
    return nce.T * mask


def infonce_terms(preds, targets, mask, episode_key, time_key, temperature, mask_radius):
    """preds, targets: (B, K, P); mask: (B, K); episode_key, time_key: (B,). Cosine similarity / temperature scores
    (SPR appendix B). Returns the per-step loss (B, K) and the positive-pair cosines (B, K)."""
    p, t = l2_normalize(preds), l2_normalize(targets)
    logits = jnp.einsum("ikd,jkd->kij", p, t, precision=HIGHEST) / temperature  # (K, B, B)
    pos_cos = jnp.diagonal(logits, axis1=-2, axis2=-1).T * temperature  # (B, K)
    return infonce_from_logits(logits, mask, episode_key, time_key, mask_radius), pos_cos


def sigreg(z, key, num_slices=256, t_max=3.0, knots=17):
    """z: (N, D). Epps-Pulley statistic against N(0, I) on `num_slices` random unit directions (resampled by the
    caller's key every call), averaged over the directions. Integrates on [0, t_max] and doubles (the characteristic
    function is symmetric), trapezoid weights times the window exp(-t^2/2), statistic scaled by N."""
    t = jnp.linspace(0.0, t_max, knots)
    dt = t_max / (knots - 1)
    window = jnp.exp(-0.5 * t ** 2)
    weights = jnp.full((knots,), 2.0 * dt).at[0].set(dt).at[-1].set(dt) * window
    directions = jax.random.normal(key, (z.shape[-1], num_slices))
    directions = directions / jnp.linalg.norm(directions, axis=0, keepdims=True)
    x_t = jnp.matmul(z, directions, precision=HIGHEST)[..., None] * t  # (N, S, knots)
    err = (jnp.cos(x_t).mean(0) - window) ** 2 + jnp.sin(x_t).mean(0) ** 2
    return (jnp.sum(err * weights, axis=-1) * z.shape[0]).mean()


def participation_ratio(z):
    """(tr C)^2 / ||C||_F^2 of the feature covariance: ~D for isotropic features, ~1 for a collapsed one."""
    c = jnp.cov(z, rowvar=False)
    return jnp.trace(c) ** 2 / (jnp.sum(c * c) + 1e-12)


def compute_ssl(discriminator, params, win, key, *, ssl_loss, temperature, mask_radius, w_sigreg, sigreg_slices, sigreg_on):
    """SSL part of the discriminator loss on one batch.

    win: dict with states (B, obs), a_seq (B, K, A), x_next (B, K, obs), mask (B, K), episode_key (B,), time_key (B,).
    Returns (ssl_loss, sigreg_loss, metrics); ssl_loss = mean over the batch of the sum over the K steps."""
    z, preds, targets = discriminator.apply(params, win["states"], win["a_seq"], win["x_next"])
    mask = win["mask"][:, : preds.shape[1]]
    targets = targets[:, : preds.shape[1]]
    if ssl_loss == "cosine":
        per_step, cos = cosine_terms(preds, targets, mask)
    elif ssl_loss == "infonce":
        per_step, cos = infonce_terms(preds, targets, mask, win["episode_key"], win["time_key"], temperature, mask_radius)
    elif ssl_loss == "mse":
        per_step, cos = mse_terms(preds, targets, mask)
    elif ssl_loss == "bilinear_infonce":
        logits = discriminator.apply(params, preds, targets, method=discriminator.bilinear_logits)
        per_step = infonce_from_logits(logits, mask, win["episode_key"], win["time_key"], mask_radius)
        cos = jnp.sum(l2_normalize(preds) * l2_normalize(targets), axis=-1)  # diagnostic only
    else:
        raise ValueError(f"ssl_loss must be 'cosine', 'infonce', 'mse' or 'bilinear_infonce', got '{ssl_loss}'")
    ssl = jnp.mean(jnp.sum(per_step, axis=1))

    sigreg_loss = jnp.asarray(0.0)
    if w_sigreg > 0.0:
        features = z if sigreg_on == "z" else discriminator.apply(params, z, method=discriminator.project)
        sigreg_loss = sigreg(features, key, sigreg_slices)

    valid = jnp.maximum(jnp.sum(mask), 1.0)
    metrics = {
        "loss/ssl_loss": ssl,
        "loss/sigreg": sigreg_loss,
        "ssl/valid_fraction": jnp.mean(mask),
        "ssl/pred_target_cosine": jnp.sum(cos * mask) / valid,
        "z/std": jnp.mean(jnp.std(z, axis=0)),
        "z/norm": jnp.mean(jnp.linalg.norm(z, axis=-1)),
        "z/participation_ratio": participation_ratio(z),
    }
    for k in range(per_step.shape[1]):
        metrics[f"ssl/loss_step{k + 1}"] = jnp.mean(per_step[:, k])
    if "is_expert" in win:
        # Diagnostic only (no gradient effect): the shared batch holds agent and expert windows; report them separately.
        per_sample = jnp.sum(per_step, axis=1)
        for name, weight in (("agent", 1.0 - win["is_expert"]), ("expert", win["is_expert"])):
            metrics[f"ssl/loss_{name}"] = jnp.sum(per_sample * weight) / jnp.maximum(jnp.sum(weight), 1.0)
            valid_weight = mask * weight[:, None]
            metrics[f"ssl/pred_target_cosine_{name}"] = jnp.sum(cos * valid_weight) / jnp.maximum(jnp.sum(valid_weight), 1.0)
    return ssl, sigreg_loss, metrics
