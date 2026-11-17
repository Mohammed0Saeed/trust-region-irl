"""Reward encoder with a BYOL-style target network and an action-conditioned latent predictor.

predictor_mode="byol_s"  (group B, identical to trirl_ppo_byol_s):
    p = q([g(f(s)), a_t])                 predicts g'(f'(s_{t+1}))                      (1 step)
predictor_mode="spr"     (SPR, Schwarzer et al. 2021, https://arxiv.org/abs/2007.05929):
    z_0 = f(s_t),  z_k = h(z_{k-1}, a_{t+k-1}),  p_k = q(g(z_k))   predicts g'(f'(s_{t+k})), k = 1..K

predictor_mode="lejepa"  (LeJEPA-style, exploratory; Balestriero & LeCun, arXiv:2511.08544):
    z_0 = f(s_t),  z_k = h(z_{k-1}, a_{t+k-1}),  p_k = z_k    predicts f(s_{t+k}), k = 1..K
    No projector, no predictor, no EMA and NO stop-gradient: the target is the same online encoder (the official
    MINIMAL.md uses `(proj.mean(0) - proj).square().mean()` without detach). Anti-collapse comes from SIGReg only.
predictor_mode="bilinear"  (CPC-style, exploratory; van den Oord et al., arXiv:1807.03748):
    same rollout, score(i, j, k) = f(s_{j,t+k})^T W_k z_{i,k} (unnormalised, one matrix W_k per step, initialised to 0),
    no EMA, no stop-gradient. The history network of CPC is dropped (the context is f(s_t) alone) and the action
    enters through the transition model (CPC|Action); the observation has no block velocity, so this is a
    simplification, not a confirmed Markov property.

f, g are the online encoder / projector, f', g' their EMA copies (target); q is the predictor.
The reward is r(s) = theta^T f(s). Where this differs from the official SPR code
(github.com/mila-iqia/spr, branch `release`) is documented next to the code:
  * MLP + LayerNorm instead of conv + BatchNorm (vector observations),
  * the continuous action is concatenated to the latent (SPR concatenates a one-hot action map),
  * no min-max "renormalize" of the latents. This is an adaptation, not something the theory asks for: it keeps the reward
    feature z identical in form to BYOL-S (group B), so the groups differ only in the predictor and the loss.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal, zeros_init

# the pairs (online -> target) that follow each other by EMA
EMA_PAIRS = (("online_encoder", "target_encoder"), ("online_projector", "target_projector"))


def _dense(features):
    return nn.Dense(features, kernel_init=orthogonal(1.0), bias_init=constant(0.0))


class Encoder(nn.Module):
    """f: 256, LN, 256, 64 (as in BYOL-S)"""
    latent_dim: int = 64
    hidden_dim: int = 256
    input_mode: str = "full26"

    @nn.compact
    def __call__(self, x):
        if self.input_mode != "full26":
            if self.input_mode not in ("task12", "object9") or x.shape[-1] != 26:
                raise ValueError("Restricted reward inputs require v3 26D observations")
            keep = 12 if self.input_mode == "task12" else 9
            x = x * (jnp.arange(26) < keep)  # preserve parameter shape; mask before ALL mixing
        z = _dense(self.hidden_dim)(x)
        z = nn.LayerNorm()(z)
        z = nn.relu(z)
        z = _dense(self.hidden_dim)(z)
        z = nn.relu(z)
        return _dense(self.latent_dim)(z)


class Projector(nn.Module):
    """g: 256, LN, 128 (as in BYOL-S)"""
    proj_dim: int = 128
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, z):
        y = _dense(self.hidden_dim)(z)
        y = nn.LayerNorm()(y)
        y = nn.relu(y)
        return _dense(self.proj_dim)(y)


class MLPPredictor(nn.Module):
    """BYOL-S predictor q([y, a]): 256, LN, 128"""
    proj_dim: int = 128
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, y, a):
        p = jnp.concatenate((y, a), axis=-1)
        p = _dense(self.hidden_dim)(p)
        p = nn.LayerNorm()(p)
        p = nn.relu(p)
        return _dense(self.proj_dim)(p)


class TransitionModel(nn.Module):
    """h(z, a): latent dynamics. [z, a] -> 256, LN, ReLU -> latent_dim (non-residual by default, like SPR)."""
    latent_dim: int = 64
    hidden_dim: int = 256
    residual: bool = False

    @nn.compact
    def __call__(self, z, a):
        h = jnp.concatenate((z, a), axis=-1)
        h = _dense(self.hidden_dim)(h)
        h = nn.LayerNorm()(h)
        h = nn.relu(h)
        h = _dense(self.latent_dim)(h)
        return z + h if self.residual else h


PREDICTOR_MODES = ("byol_s", "spr", "lejepa", "bilinear")


class SPR(nn.Module):
    predictor_mode: str = "byol_s"  # "byol_s" | "spr" | "lejepa" | "bilinear"
    latent_dim: int = 64
    proj_dim: int = 128
    hidden_dim: int = 256
    transition_residual: bool = False
    num_steps: int = 5  # K, only used by "bilinear" (one W_k per step)
    reward_input_mode: str = "full26"

    def setup(self):
        if self.predictor_mode not in PREDICTOR_MODES:
            raise ValueError(f"predictor_mode must be one of {PREDICTOR_MODES}, got '{self.predictor_mode}'")
        self.online_encoder = Encoder(self.latent_dim, self.hidden_dim, self.reward_input_mode)
        # Flax creates parameters only for modules that are called, so the modes without EMA / projector never get
        # target_* or *_projector parameters (and init_target_from_online / ema_update_target skip the missing pairs).
        self.online_projector = Projector(self.proj_dim, self.hidden_dim)
        self.target_encoder = Encoder(self.latent_dim, self.hidden_dim, self.reward_input_mode)
        self.target_projector = Projector(self.proj_dim, self.hidden_dim)
        if self.predictor_mode == "byol_s":
            self.predictor = MLPPredictor(self.proj_dim, self.hidden_dim)
        else:
            self.transition = TransitionModel(self.latent_dim, self.hidden_dim, self.transition_residual)
            if self.predictor_mode == "spr":
                self.predictor = _dense(self.proj_dim)  # linear q, as in SPR
            elif self.predictor_mode == "bilinear":
                self.bilinear_w = self.param("bilinear_w", zeros_init(), (self.num_steps, self.latent_dim, self.latent_dim))
        self.theta = self.param("theta", constant(0.001), (self.latent_dim,))

    # ---- reward -------------------------------------------------------------------------
    def _reward(self, x, a=None):
        return self.online_encoder(x) @ self.theta

    # ---- pieces -------------------------------------------------------------------------
    def encode(self, x):
        return self.online_encoder(x)

    def project(self, z):
        return self.online_projector(z)

    def target_projection(self, x):
        """g'(f'(x)), no gradient"""
        return jax.lax.stop_gradient(self.target_projector(self.target_encoder(x)))

    def predictions(self, z, a_seq):
        """z: (..., latent_dim), a_seq: (..., K, action_dim) -> (..., K, proj_dim)"""
        if self.predictor_mode == "byol_s":
            y = self.online_projector(z)
            return self.predictor(y, a_seq[..., 0, :])[..., None, :]
        preds = []
        h = z
        for k in range(a_seq.shape[-2]):
            h = self.transition(h, a_seq[..., k, :])
            # lejepa / bilinear predict in the latent space itself; spr goes through projector and linear predictor
            preds.append(h if self.predictor_mode in ("lejepa", "bilinear") else self.predictor(self.online_projector(h)))
        return jnp.stack(preds, axis=-2)

    def bilinear_logits(self, preds, targets):
        """preds, targets: (B, K, D) -> logits (K, B, B), logits[k, i, j] = targets[j, k]^T W_k preds[i, k] (no scaling)."""
        projected = jnp.einsum("ike,kde->ikd", preds, self.bilinear_w, precision=jax.lax.Precision.HIGHEST)
        return jnp.einsum("ikd,jkd->kij", projected, targets, precision=jax.lax.Precision.HIGHEST)

    def __call__(self, x, a_seq, x_next_seq):
        """x: (..., obs), a_seq: (..., K, act), x_next_seq: (..., K, obs) ->
        z (..., latent), predictions (..., K', proj), targets (..., K, proj)   [K' = 1 for 'byol_s']"""
        z = self.online_encoder(x)
        if self.predictor_mode in ("lejepa", "bilinear"):
            targets = self.online_encoder(x_next_seq)  # same online encoder, gradients flow (no stop-gradient, no EMA)
        else:
            targets = self.target_projection(x_next_seq)
        return z, self.predictions(z, a_seq), targets


def init_target_from_online(params):
    """Make the target networks an exact copy of the online ones (after init)."""
    params = dict(params)
    params["params"] = dict(params["params"])
    for online, target in EMA_PAIRS:
        # only the modes that use a target network have it in the initialised tree (Flax creates parameters of called
        # modules only); lejepa / bilinear have no target network and must not get a copy
        if target in params["params"]:
            params["params"][target] = params["params"][online]
    return params


def ema_update_target(params, momentum):
    """target <- momentum * target + (1 - momentum) * online  (encoder and projector; predictor stays online-only)"""
    params = dict(params)
    params["params"] = dict(params["params"])
    for online, target in EMA_PAIRS:
        if target not in params["params"]:
            continue
        params["params"][target] = jax.tree_util.tree_map(
            lambda t, o: momentum * t + (1.0 - momentum) * o, params["params"][target], params["params"][online])
    return params


def count_params(params):
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(params))
