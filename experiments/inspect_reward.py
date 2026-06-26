"""Read-only inspection of the learned feature-based reward.

Loads a trirl_ppo_fb / trirl_trpl_fb checkpoint and reports, for the linear
feature reward  r = theta^T phi  with  phi = [pos_err, orn_err, ee_to_block_dist]:

  * the raw learned weights theta
  * per-feature std over the expert data (features live on different scales)
  * the *effective contribution*  |theta_i| * std(phi_i)  of each feature

Nothing is trained and nothing in the repo is modified. Run with CPU forced:

    JAX_PLATFORMS=cpu python experiments/inspect_reward.py [MODEL_ZIP] [EXPERT_NPZ]
"""
import sys, zipfile, tempfile, json
import numpy as np
import jax
import orbax.checkpoint as ocp

FEATURE_NAMES = ["pos_err", "orn_err", "ee_to_block_dist"]

DEFAULT_MODEL = "experiments/runs/trust_region_irl/pusht_ppo_fb_editted_hyperparams/1782035901/models/latest.model.zip"
DEFAULT_EXPERT = "trirl_dataset/rl_expert/expert_dataset_pusht_50_episodes.npz"


def base_features(obs, act):
    """Mirror franka_pusht environment.feature_from_transition (feature_fn='base')."""
    obs = np.asarray(obs, np.float32)
    act = np.asarray(act, np.float32)
    pos_err = np.linalg.norm(obs[:, 0:3], axis=-1)
    w = np.clip(np.abs(obs[:, 3]), 0.0, 1.0)
    orn_err = 2.0 * np.arccos(w)
    ee_to_block = np.linalg.norm(obs[:, 0:3] - obs[:, 7:10], axis=-1)
    return np.stack([pos_err, orn_err, ee_to_block], axis=-1)


def find_theta(tree):
    """Recursively locate the (4,) theta leaf in a restored checkpoint dict."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            if k == "theta" and hasattr(v, "shape"):
                return np.asarray(v)
            r = find_theta(v)
            if r is not None:
                return r
    return None


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    expert = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_EXPERT

    d = tempfile.mkdtemp()
    with zipfile.ZipFile(model) as z:
        z.extractall(d)
        cfg = json.load(z.open("config_algorithm.json"))
    ckpt = ocp.PyTreeCheckpointer().restore(d)

    theta_disc = find_theta(ckpt.get("discriminator", {}))
    theta_corr = find_theta(ckpt.get("corrected_reward", {}))

    # The reward the POLICY optimizes (trirl_ppo.py:387):
    #   r = entropy_coef * [ (1-eps)*theta_corr.phi + eps*beta*theta_disc.phi ]
    # entropy_coef is a positive global scale -> irrelevant to signs/shares; we
    # fold it in anyway so magnitudes match what the policy sees.
    eps = float(cfg["epsilon"]); beta = float(cfg["beta"]); ec = float(cfg["entropy_coef"])
    theta = ec * ((1.0 - eps) * theta_corr + eps * beta * theta_disc)
    print(f"blend: entropy_coef={ec}, epsilon={eps:.4f}, beta={beta}  "
          f"=> corr weight={(1-eps):.3f}, disc weight={eps*beta:.1f}")
    print(f"  theta_disc = {np.round(theta_disc,4)}")
    print(f"  theta_corr = {np.round(theta_corr,4)}")
    print(f"  theta_EFFECTIVE (policy reward) = {np.round(theta,4)}\n")

    data = np.load(expert, allow_pickle=True)
    obs, act = data["states"], data["actions"]
    phi = base_features(obs, act)
    phi_std = phi.std(0)
    phi_mean = phi.mean(0)

    print(f"model:  {model}")
    print(f"expert: {expert}   (N={len(obs)} transitions)\n")

    print(f"{'feature':18s}{'theta':>12s}{'phi_mean':>12s}{'phi_std':>12s}"
          f"{'|theta|*std':>14s}")
    contrib = np.abs(theta) * phi_std
    for i, name in enumerate(FEATURE_NAMES):
        print(f"{name:18s}{theta[i]:12.4f}{phi_mean[i]:12.4f}{phi_std[i]:12.4f}{contrib[i]:14.4f}")

    share = 100 * contrib / contrib.sum()
    print("\neffective contribution share (%):")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:18s}{share[i]:6.1f}%")


if __name__ == "__main__":
    main()
