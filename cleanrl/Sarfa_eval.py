import json
import importlib
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import tyro
from stable_baselines3.common.vec_env import DummyVecEnv


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
for _p in (str(THIS_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from cleanrl import ppo_atari_oc as train_mod
except ModuleNotFoundError:
    import ppo_atari_oc as train_mod


def _import_ppo_architectures() -> tuple[type[torch.nn.Module], type[torch.nn.Module]]:
    try:
        from cleanrl.architectures.ppo import PPODefault, PPObj
    except ModuleNotFoundError:
        from architectures.ppo import PPODefault, PPObj
    return PPODefault, PPObj


@dataclass
class EvalArgs:
    checkpoint: str
    """Path to .cleanrl_model checkpoint."""
    episodes: int = 10
    """Number of evaluation episodes."""
    output_json: str = ""
    """Output JSON path. Empty -> auto under evaluation/<game>/..."""
    seed: int = 0 #42
    """Evaluation seed (per env i uses seed+i)."""
    device: str = "auto"
    """auto, cpu, or cuda."""

    # Optional overrides (if empty/None, values from checkpoint args are used)
    env_id: str = ""
    backend: str = ""
    obs_mode: str = ""
    architecture: str = ""
    sarfa_five_mode: str = ""
    frameskip: int | None = None
    buffer_window_size: int | None = None
    modifs: str = ""
    new_rf: str = ""

    capture_video: bool = False
    run_name: str = ""


def _resolve_checkpoint_path(raw: str) -> str:
    raw = raw.strip()
    # Convenience for accidental artifact-like suffix such as "foo.cleanrl:model".
    if raw.endswith(":model"):
        raw = raw[:-6] + "_model"
    return str(Path(raw))


def _sanitize_path_component(value: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("._")
    return cleaned or "unknown_game"


def _extract_game_name(env_id: str) -> str:
    # Typical formats: ALE/Amidar-v5, AmidarNoFrameskip-v4, Pong-v5
    base = (env_id or "").split("/")[-1]
    if "-v" in base:
        base = base.split("-v", 1)[0]
    if "NoFrameskip" in base:
        base = base.split("NoFrameskip", 1)[0]
    return _sanitize_path_component(base)


def _extract_modifs_name(modifs: str) -> str:
    # Keep ordering so folder names stay consistent with CLI input.
    parts = [p for p in modifs.split(" ") if p.strip()]
    if not parts:
        return "none"
    return _sanitize_path_component("__".join(parts))


def _parse_modifs(modifs_raw: str) -> list[str]:
    return [m for m in str(modifs_raw).split(" ") if m]


def _available_hackatari_modifs(env_id: str) -> set[str]:
    game_name = _extract_game_name(env_id)
    try:
        modif_module = importlib.import_module(f"hackatari.games.{game_name.lower()}")
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"Could not resolve HackAtari game module for env_id '{env_id}' "
            f"(expected hackatari.games.{game_name.lower()})."
        ) from exc

    game_mods = getattr(modif_module, "GameModifications", None)
    if game_mods is None:
        return set()

    return {
        name
        for name, value in vars(game_mods).items()
        if callable(value) and not name.startswith("_")
    }


def _validate_eval_modifs(cfg: train_mod.Args) -> None:
    requested_modifs = _parse_modifs(getattr(cfg, "modifs", ""))
    if not requested_modifs:
        return

    if cfg.backend != "HackAtari":
        raise ValueError(
            f"--modifs was provided ({requested_modifs}) but backend is '{cfg.backend}'. "
            "Modifications are only supported with backend='HackAtari'."
        )

    available = _available_hackatari_modifs(cfg.env_id)
    unknown = sorted(set(requested_modifs) - available)
    if unknown:
        available_list = ", ".join(sorted(available)) if available else "<none>"
        raise ValueError(
            f"Unknown HackAtari modification(s) for env_id '{cfg.env_id}': {', '.join(unknown)}. "
            f"Available modifications: {available_list}"
        )


def _pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_checkpoint(path: str, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_weights" in checkpoint:
        return checkpoint["model_weights"], dict(checkpoint.get("args", {}))

    if isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint, {}

    raise ValueError(
        "Unsupported checkpoint format. Expected dict with 'model_weights' or plain state_dict."
    )


def _infer_input_channels_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int | None:
    conv0 = state_dict.get("network.0.weight")
    if conv0 is None or getattr(conv0, "ndim", 0) != 4:
        return None
    return int(conv0.shape[1])


def _canonicalize_masked_wrapper(masked_wrapper: Any, state_dict: dict[str, torch.Tensor] | None = None) -> Any:
    if not isinstance(masked_wrapper, str) or not masked_wrapper.strip():
        return masked_wrapper

    key = masked_wrapper.strip()
    # Backward-compatibility: older checkpoints used a generic SARFA dual key.
    if key == "masked_dqn_sarfa_dual":
        in_channels = _infer_input_channels_from_state_dict(state_dict or {})
        if in_channels == 8:
            return "masked_dqn_sarfa_dual_eight"
        return "masked_dqn_sarfa_dual_five"

    return key


def _merge_args(eval_args: EvalArgs, ckpt_args: dict[str, Any]) -> train_mod.Args:
    cfg = train_mod.Args()

    for key, value in ckpt_args.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    if eval_args.env_id:
        cfg.env_id = eval_args.env_id
    if eval_args.backend:
        cfg.backend = cast(Any, eval_args.backend)
    if eval_args.obs_mode:
        cfg.obs_mode = cast(Any, eval_args.obs_mode)
    if eval_args.architecture:
        cfg.architecture = eval_args.architecture
    if eval_args.sarfa_five_mode:
        cfg.sarfa_five_mode = cast(Any, eval_args.sarfa_five_mode)
    if eval_args.frameskip is not None:
        cfg.frameskip = eval_args.frameskip
    if eval_args.buffer_window_size is not None:
        cfg.buffer_window_size = eval_args.buffer_window_size
    if eval_args.modifs:
        cfg.modifs = eval_args.modifs
    if eval_args.new_rf:
        cfg.new_rf = eval_args.new_rf

    cfg.seed = eval_args.seed
    cfg.track = False
    cfg.capture_video = eval_args.capture_video

    # Match training logic for masked wrapper flags.
    # Keep checkpoint values (masked_wrapper/add_pixels) unless obs_mode is explicitly overridden.
    if eval_args.obs_mode:
        cfg.masked_wrapper = cast(Any, None)
        cfg.add_pixels = False

    if "masked" in cfg.obs_mode:
        if cfg.obs_mode.endswith("+pixels"):
            cfg.masked_wrapper = cfg.obs_mode[:-7]
            cfg.add_pixels = True
        else:
            cfg.masked_wrapper = cfg.obs_mode
            if not hasattr(cfg, "add_pixels"):
                cfg.add_pixels = False
        cfg.obs_mode = cast(Any, "ori")


    return cfg


def _infer_sarfa_five_mode_from_checkpoint_name(checkpoint_path: str) -> str | None:
    name = Path(checkpoint_path).stem.lower()
    if "x4" in name:
        return "X4"
    if "all" in name:
        return "All"
    return None


def _build_agent(cfg: train_mod.Args, envs: DummyVecEnv, device: torch.device) -> torch.nn.Module:
    PPODefault, PPObj = _import_ppo_architectures()

    if cfg.architecture == "PPO":
        return PPODefault(envs, device).to(device)

    if cfg.architecture == "PPO_OBJ":
        encoder_dims = tuple(cfg.encoder_dims)
        decoder_dims = tuple(cfg.decoder_dims)
        return PPObj(envs, device, encoder_dims, decoder_dims).to(device)

    raise NotImplementedError(
        f"Only PPO/PPO_OBJ are supported by Sarfa_eval.py right now, got {cfg.architecture}."
    )


def _inject_agent_into_sarfa_wrappers(cfg: train_mod.Args, envs: DummyVecEnv, agent: torch.nn.Module) -> None:
    sarfa_modes = {
        "masked_dqn_sarfa_saliency",
        "masked_dqn_sarfa_dual_five",
        "masked_dqn_sarfa_dual_eight",
    }
    if getattr(cfg, "masked_wrapper", None) not in sarfa_modes:
        return

    wrappers_mod = getattr(train_mod, "ocatari_wrappers", None)
    if wrappers_mod is None:
        raise RuntimeError("Could not access ocatari_wrappers from ppo_atari_oc for SARFA injection.")

    sarfa_wrapper_types = (
        wrappers_mod.SarfaSaliencyWrapper,
        wrappers_mod.SarfaDualWrapperFive,
        wrappers_mod.SarfaDualWrapperEight,
    )

    for env_idx, env in enumerate(envs.envs):
        current_wrapper = env
        found = False
        while hasattr(current_wrapper, "env"):
            if isinstance(current_wrapper, sarfa_wrapper_types):
                current_wrapper.set_model(agent)
                found = True
                break
            current_wrapper = current_wrapper.env
        if not found:
            print(f"WARNING: SARFA wrapper not found in eval env {env_idx}")


def _assert_expected_mask_wrapper(cfg: train_mod.Args, envs: DummyVecEnv) -> None:
    wrapper_key = getattr(cfg, "masked_wrapper", None)
    if not wrapper_key:
        return

    wrappers_mod = getattr(train_mod, "ocatari_wrappers", None)
    if wrappers_mod is None:
        raise RuntimeError("Could not access ocatari_wrappers from ppo_atari_oc.")

    wrapper_class_names = {
        "masked_dqn_bin": "BinaryMaskWrapper",
        "masked_dqn_pixels": "PixelMaskWrapper",
        "masked_dqn_grayscale": "ObjectTypeMaskWrapper",
        "masked_dqn_planes": "ObjectTypeMaskPlanesWrapper",
        "masked_dqn_parallelplanes": "BigPlaneWrapper",
        "masked_dqn_pixel_planes": "PixelMaskPlanesWrapper",
        "masked_dqn_sarfa_saliency": "SarfaSaliencyWrapper",
        "masked_dqn_sarfa_dual_five": "SarfaDualWrapperFive",
        "masked_dqn_sarfa_dual_eight": "SarfaDualWrapperEight",
    }

    class_name = wrapper_class_names.get(wrapper_key)
    if class_name is None:
        return

    expected_wrapper_type = getattr(wrappers_mod, class_name, None)
    if expected_wrapper_type is None:
        raise RuntimeError(f"Expected wrapper class not found: {class_name}")

    for env_idx, env in enumerate(envs.envs):
        current_wrapper = env
        found = False
        while hasattr(current_wrapper, "env"):
            if isinstance(current_wrapper, expected_wrapper_type):
                found = True
                break
            current_wrapper = current_wrapper.env
        if not found:
            raise RuntimeError(
                f"Expected wrapper '{class_name}' for masked_wrapper='{wrapper_key}' not found in eval env {env_idx}."
            )


def evaluate(eval_args: EvalArgs) -> dict[str, Any]:
    checkpoint_path = _resolve_checkpoint_path(eval_args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = _pick_device(eval_args.device)
    state_dict, ckpt_args = _load_checkpoint(checkpoint_path, device)

    cfg = _merge_args(eval_args, ckpt_args)

    # detect X4 and All from checkpoint name
    if (
        getattr(cfg, "masked_wrapper", None) == "masked_dqn_sarfa_dual_five"
        and not eval_args.sarfa_five_mode
        and "sarfa_five_mode" not in ckpt_args
    ):
        inferred_mode = _infer_sarfa_five_mode_from_checkpoint_name(checkpoint_path)
        if inferred_mode is not None:
            cfg.sarfa_five_mode = cast(Any, inferred_mode)

    cfg.masked_wrapper = cast(Any, _canonicalize_masked_wrapper(getattr(cfg, "masked_wrapper", None), state_dict))
    _validate_eval_modifs(cfg)
    train_mod.args = cfg
    train_mod.seed_everything(cfg.seed, cuda=(device.type == "cuda"), torch_deterministic=cfg.torch_deterministic)

    run_name = eval_args.run_name or f"eval_{Path(checkpoint_path).stem}_{int(time.time())}"
    run_dir = str(Path("runs") / run_name)

    envs = DummyVecEnv(
        [
            train_mod.make_env(
                cfg.env_id,
                i,
                cfg.capture_video,
                run_dir,
                seed=cfg.seed + i,
                agent=None,
                evaluating=True
            )
            for i in range(cfg.num_envs)
        ]
    )
    _assert_expected_mask_wrapper(cfg, envs)

    agent = _build_agent(cfg, envs, device)
    agent.load_state_dict(state_dict, strict=True)
    agent.eval()
    _inject_agent_into_sarfa_wrappers(cfg, envs, agent)

    obs = envs.reset()

    returns = np.zeros(cfg.num_envs, dtype=np.float64)
    lengths = np.zeros(cfg.num_envs, dtype=np.int64)
    episodic_returns: list[float] = []
    episodic_lengths: list[int] = []

    with torch.no_grad():
        while len(episodic_returns) < eval_args.episodes:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            actions = agent.get_action_and_value(obs_tensor)[0]
            obs, rewards, dones, _infos = envs.step(actions.cpu().numpy())

            rewards = np.asarray(rewards, dtype=np.float64)
            dones = np.asarray(dones, dtype=bool)

            returns += rewards
            lengths += 1

            done_idxs = np.where(dones)[0]
            for idx in done_idxs:
                episodic_returns.append(float(returns[idx]))
                episodic_lengths.append(int(lengths[idx]))
                returns[idx] = 0.0
                lengths[idx] = 0

    envs.close()

    episodic_returns = episodic_returns[: eval_args.episodes]
    episodic_lengths = episodic_lengths[: eval_args.episodes]

    result = {
        "checkpoint": checkpoint_path,
        "episodes": int(eval_args.episodes),
        "device": str(device),
        "env": {
            "env_id": cfg.env_id,
            "backend": cfg.backend,
            "obs_mode": cfg.obs_mode,
            "masked_wrapper": cfg.masked_wrapper,
            "sarfa_five_mode": getattr(cfg, "sarfa_five_mode", "X4"),
            "architecture": cfg.architecture,
            "num_envs": cfg.num_envs,
        },
        "returns": episodic_returns,
        "lengths": episodic_lengths,
        "return_mean": float(np.mean(episodic_returns)),
        "return_std": float(np.std(episodic_returns)),
        "return_min": float(np.min(episodic_returns)),
        "return_max": float(np.max(episodic_returns)),
        "length_mean": float(np.mean(episodic_lengths)),
        "seed": cfg.seed,
        "eval_args": asdict(eval_args),
    }

    output_json = eval_args.output_json.strip()
    if not output_json:
        game_name = _extract_game_name(cfg.env_id)
        if str(getattr(cfg, "modifs", "")).strip():
            modifs_name = _extract_modifs_name(str(cfg.modifs))
            output_json = str(
                Path("evaluation")
                / "modifications"
                / game_name
                / modifs_name
                / f"{Path(checkpoint_path).stem}.eval.json"
            )
        else:
            output_json = str(Path("evaluation") / game_name / f"{Path(checkpoint_path).stem}.eval.json")
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    result["output_json"] = output_json

    return result


def main() -> None:
    eval_args = tyro.cli(EvalArgs)
    result = evaluate(eval_args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()





