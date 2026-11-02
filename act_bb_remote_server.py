#!/usr/bin/env python3
"""GPU ACT server using the installed LeRobot remote deployment."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset-root", required=True, type=Path,
                   help="Training dataset root containing meta/info.json")
    p.add_argument("--deployment", default="pink_act")
    p.add_argument("--listen", default="tcp/0.0.0.0:7447")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fps", type=float, default=30)
    p.add_argument("--n-action-steps", type=int, default=50)
    p.add_argument("--semantics", default="rby1_left_joint_v1")
    args = p.parse_args()

    from lerobot.remote_inference.configs import ServerConfig, ModelConfig, ExecutionConfig
    from lerobot.transport import ZenohConfig
    from lerobot.rollout.context import _feature_spec
    from lerobot.scripts.lerobot_policy_server import serve

    features = json.loads((args.dataset_root / "meta/info.json").read_text())["features"]

    # 현재 RBY1 플러그인의 position feature 이름과 맞춤
    for key in ("observation.state", "action"):
        features[key]["names"] = [
            name if name.endswith(".pos") else name + ".pos"
            for name in features[key]["names"]
        ]

    # Dataset metadata supplies the actual motor names/order; never guess .pos suffixes.
    observation = [
        _feature_spec("observation.state", features["observation.state"], "joint_position"),
        _feature_spec("observation.images.left", features["observation.images.left"], "rgb"),
    ]
    cfg = ServerConfig(
        deployment=args.deployment,
        model=ModelConfig(repo_or_path=args.checkpoint, device=args.device),
        execution=ExecutionConfig(action_fps=args.fps, n_action_steps=args.n_action_steps),
        zenoh=ZenohConfig(mode="peer", listen_endpoints=[args.listen]),
        semantics=args.semantics, robot_type="rby1",
        features=observation,
        action_feature=_feature_spec("action", features["action"], "joint_position"),
    )
    serve(cfg)


if __name__ == "__main__":
    main()
