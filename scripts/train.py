"""Training launcher.

Single process:  python scripts/train.py --config configs/iris3b/stage1_256.yaml
Multi GPU:       torchrun --nproc_per_node=8 scripts/train.py --config configs/iris3b/stage1_256.yaml \
                     train.expected_global_batch=128
Multi node:      scripts/launch_multinode.sh -- one rendezvous spanning every node.
                 Never start a per-node torchrun by hand: that builds one world
                 per node, each with its own rank 0.

Trailing ``key=value`` pairs are dotlist overrides merged onto the YAML config.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iris3b.config import Config, load_config  # noqa: E402
from iris3b.train import Trainer  # noqa: E402
from iris3b.train.world import World, assert_single_world, describe_world, log_world  # noqa: E402


def prestage(cfg: Config, world: World) -> None:
    """Fill this node's lazily-populated caches serially, then exit.

    Both caches are node-local. On node 0 the ranks race harmlessly because
    something already warmed them; on every other node all local ranks start
    cold at the same instant, each paying a full metadata walk. Runs under the
    same environment contract as training, so a bad topology fails here first.
    """
    if cfg.repa.weight > 0:
        from iris3b.repa import REPALoss

        teacher = REPALoss(cfg.repa, student_dim=cfg.model.hidden_size)
        print(f"[prestage] repa teacher {cfg.repa.teacher} available={teacher.available}", flush=True)

    from iris3b.data import datasets as _datasets  # noqa: F401  (registers dataset types)
    from iris3b.data.builder import shape_policy_id
    from iris3b.data.samplers import RangedSampler, ShapeBatchSampler, key_cache_path
    from iris3b.registry import DATASETS

    dataset = DATASETS.build(cfg.data.type, cfg.data, cfg.model.patch_size)
    if dataset.policy.uniform or not cfg.data.data_dirs:
        print("[prestage] one shape per batch by construction; no plan cache", flush=True)
        return
    policy_id = shape_policy_id(cfg.data, cfg.model.patch_size)
    for local_rank in range(world.local_world_size):
        rank = world.node_rank * world.local_world_size + local_rank
        path = key_cache_path(cfg.data.data_dirs, world.world_size, rank, policy_id)
        sampler = RangedSampler(len(dataset), rank=rank, world_size=world.world_size)
        plan = ShapeBatchSampler(
            sampler, dataset, cfg.train.batch_size, drop_last=True, cache_path=path
        ).plan()
        print(f"[prestage] rank {rank}: {len(plan)} batches -> {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pixel-space rectified-flow DiT")
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument(
        "--expected-nodes",
        type=int,
        default=None,
        help="refuse to start unless the world spans this many nodes",
    )
    parser.add_argument("--prestage", action="store_true", help="fill node-local caches and exit")
    parser.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. train.batch_size=8")
    args = parser.parse_args()
    world = describe_world()
    log_world(world)
    assert_single_world(world, args.expected_nodes)
    cfg = load_config(args.config, args.overrides)
    if world.num_nodes > 1 and not cfg.train.perf.lazy_sync:
        raise ValueError(
            "multi-node runs require train.perf.lazy_sync=true. Without it the non-finite guard "
            "runs rank-locally on the loss BEFORE backward, so a rank that sees a NaN skips its "
            "backward while every other rank blocks in the gradient all-reduce and the job hangs "
            "with no error. lazy_sync moves the guard onto the clipped grad norm, which is "
            "identical on every rank, making the skip collective."
        )
    if args.prestage:
        prestage(cfg, world)
        return
    Trainer(cfg).run()


if __name__ == "__main__":
    main()
