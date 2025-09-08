"""
anyBrainer – CLI entry‑point

Run any workflow registered in the package from the command line; e.g.,:

    $ anyBrainer TrainWorkflow configs/pretrain.yaml

Config file must be YAML or JSON and at minimum contain the keys
expected by the workflow constructor (e.g. ``global_settings``,
``pl_datamodule_settings`` …).

Each workflow class is looked‑up in the ``WORKFLOW`` registry by the
name provided as the first positional argument.
"""

import argparse
from pathlib import Path
import sys

from anyBrainer.core.utils import load_config
from anyBrainer.registry import get, RegistryKind as RK


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="anyBrainer",
        description="Run anyBrainer workflows from the CLI",
    )
    parser.add_argument(
        "workflow",
        help="Registered workflow name (e.g. 'train_workflow')",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML or JSON configuration file",
    )

    args = parser.parse_args(argv)

    WorkflowCls = get(RK.WORKFLOW, args.workflow)
    cfg = load_config(args.config)

    if not isinstance(cfg, dict):
        sys.exit("Top‑level config must be a mapping/dictionary.")

    try:
        workflow = WorkflowCls(**cfg)
    except TypeError as exc:
        sys.exit(f"Failed to construct workflow: {exc}")

    if callable(workflow):
        workflow()
    elif hasattr(workflow, "run"):
        workflow.run()
    else:
        sys.exit("Workflow doesn’t implement __call__() or run().")


if __name__ == "__main__":
    main()
