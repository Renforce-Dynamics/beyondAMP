import os
import sys
import argparse
import subprocess
from pathlib import Path


def get_isaaclab_path() -> Path:
    """Safely import isaaclab and return its root directory."""
    import importlib

    isaaclab = importlib.import_module("isaaclab")
    isaaclab_file = Path(isaaclab.__file__).resolve()
    # Typically: /.../site-packages/isaaclab/__init__.py
    return isaaclab_file.parent.parent.parent


def create_symlink(target: Path, link_dir: Path):
    """Create a symbolic link inside link_dir pointing to target."""
    link_path = link_dir / target.name

    if link_path.exists() or link_path.is_symlink():
        print(f"[SKIPPED] {link_path} already exists.")
        return

    try:
        subprocess.run(["ln", "-s", str(target), str(link_path)], check=True)
        print(f"[LINKED] {target} → {link_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to create link: {e}")


modules = [
    "./source/robotlib",
    "./source/rsl_rl_amp",
    "./source/amp_tasks",
    "./source/beyondMimic",
    "./source/beyondAMP",
]

def main():
    parser = argparse.ArgumentParser(
        description="Create symbolic links for local repos under IsaacLab."
    )
    parser.add_argument(
        "--isaaclab-path",
        type=str,
        default=None,
        help="Absolute path to IsaacLab root. If not provided, will auto-detect.",
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        default=modules,
        help="List of local repositories to link into IsaacLab.",
    )
    args = parser.parse_args()

    # Resolve IsaacLab path
    if args.isaaclab_path:
        isaaclab_path = Path(args.isaaclab_path).expanduser().resolve()
    else:
        isaaclab_path = get_isaaclab_path()
        print(f"[INFO] Auto-detected IsaacLab path: {isaaclab_path}")

    if not isaaclab_path.exists():
        sys.exit(f"[ERROR] IsaacLab path does not exist: {isaaclab_path}")

    # Target directory (usually isaaclab/)
    link_dir = isaaclab_path
    print(f"[INFO] Linking repositories into: {link_dir}")

    # Create symlinks
    script_dir = Path(__file__).parent.parent.resolve()
    for repo_rel in args.repos:
        repo_path = (script_dir / repo_rel).resolve()
        if not repo_path.exists():
            print(f"[WARN] Repository not found: {repo_path}")
            continue
        create_symlink(repo_path, link_dir)

    print("[DONE] All symlinks created successfully.")


if __name__ == "__main__":
    main()