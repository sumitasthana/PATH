"""Prepare the latest main checkout in a fresh Colab session, before imports."""

import importlib
import os
from pathlib import Path
import subprocess
import sys


REMOTE = "https://github.com/sumitasthana/PATH.git"
DEFAULT_REPO = "/content/uncle-diagnostics"


def setup_repo(repo=DEFAULT_REPO, remote=REMOTE, branch="main", *, install=True):
    """Clone or fast-forward a clean checkout and keep Colab's GPU packages.

    Run once before importing uncle. Updating an already imported package can
    leave old Python classes running against new files, so restart to update it.
    Existing local edits, other branches, and local commits are left alone.
    """
    if any(name == "uncle" or name.startswith("uncle.") for name in sys.modules):
        raise RuntimeError(
            "Setup runs before importing uncle. Restart the runtime to update "
            "the code, or continue with the remaining cells in this session."
        )
    repo = Path(repo).expanduser().resolve()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True
        ).strip()

    if repo.exists():
        if not (repo / ".git").exists():
            raise RuntimeError(f"Not a Git checkout: {repo}")
        if git("remote", "get-url", "origin").removesuffix(".git") != remote.removesuffix(".git"):
            raise RuntimeError(f"Different repository at {repo}")
        if git("status", "--porcelain"):
            raise RuntimeError(f"Local changes in {repo}; preserve them before updating.")
        if git("branch", "--show-current") != branch:
            raise RuntimeError(f"{repo} is not on {branch}; use a fresh checkout.")
        subprocess.run(["git", "-C", str(repo), "fetch", "origin", branch], check=True)
        result = subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", "HEAD", "FETCH_HEAD"],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("Local commits differ from the remote; use a fresh checkout.")
        subprocess.run(
            ["git", "-C", str(repo), "merge", "--ff-only", "FETCH_HEAD"], check=True
        )
    else:
        subprocess.run(
            ["git", "clone", "--branch", branch, remote, str(repo)], check=True
        )

    if install:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(repo)],
            check=True,
        )
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    importlib.invalidate_caches()
    print("Repository:", repo)
    print("Commit:", git("rev-parse", "HEAD"))
    return repo


if __name__ == "__main__":
    setup_repo()
