"""Git plumbing for the private raw-data repository.

``main`` holds append-only observation files and the account pool. Each job also owns
one state branch that always has exactly one commit and is force-pushed, so large
tables that change every run do not pile up in history.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import List

URL = os.environ.get("RAW_REPO_URL", "git@github.com:fabkury/servoom-raw.git")
WORK = Path(os.environ.get("RAW_WORKDIR", "_raw")).resolve()


def git(*args: str, cwd: Path, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])} failed: {r.stderr.strip()[-400:]}")
    return r.stdout.strip()


def _rmtree(path: Path) -> None:
    def on_error(fn, p, exc):            # git marks object files read-only on Windows
        os.chmod(p, stat.S_IWRITE)
        fn(p)
    shutil.rmtree(path, onerror=on_error)


def _identity(path: Path) -> None:
    git("config", "user.name", "servoom-stats", cwd=path)
    git("config", "user.email", "servoom-stats@users.noreply.github.com", cwd=path)


def clone_main(sparse: List[str]) -> Path:
    """Blobless, shallow, sparse clone of ``main``: only ``sparse`` dirs are fetched."""
    path = WORK / "main"
    if path.exists():
        _rmtree(path)
    WORK.mkdir(parents=True, exist_ok=True)
    git("clone", "--filter=blob:none", "--depth", "1", "--no-checkout", "--branch", "main", URL, str(path), cwd=WORK)
    git("sparse-checkout", "init", "--cone", cwd=path)
    git("sparse-checkout", "set", *sparse, cwd=path)
    git("checkout", "main", cwd=path)
    _identity(path)
    return path


def push_main(message: str) -> None:
    path = WORK / "main"
    git("add", "--sparse", "-A", cwd=path)
    if not git("status", "--porcelain", cwd=path):
        return
    git("commit", "-q", "-m", message, cwd=path)
    for _ in range(5):
        r = subprocess.run(["git", "push", "-q", "origin", "main"], cwd=path, capture_output=True, text=True)
        if r.returncode == 0:
            return
        git("fetch", "-q", "--depth", "50", "origin", "main", cwd=path)
        # A shallow clone whose base has fallen more than 50 commits behind has no merge
        # base with origin/main; rebasing then replays the grafted root commit (the whole
        # tree) and conflicts with every file changed upstream. This happened on
        # 2026-10-06 when the like backfill committed every few minutes during a snapshot.
        # Deepening to the full history is cheap: the clone is blobless.
        if not git("merge-base", "HEAD", "origin/main", cwd=path, check=False):
            if git("rev-parse", "--is-shallow-repository", cwd=path) == "true":
                git("fetch", "-q", "--unshallow", "origin", "main", cwd=path)
        git("rebase", "origin/main", cwd=path)
    raise RuntimeError("could not push raw main")


def clone_state(branch: str) -> Path:
    path = WORK / branch
    if path.exists():
        _rmtree(path)
    WORK.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["git", "clone", "-q", "--depth", "1", "--single-branch", "--branch", branch, URL, str(path)],
                       cwd=WORK, capture_output=True, text=True)
    if r.returncode != 0:                       # first run: the branch does not exist yet
        path.mkdir(parents=True)
        git("init", "-q", cwd=path)
        git("remote", "add", "origin", URL, cwd=path)
    _identity(path)
    return path


def push_state(branch: str, message: str) -> None:
    path = WORK / branch
    git("checkout", "-q", "--orphan", "_new", cwd=path)
    git("add", "-A", cwd=path)
    git("commit", "-q", "-m", message, cwd=path)
    git("branch", "-M", branch, cwd=path)
    git("push", "-q", "--force", "origin", branch, cwd=path)
