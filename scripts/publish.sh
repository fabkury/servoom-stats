#!/usr/bin/env bash
# Commit data/ if it changed, retrying on a concurrent push from the other job.
set -euo pipefail
git config user.name "servoom-stats"
git config user.email "servoom-stats@users.noreply.github.com"
git add -A data
if git diff --cached --quiet; then
  echo "changed=false" >> "$GITHUB_OUTPUT"; exit 0
fi
git commit -q -m "data: $1 $(date -u +'%Y-%m-%d %H:%M')"
for i in 1 2 3 4 5; do
  if git push -q origin HEAD:main; then echo "changed=true" >> "$GITHUB_OUTPUT"; exit 0; fi
  git fetch -q origin main
  if ! git rebase -q origin/main; then
    # Another job published meanwhile and both rewrote whole JSON files. The commit being
    # replayed is this job's own, newer and whole, so its version of every conflicted file
    # wins ("theirs" during a rebase); files only the other job touched merge cleanly.
    git diff --name-only --diff-filter=U | xargs -r git checkout --theirs --
    git add -A data
    GIT_EDITOR=true git rebase --continue
  fi
done
exit 1
