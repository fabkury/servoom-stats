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
  git pull -q --rebase origin main
done
exit 1
