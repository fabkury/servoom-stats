#!/usr/bin/env bash
# Turn notes left by the pipeline into GitHub issues, one per title, without duplicates.
set -uo pipefail
shopt -s nullglob
for f in _issues/*.md; do
  title="$(head -n 1 "$f")"
  if [ -z "$(gh issue list --state open --search "in:title \"$title\"" --json number --jq '.[0].number')" ]; then
    tail -n +3 "$f" | gh issue create --title "$title" --body-file - || true
  fi
done
