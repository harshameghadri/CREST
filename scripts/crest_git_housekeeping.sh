#!/usr/bin/env bash
# CREST repository housekeeping - run on your own machine.
#
#   bash crest_git_housekeeping.sh            # dry run: prints every command, changes nothing
#   bash crest_git_housekeeping.sh --apply    # performs the steps, asking before each one
#   bash crest_git_housekeeping.sh --apply --yes   # no per-step questions
#
# What it does (each step is skipped if already done):
#   1. clone (or reuse) harshameghadri/CREST and fetch everything
#   2. make sure `dev` exists and starts from `main`
#   3. archive the stale branches as tags `archive/<branch>` and delete the branches
#   4. list other remote branches already merged into main (for you to review)
#   5. optionally open a PR from the current Claude work branch into `dev` (needs `gh`)
#
# Needs: git with push access (SSH key or credential helper). `gh` is optional.

set -euo pipefail

REPO_URL="${REPO_URL:-git@github.com:harshameghadri/CREST.git}"
WORKDIR="${WORKDIR:-$HOME/CREST}"
STALE_BRANCHES=(feat/leiden-hnsw-faer fix/production-readiness)
WORK_BRANCH="${WORK_BRANCH:-claude/awesome-lamport-dsvfv2}"

APPLY=0
YES=0
for a in "$@"; do
  case "$a" in
    --apply) APPLY=1 ;;
    --yes) YES=1 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done

bold() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
run() {
  # run "description" cmd args...
  local desc="$1"; shift
  printf '  $ %s\n' "$*"
  if [[ $APPLY -eq 0 ]]; then return 0; fi
  if [[ $YES -eq 0 ]]; then
    read -r -p "  -> $desc? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || { echo "  skipped"; return 0; }
  fi
  "$@"
}

[[ $APPLY -eq 0 ]] && echo "DRY RUN - nothing will be changed. Re-run with --apply to perform the steps."

bold "1. Repository"
if [[ -d "$WORKDIR/.git" ]]; then
  echo "  using $WORKDIR"
else
  run "clone $REPO_URL" git clone "$REPO_URL" "$WORKDIR"
fi
if [[ -d "$WORKDIR/.git" ]]; then
  cd "$WORKDIR"
  git fetch --prune --tags origin
else
  echo "  (dry run: no clone yet, remaining commands shown only)"
fi

remote_has() { git ls-remote --exit-code --heads origin "$1" >/dev/null 2>&1; }
tag_has() { git ls-remote --exit-code --tags origin "refs/tags/$1" >/dev/null 2>&1; }

bold "2. dev branch"
if [[ -d .git ]] && remote_has dev; then
  ahead=$(git rev-list --count origin/main..origin/dev)
  behind=$(git rev-list --count origin/dev..origin/main)
  echo "  origin/dev exists ($ahead ahead, $behind behind main)"
  if [[ "$behind" -gt 0 && "$ahead" -eq 0 ]]; then
    run "fast-forward dev to main" git push origin origin/main:refs/heads/dev
  fi
else
  run "create dev from main" git push origin origin/main:refs/heads/dev
fi

bold "3. Archive and delete stale branches"
for b in "${STALE_BRANCHES[@]}"; do
  tag="archive/${b//\//-}"
  if [[ -d .git ]] && ! remote_has "$b"; then
    echo "  $b: already gone"
    continue
  fi
  if [[ -d .git ]] && tag_has "$tag"; then
    echo "  $tag already exists"
  else
    run "tag $b as $tag" git push origin "refs/remotes/origin/$b:refs/tags/$tag"
  fi
  run "delete remote branch $b" git push origin --delete "$b"
done

bold "4. Other remote branches already merged into main (review; not deleted automatically)"
if [[ -d .git ]]; then
  git branch -r --merged origin/main | sed 's/^ *//' | grep -v -E '^origin/(main|dev|HEAD)' || echo "  none"
fi

bold "5. Pull request $WORK_BRANCH -> dev"
if command -v gh >/dev/null && [[ -d .git ]] && remote_has "$WORK_BRANCH"; then
  if [[ $(git rev-list --count "origin/dev..origin/$WORK_BRANCH" 2>/dev/null || echo 0) -gt 0 ]]; then
    run "open PR $WORK_BRANCH -> dev" gh pr create --repo harshameghadri/CREST --base dev --head "$WORK_BRANCH" --fill
  else
    echo "  nothing to merge into dev"
  fi
else
  echo "  skipped (needs gh, or the branch has no new commits)"
fi

bold "Done"
[[ -d .git ]] && git ls-remote --heads origin | awk '{print "  " $2}'
