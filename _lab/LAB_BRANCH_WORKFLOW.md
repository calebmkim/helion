# Lab branch workflow — `reduction-pr-with-lab`

> Read this before touching the `reduction-pr-with-lab` branch or the `_lab/` directory.
> It explains how the lab branch relates to the PR and how to keep them in sync **without
> ever leaking `_lab/` into the PR**.

## What this branch is

`reduction-pr-with-lab` = **the latest reduction-heuristics PR code** + **one commit on top
that adds `_lab/`** (this harness, the recorder, oracle caches, ledger, hub/worker logs,
bench scripts).

It exists so a fresh checkout on **any server** sees the latest PR code AND the lab history
together. **It is NOT a PR** and must never be opened as one.

```
  ... PR commits ...
        │
        ▼
   <PR head>                 ← the real PR tip (see "Branch map" below)
        │
        ▼
   <lab commit>              ← tip of reduction-pr-with-lab; ADDS _lab/ ONLY
```

The lab commit touches **only** files under `_lab/`. The PR never touches `_lab/`. Because
their paths never overlap, the lab commit replays onto a newer PR head with zero conflicts,
and the PR branch never sees `_lab/`.

## Branch map (names matter — they differ)

| Role | Branch name | Where |
|---|---|---|
| The PR (what reviewers see) | `reduction-seed-heuristic-run2` | remote `origin` (= the fork) + GitHub PR #2704 |
| Local worktree holding the PR | `pr-2704-review-fixes` | local worktree (`helion/`) — its tip == the PR head |
| **This lab branch** | `reduction-pr-with-lab` | remote `origin` + local worktree (`helion-pr-with-lab/`) |

> ⚠️ The local PR branch (`pr-2704-review-fixes`) and the remote PR ref
> (`reduction-seed-heuristic-run2`) have **different names**. When syncing the lab forward,
> rebase onto **whatever local branch currently holds the latest PR head** — verify with
> `git log -1` before rebasing, don't assume the name.

## THE ONE INVARIANT (do not break)

**`_lab/` must never appear in the PR or the PR branch.**

Concretely:
- **Only ever push this branch to its own ref:** `reduction-pr-with-lab:reduction-pr-with-lab`.
- **NEVER** push it onto `reduction-seed-heuristic-run2` (or any PR ref). One wrong refspec
  publishes `_lab/` into the PR.
- The PR is pushed only from the PR worktree, never from here.

## Fresh checkout on a new server

```bash
git clone <fork-url> helion
cd helion
git fetch origin
git checkout reduction-pr-with-lab     # gets latest-known PR code + _lab/
```

Now you have the PR code and the full lab. Note: harness scripts under `_lab/` may contain
hardcoded `/home/dev/local/helion...` paths from the original machine — **they are a portable
RECORD, not portable-executable**. Before running a script on a new server, fix its paths
(prefer a `PYTHONPATH=<this-worktree>` override; never `pip install`).

## Keeping the lab in sync after the PR advances

When new commits land on the PR, replay the single lab commit on top of the new PR head:

```bash
cd <lab worktree>                         # e.g. helion-pr-with-lab/

# 1. Make sure the local PR branch is up to date with the PR head you want to track.
#    (Fetch / fast-forward whatever branch holds the PR — verify its tip first.)
git fetch origin
git log -1 --oneline <local-PR-branch>    # confirm this is the PR head you intend

# 2. Replay the lab commit onto that PR head. Zero conflicts (paths never overlap).
git rebase <local-PR-branch>              # e.g. git rebase pr-2704-review-fixes

# 3. Force-push (rebase rewrote the lab commit's base). force-with-lease = safe.
git push --force-with-lease origin reduction-pr-with-lab:reduction-pr-with-lab
```

After step 2, sanity-check before pushing:

```bash
git log --oneline -2                      # tip = lab commit, HEAD~1 = the new PR head
git rev-parse HEAD~1                       # MUST equal the PR head you rebased onto
git diff --cached --name-only             # (should be empty; nothing staged)
git show --stat HEAD | grep -v '^ _lab/' | grep -E '^\s+\S+/' && echo "LEAK: non-_lab file in lab commit!" || echo "OK: lab commit touches only _lab/"
```

If the last line prints `LEAK`, **stop** — the lab commit picked up a non-`_lab/` change; fix
it (`git reset`, re-stage only `_lab/`) before pushing.

## Updating `_lab/` contents (new logs, harness edits)

Just edit under `_lab/` and amend (or add) on this branch:

```bash
cd <lab worktree>
git add _lab
git status --short | grep -v '^?? _lab/' | grep -v '^[ AM]  _lab/' && echo "WARNING: staged non-_lab change" || true
git commit --amend --no-edit              # keep it ONE lab commit on top of the PR
git push --force-with-lease origin reduction-pr-with-lab:reduction-pr-with-lab
```

Keeping it to a **single** lab commit on top is what makes the forward-rebase trivial. If you
prefer history, a chain of `_lab/`-only commits also rebases cleanly — just never interleave
PR-code changes into them.

## Hygiene

- `_lab/.gitignore` excludes `*.pyc` / `__pycache__/`. Don't commit build artifacts or
  multi-hundred-MB GPU dumps; the lab is text/JSON logs + scripts (~7 MB).
- Don't commit anything machine-specific that you wouldn't want another server to inherit
  (absolute paths in *runnable* scripts are the main hazard — see the portability note above).

## Quick reference

| Goal | Command |
|---|---|
| Get PR + lab on a new server | `git checkout reduction-pr-with-lab` |
| Sync lab to newer PR | `git rebase <local-PR-branch>` then force-push to **own ref** |
| Update lab contents | edit `_lab/`, `git add _lab`, `git commit --amend`, force-push to **own ref** |
| Push (always) | `git push --force-with-lease origin reduction-pr-with-lab:reduction-pr-with-lab` |
| ❌ Never | push this branch onto `reduction-seed-heuristic-run2` / any PR ref |
