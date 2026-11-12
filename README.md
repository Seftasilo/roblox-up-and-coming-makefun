# Predict Roblox's Up-and-Coming top 10

Entry for make.fun bounty `roblox-up-and-coming-forecast-01` ($500, ends
2026-11-13 06:00 UTC). A Python 3.13 program (standard library only) that
reads the chart history in `/input` and writes `/output/forecast.json`:
for each of the top 10 places, a probability for each game it names being at
that place, plus its top-10 guess in order.

## Method — empirical rank-transition kernel

The Up-and-Coming chart is a fast-moving, mean-reverting diffusion, not a
trending market. Measured on the kit's history (34 archive reads,
2026-10-30–2026-11-07):

- Rank velocity has ~zero persistence: corr(24 h rank change, next 24 h
  rank change) = 0.055. Trend-following on rank does not help.
- Player counts do not predict rank moves either: corr(rank change,
  log player change) = -0.018.
- Top-10 survival: 83% at 12 h, 70% at 24 h, 51% at 48 h, 38% at 72 h.
  The #1 almost never survives 72 h (0/23).
- Entrants to the top 10 at ~48 h come from everywhere: ranks 11–20 (35),
  21–50 (44), 51+ (28), brand new to the chart (26).

So the forecaster (`main.py`) does not chase trends. For every lag in
{6, 12, 24, 36, 48, 60, 72} hours it estimates P(rank at lag h | rank now)
from every pair of snapshots in the history, smoothed with a separable
box blur (a Gaussian approximation) over both the starting rank and the
landing rank, plus an exit mass for games that leave the chart. The forecast
is the uniform average of that kernel over the window hours 0–72 — the
score is the mean over the window's reads, so the time-averaged distribution
is the honest forecast. A game is named at a place only while its
probability beats the unnamed share it would otherwise receive, so the long
tail is left to the unnamed share. `top10` is the greedy per-place argmax
in place order. No network, no wall clock, no randomness: two runs on the
same history write byte-identical output.

Variants tested and rejected (see `tests/tune.py`): rank-velocity drift
(hurts practice badly: -33.37 → -35.19 at shrink 0.2), heavier smoothing,
fewer lags. Smoothing radii (2, 2) and lag tolerance 0.4 were chosen by a
backtest sweep.

## Data sources

- Kit: official make.fun kit for `roblox-up-and-coming-forecast-01`,
  downloaded from `https://make.fun/c/roblox-up-and-coming-forecast-01/kit`;
  all 13 files verified OK against `kit.json` before use.
- History: the kit's `input/history.ndjson` (34 reads, 2026-10-30–2026-11-07).
- Practice-window validation reads: the public MIT-licensed archive
  `github.com/Kenny-Dhub/roblox-snapshots` (`data/discover/latest.json`),
  the data source the kit itself names in `input/NOTICE.txt` and
  `competition.yaml`. The 10 reads from 2026-11-08T00:37Z to
  2026-11-10T17:51Z are the practice window's reads; they were used only to
  validate the local scorer and the forecaster, never shipped in the entry
  (the entry reads only `/input` at grading time).

## Verification (all local, reproducible)

Local scorer (`tests/score.py`) is an exact port of the rules in
`competition.yaml` / `RULES.txt`. It reproduces the published house-baseline
practice score exactly: **-43.512039** (typical 0.012891/place, 6.4 top-10
hits, 53.5 agreement).

| Evaluation | This entry | House baseline |
|---|---|---|
| Practice window (10 reads, 11-07→11-10) | **-33.369** (typical 0.0355, 6.4 hits, 53.5 agr) | -43.512 |
| Walk-forward, 5 cuts (next 72 h each) | **-36.71 mean** | -43.39 mean |

Margin over the house persistence baseline: **+10.14 nats** on practice,
**+6.68** walk-forward mean. A prize also needs beating the house AI model
forecast, whose score is not public.

Official pre-submit check (`makefun check --kit <kit> --repo .`):
4 passed — `competition_open`, `snapshot` (8 files, 54,458 bytes, within
caps), `license_osi` (MIT), `entrypoint` (`main.py` → python3.13).
The only failure is `transcripts_present`: transcripts are bundled by the
submitting agent at submit time (the kit requires the top-level agent to
bundle once its subagents have finished). Repo checks (`repo_public`,
`repo_owner`, `repo_created_after_open`, `commit_in_repo`,
`duplicate_snapshot`) can only pass once the public GitHub repo exists;
`.makefun/owner.json` (wallet `DDyzz3dCMDs5Btda4BPyQCp71nVURJweRFS7qW2VLHH7`)
is included for `repo_owner`.

Determinism: two runs on the practice input write byte-identical
`forecast.json` (23,511 bytes, 458 probability rows). Runtime ~0.3 s on
the practice history (limit 300 s).

## Files

- `main.py` — the entry (Python 3.13, standard library only).
- `LICENSE` — MIT.
- `.makefun/owner.json` — wallet proof for `repo_owner`.
- `tests/score.py` — local scorer (exact port of the rules).
- `tests/tune.py` — kernel forecaster + walk-forward / practice tuning harness.
