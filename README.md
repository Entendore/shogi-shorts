# Shogi Shorts Studio

Generates tsumeshogi / tactics puzzles and renders vertical YouTube-Shorts
MP4s (hook → think timer → animated solution → alt-defense bonus line →
recap → outro) with SFX + BGM, from a PySide6 GUI.

## Run
    source .venv/bin/activate
    python shogi_shorts.py            # GUI
    python shogi_shorts.py --selftest # full self-test

## Highlights
- 26 categories: mate 1-4, corner/edge, double & discovered check, knight /
  pawn / drop / sacrifice / gold-silver-hand mates, Gote-to-mate (vertical
  mirror), find-ALL-mates, mate race, fork, skewer, pin, promotion,
  promotion-choice, quiet-key best move, parry (find the only defense),
  discovered attack, win-a-piece
- Hybrid solver: early-exit AND/OR (n<=2) + df-pn with shared Zobrist TT
  (n>=3), pluggable evasion ordering, cross-solver consensus validation
- Rule-legal positions: 40-piece inventory, forced-promotion squares, nifu,
  king placement; strict 40-piece fill option
- Repair pipeline: dual-repair, defender strengthening, greedy + simulated-
  annealing minimization, near-miss evolutionary rescue
- Construction: retrograde (mate-first) & constraint-ordered backtracking
  clusters, optional CP-SAT proposer (pip install ortools), template mining,
  UCB1 knob bandit, MAP-Elites niche variety
- Persistent manifest.json: mirror-canonical dedupe, key-pattern blocking,
  niche archive — no repeated puzzles across sessions
- Property-tested movegen (mirror law, naive differential, attack symmetry)
  plus perft(3) = 25470 in the selftest
- Export: MP4 (faststart, crf19) + thumbnail + metadata txt + pack.json;
  synthesized SFX and lo-fi BGM muxed via bundled ffmpeg

## Linux GUI deps (Debian/Ubuntu)
    sudo apt install -y libgl1 libegl1 libxkbcommon0 libglib2.0-0
