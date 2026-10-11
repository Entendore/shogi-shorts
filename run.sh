#!/usr/bin/env bash
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate
exec python shogi_shorts.py "$@"
