#!/usr/bin/env python3
"""Shogi Shorts Studio — tsumeshogi/tactics generator -> vertical MP4s.
26 categories, hybrid AND/OR + df-pn solver with shared Zobrist TT,
rule-legal positions, repair pipeline (dual-repair, defender strengthening,
greedy+SA minimization, near-miss rescue), brute-force + consensus validation,
mirror/key-pattern dedupe with persistent manifest, difficulty targeting,
retrograde/constructive/CP-SAT construction, template mining, UCB1 knobs,
MAP-Elites variety, hook/alt-defense/recap timeline, SFX+BGM, packs."""
from __future__ import annotations
import json, math, os, random, subprocess, sys, time, wave
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, QSettings, QThread, QTimer, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QFontDatabase, QFontMetrics,
                           QImage, QGuiApplication, QLinearGradient, QPainter,
                           QPainterPath, QPen, QPixmap, QRadialGradient)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                               QDoubleSpinBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QMainWindow, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

# ============================================================================
# 1. ENGINE   (row 0 = top / Gote home; Sente moves "up" = -row)
# ============================================================================
PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK, KING = range(1, 9)
PROMOTED = 8
SENTE, GOTE = 1, -1
_GOLD   = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,0))
_SILVER = ((-1,-1),(-1,0),(-1,1),(1,-1),(1,1))
_DIAG, _ORTH = ((-1,-1),(-1,1),(1,-1),(1,1)), ((-1,0),(1,0),(0,-1),(0,1))
_ALL8 = _DIAG + _ORTH
KANJI   = {PAWN:'歩',LANCE:'香',KNIGHT:'桂',SILVER:'銀',GOLD:'金',BISHOP:'角',ROOK:'飛',KING:'玉'}
KANJI_P = {PAWN:'と',LANCE:'杏',KNIGHT:'圭',SILVER:'全',BISHOP:'馬',ROOK:'龍'}
LETTER  = {PAWN:'P',LANCE:'L',KNIGHT:'N',SILVER:'S',GOLD:'G',BISHOP:'B',ROOK:'R',KING:'K'}
LCODE   = {v: k for k, v in LETTER.items()}
EN   = {PAWN:'Pawn',LANCE:'Lance',KNIGHT:'Knight',SILVER:'Silver',GOLD:'Gold',
        BISHOP:'Bishop',ROOK:'Rook',KING:'King'}
EN_P = {PAWN:'Tokin',LANCE:'+Lance',KNIGHT:'+Knight',SILVER:'+Silver',
        BISHOP:'Horse',ROOK:'Dragon'}
FW, KJ = '１２３４５６７８９', '一二三四五六七八九'

def _moves(base, prom):
    if prom:
        if base in (PAWN, LANCE, KNIGHT, SILVER): return _GOLD, ()
        if base == BISHOP: return _ORTH, _DIAG
        if base == ROOK:   return _DIAG, _ORTH
    steps = {PAWN:((-1,0),), LANCE:(), KNIGHT:((-2,-1),(-2,1)), SILVER:_SILVER,
             GOLD:_GOLD, BISHOP:_DIAG, ROOK:_ORTH, KING:_ALL8}[base]
    slides = {PAWN:(), LANCE:((-1,0),), KNIGHT:(), SILVER:(), GOLD:(),
              BISHOP:_DIAG, ROOK:_ORTH, KING:()}[base]
    return steps, slides

MOVE_TABLE = {}
for _s in (SENTE, GOTE):
    for _b in range(1, 9):
        for _pr in (False, True):
            if _pr and _b in (GOLD, KING): continue
            _st, _sl = _moves(_b, _pr)
            if _s == GOTE:
                _st = tuple((-a, c) for a, c in _st)
                _sl = tuple((-a, c) for a, c in _sl)
            MOVE_TABLE[_s * (_b + (PROMOTED if _pr else 0))] = (_st, _sl)

# ---- precomputed geometry (hot-loop tables) ----
TARGETS = {}   # signed code -> per-square tuple of step-destination indices
RAYS = {}      # signed code -> per-square tuple of ordered ray tuples
for _code, (_st, _sl) in MOVE_TABLE.items():
    _tg, _ry = [()] * 81, [()] * 81
    for _sq in range(81):
        _r, _c = divmod(_sq, 9)
        _tg[_sq] = tuple((_r+dr)*9 + (_c+dc) for dr, dc in _st
                         if 0 <= _r+dr < 9 and 0 <= _c+dc < 9)
        _rays = []
        for dr, dc in _sl:
            path, rr, cc = [], _r+dr, _c+dc
            while 0 <= rr < 9 and 0 <= cc < 9:
                path.append(rr*9+cc); rr += dr; cc += dc
            if path: _rays.append(tuple(path))
        _ry[_sq] = tuple(_rays)
    TARGETS[_code] = tuple(_tg)
    RAYS[_code] = tuple(_ry)
KNIGHT_FROM = [()] * 81     # squares from which a Sente knight attacks sq
for _sq in range(81):
    _r, _c = divmod(_sq, 9)
    if _r + 2 < 9:
        KNIGHT_FROM[_sq] = tuple((_r+2)*9+cc for cc in (_c-1, _c+1)
                                 if 0 <= cc < 9)

# ---- Zobrist ----
_zr = random.Random(0x5100D)
ZPS = [[0]*17 for _ in range(81)]
for _sq in range(81):
    for _ci in range(17):
        if _ci != 8:
            ZPS[_sq][_ci] = _zr.getrandbits(64)
ZSIDE = _zr.getrandbits(64)
ZH = [[[0]*19 for _ in range(9)] for _ in range(2)]
for _s in range(2):
    for _t in range(1, 9):
        for _n in range(19):
            ZH[_s][_t][_n] = _zr.getrandbits(64)

@dataclass(frozen=True)
class Move:
    fr: int; fc: int; tr: int; tc: int
    piece: int; promo: bool = False
    @property
    def drop(self): return self.fr < 0
    def usi(self):
        if self.fr < 0:
            return f'{LETTER[abs(self.piece)]}*{sq_usi(self.tr, self.tc)}'
        return f'{sq_usi(self.fr,self.fc)}{sq_usi(self.tr,self.tc)}{"+" if self.promo else ""}'

def sq_usi(r, c): return f'{9-c}{chr(97+r)}'

def _emit_list(side, fr, fc, tr, tc, piece):
    out = []
    base = abs(piece)
    if base > KING or base in (GOLD, KING):
        out.append(Move(fr, fc, tr, tc, piece)); return out
    dest_z = tr <= 2 if side == SENTE else tr >= 6
    from_z = fr <= 2 if side == SENTE else fr >= 6
    must = ((base in (PAWN, LANCE) and (tr == 0 if side == SENTE else tr == 8)) or
            (base == KNIGHT and (tr <= 1 if side == SENTE else tr >= 7)))
    if must:
        out.append(Move(fr, fc, tr, tc, piece, promo=True))
    elif dest_z or from_z:
        out.append(Move(fr, fc, tr, tc, piece))
        out.append(Move(fr, fc, tr, tc, piece, promo=True))
    else:
        out.append(Move(fr, fc, tr, tc, piece))
    return out

class Position:
    __slots__ = ('bd', 'hands', 'side', '_key')
    def __init__(self, bd=None, hands=None, side=SENTE, key=None):
        self.bd = list(bd) if bd else [0]*81
        self.hands = hands if hands is not None else {SENTE: {}, GOTE: {}}
        self.side = side
        self._key = key
    def zkey(self):
        k = self._key
        if k is None:
            k = 0
            for sq in range(81):
                p = self.bd[sq]
                if p: k ^= ZPS[sq][p+8]
            if self.side == GOTE: k ^= ZSIDE
            for si, s in ((0, SENTE), (1, GOTE)):
                for t, n in self.hands[s].items():
                    if n: k ^= ZH[si][t][n]
            self._key = k
        return k
    def at(self, r, c): return self.bd[r*9+c]
    def king(self, side):
        t = KING*side
        for i in range(81):
            if self.bd[i] == t: return divmod(i, 9)
        return None
    def is_attacked(self, r, c, by):
        bd = self.bd
        kn = KNIGHT * by
        for asq in KNIGHT_FROM[r*9+c]:
            if bd[asq] == kn: return True
        for dr, dc in _ALL8:
            rr, cc, d = r+dr, c+dc, 1
            while 0 <= rr < 9 and 0 <= cc < 9:
                p = bd[rr*9+cc]
                if p:
                    if (p > 0) == (by > 0):
                        u = (-dr, -dc); st, sl = MOVE_TABLE[p]
                        if (d == 1 and u in st) or u in sl: return True
                    break
                rr += dr; cc += dc; d += 1
        return False
    def in_check(self, side=None):
        if side is None: side = self.side
        k = self.king(side)
        return k is not None and self.is_attacked(k[0], k[1], -side)
    def apply(self, m):
        bd = self.bd[:]
        hs = self.hands
        k0 = self._key
        if m.fr < 0:
            t = abs(m.piece)
            si = 0 if self.side == SENTE else 1
            oldn = hs[self.side].get(t, 0)
            h = dict(hs[self.side]); h[t] = h.get(t, 0) - 1
            if h[t] <= 0: del h[t]
            nh = {SENTE: hs[SENTE], GOTE: hs[GOTE]}; nh[self.side] = h
            pos = Position(bd, nh, -self.side)
            if k0 is not None:
                pos._key = (k0 ^ ZSIDE ^ ZPS[m.tr*9+m.tc][m.piece+8]
                            ^ ZH[si][t][oldn] ^ ZH[si][t][oldn-1])
            return pos
        cap = bd[m.tr*9+m.tc]
        nh, k = hs, k0
        if cap:
            cb = abs(cap)
            if cb > KING: cb -= PROMOTED
            si = 0 if self.side == SENTE else 1
            oldn = hs[self.side].get(cb, 0)
            h = dict(hs[self.side]); h[cb] = h.get(cb, 0) + 1
            nh = {SENTE: hs[SENTE], GOTE: hs[GOTE]}; nh[self.side] = h
            if k is not None:
                k ^= ZH[si][cb][oldn] ^ ZH[si][cb][oldn+1]
        bd[m.tr*9+m.tc] = m.piece + (PROMOTED if m.promo else 0)
        bd[m.fr*9+m.fc] = 0
        pos = Position(bd, nh, -self.side)
        if k is not None:
            k ^= ZSIDE ^ ZPS[m.fr*9+m.fc][m.piece+8]
            k ^= ZPS[m.tr*9+m.tc][(m.piece + (PROMOTED if m.promo else 0))+8]
            if cap: k ^= ZPS[m.tr*9+m.tc][cap+8]
            pos._key = k
        return pos
    def pseudo_moves(self):
        bd, side, out = self.bd, self.side, []
        for sq in range(81):
            p = bd[sq]
            if not p or (p > 0) != (side > 0): continue
            fr, fc = sq // 9, sq % 9
            for dsq in TARGETS[p][sq]:
                q = bd[dsq]
                if q == 0 or (q > 0) != (side > 0):
                    tr, tc = divmod(dsq, 9)
                    out.extend(_emit_list(side, fr, fc, tr, tc, p))
            for ray in RAYS[p][sq]:
                for dsq in ray:
                    q = bd[dsq]
                    if q and (q > 0) == (side > 0): break
                    tr, tc = divmod(dsq, 9)
                    out.extend(_emit_list(side, fr, fc, tr, tc, p))
                    if q: break
        hand = self.hands.get(side) or {}
        if any(v > 0 for v in hand.values()):
            for i in range(81):
                if bd[i]: continue
                tr, tc = divmod(i, 9)
                for pt, cnt in hand.items():
                    if cnt <= 0: continue
                    if side == SENTE:
                        if pt in (PAWN, LANCE) and tr == 0: continue
                        if pt == KNIGHT and tr <= 1:        continue
                    else:
                        if pt in (PAWN, LANCE) and tr == 8: continue
                        if pt == KNIGHT and tr >= 7:        continue
                    if pt == PAWN and any(bd[rr*9+tc] == side*PAWN for rr in range(9)):
                        continue
                    out.append(Move(-1, -1, tr, tc, side*pt))
        return out
    def legal_moves(self):
        res = []
        for m in self.pseudo_moves():
            ch = self.apply(m)
            k = ch.king(self.side)
            if k is None or not ch.is_attacked(k[0], k[1], -self.side):
                res.append(m)
        return res
    def __getstate__(self): return (self.bd, self.hands, self.side, self._key)
    def __setstate__(self, st):
        self.bd, self.hands, self.side, self._key = st

def startpos():
    bd = [0]*81
    back = [LANCE, KNIGHT, SILVER, GOLD, KING, GOLD, SILVER, KNIGHT, LANCE]
    for c in range(9):
        bd[c] = -back[c]; bd[72+c] = back[c]
        bd[18+c] = -PAWN; bd[54+c] = PAWN
    bd[8], bd[16] = -ROOK, -BISHOP
    bd[64], bd[70] = BISHOP, ROOK
    return Position(bd, {SENTE: {}, GOTE: {}}, SENTE)

# ============================================================================
# 2. ATTACKS / EVASIONS / CHECKING-MOVE GENERATOR
# ============================================================================
def attackers_of(pos, r, c, by):
    out, bd = [], pos.bd
    rr = r+2 if by == SENTE else r-2
    if 0 <= rr < 9:
        for cc in (c-1, c+1):
            if 0 <= cc < 9 and bd[rr*9+cc] == by*KNIGHT:
                out.append(rr*9+cc)
    for dr, dc in _ALL8:
        rr, cc, d = r+dr, c+dc, 1
        while 0 <= rr < 9 and 0 <= cc < 9:
            p = bd[rr*9+cc]
            if p:
                if (p > 0) == (by > 0):
                    u = (-dr, -dc); st, sl = MOVE_TABLE[p]
                    if (d == 1 and u in st) or u in sl:
                        out.append(rr*9+cc)
                break
            rr += dr; cc += dc; d += 1
    return out

def check_evasions(pos, mode='std'):
    side = pos.side
    k = pos.king(side)
    if k is None: return []
    kr, kc = k
    atts = attackers_of(pos, kr, kc, -side)
    out, seen = [], set()
    def ok(m):
        ch = pos.apply(m)
        kk = ch.king(side)
        return kk is not None and not ch.is_attacked(kk[0], kk[1], -side)
    def add(m):
        key = (m.fr, m.fc, m.tr, m.tc, m.promo)
        if key in seen: return
        if ok(m): seen.add(key); out.append(m)
    kp = pos.bd[kr*9+kc]
    kst, _ = MOVE_TABLE[kp]
    for dr, dc in kst:
        tr, tc = kr+dr, kc+dc
        if 0 <= tr < 9 and 0 <= tc < 9:
            q = pos.bd[tr*9+tc]
            if q == 0 or (q > 0) != (side > 0):
                add(Move(kr, kc, tr, tc, kp))
    if len(atts) != 1: return out
    a = atts[0]; ar, ac = divmod(a, 9)
    dr_, dc_ = ar-kr, ac-kc
    tset = {a}
    if max(abs(dr_), abs(dc_)) > 1 and ((dr_ == 0) != (dc_ == 0) or abs(dr_) == abs(dc_)):
        sdr, sdc = (dr_ > 0)-(dr_ < 0), (dc_ > 0)-(dc_ < 0)
        rr, cc = kr+sdr, kc+sdc
        while (rr, cc) != (ar, ac):
            tset.add(rr*9+cc); rr += sdr; cc += sdc
    for sqi in range(81):
        pce = pos.bd[sqi]
        if not pce or (pce > 0) != (side > 0) or abs(pce) == KING: continue
        r, c = divmod(sqi, 9)
        st, sl = MOVE_TABLE[pce]
        for dr2, dc2 in st:
            tr, tc = r+dr2, c+dc2
            if 0 <= tr < 9 and 0 <= tc < 9 and tr*9+tc in tset:
                q = pos.bd[tr*9+tc]
                if q == 0 or (q > 0) != (side > 0):
                    for mm in _emit_list(side, r, c, tr, tc, pce): add(mm)
        for dr2, dc2 in sl:
            tr, tc = r+dr2, c+dc2
            while 0 <= tr < 9 and 0 <= tc < 9:
                i = tr*9+tc; q = pos.bd[i]
                if q and (q > 0) == (side > 0): break
                if i in tset:
                    for mm in _emit_list(side, r, c, tr, tc, pce): add(mm)
                if q: break
                tr += dr2; tc += dc2
    hand = pos.hands.get(side) or {}
    if any(v > 0 for v in hand.values()):
        for i in list(tset):
            if pos.bd[i]: continue
            tr, tc = divmod(i, 9)
            for pt, cnt in hand.items():
                if cnt <= 0: continue
                if side == SENTE:
                    if pt in (PAWN, LANCE) and tr == 0: continue
                    if pt == KNIGHT and tr <= 1:        continue
                else:
                    if pt in (PAWN, LANCE) and tr == 8: continue
                    if pt == KNIGHT and tr >= 7:        continue
                if pt == PAWN and any(pos.bd[rr*9+tc] == side*PAWN for rr in range(9)):
                    continue
                add(Move(-1, -1, tr, tc, side*pt))
    att_sq = atts[0]
    if mode == 'fast':                     # refutation-first ordering
        def _sc(m):
            if m.fr >= 0 and m.tr*9+m.tc == att_sq: s = 0
            elif m.fr == kr and m.fc == kc: s = 1
            elif m.fr >= 0 and pos.at(m.tr, m.tc): s = 2
            elif m.fr < 0: s = 4
            else: s = 3
            return (s, m.tr*9+m.tc)
        out.sort(key=_sc)
    elif isinstance(mode, int):            # deterministic shuffle (consensus)
        random.Random((mode & 0xffffffff) ^ (pos.zkey() & 0xffffffff)).shuffle(out)
    else:                                  # 'std' — original behaviour
        out.sort(key=lambda m: 0 if (m.fr == kr and m.fc == kc) else
                 (1 if (m.fr >= 0 and pos.at(m.tr, m.tc)) else 2))
    return out

def checking_moves(pos):
    side = pos.side
    k = pos.king(-side)
    if k is None: return []
    kr, kc = k
    bd = pos.bd
    cands = set()
    def try_move(fr, fc, tr, tc, piece):
        for m in _emit_list(side, fr, fc, tr, tc, piece):
            cands.add((m.fr, m.fc, m.tr, m.tc, m.promo, piece))
    for sq in range(81):
        p = bd[sq]
        if not p or (p > 0) != (side > 0): continue
        r, c = divmod(sq, 9)
        base = abs(p)
        b0 = base - PROMOTED if base > KING else base
        variants = [p] if b0 in (GOLD, KING) else [p, p + PROMOTED]
        for v in variants:
            st, sl = MOVE_TABLE[v]
            for dr, dc in st:
                tr, tc = kr-dr, kc-dc
                if 0 <= tr < 9 and 0 <= tc < 9 and (tr, tc) != (r, c):
                    q = bd[tr*9+tc]
                    if q == 0 or (q > 0) != (side > 0):
                        try_move(r, c, tr, tc, p)
            for dr, dc in sl:
                for kk in range(1, 10):
                    tr, tc = kr-dr*kk, kc-dc*kk
                    if not (0 <= tr < 9 and 0 <= tc < 9): break
                    if (tr, tc) == (r, c): continue
                    clear = True
                    for j in range(1, kk):
                        qr, qc = kr-dr*j, kc-dc*j
                        if (qr, qc) != (r, c) and bd[qr*9+qc]:
                            clear = False; break
                    if not clear: break
                    q = bd[tr*9+tc]
                    if q == 0 or (q > 0) != (side > 0):
                        try_move(r, c, tr, tc, p)
                    if q: break
    hand = pos.hands.get(side) or {}
    for pt, cnt in hand.items():
        if cnt <= 0: continue
        st, _ = MOVE_TABLE[side*pt]
        for dr, dc in st:
            tr, tc = kr-dr, kc-dc
            if not (0 <= tr < 9 and 0 <= tc < 9) or bd[tr*9+tc]: continue
            if side == SENTE:
                if pt in (PAWN, LANCE) and tr == 0: continue
                if pt == KNIGHT and tr <= 1:        continue
            else:
                if pt in (PAWN, LANCE) and tr == 8: continue
                if pt == KNIGHT and tr >= 7:        continue
            if pt == PAWN and any(bd[rr*9+tc] == side*PAWN for rr in range(9)):
                continue
            cands.add((-1, -1, tr, tc, False, side*pt))
    blockers = set()
    for sq in range(81):
        p = bd[sq]
        if not p or (p > 0) != (side > 0): continue
        st, sl = MOVE_TABLE[p]
        if not sl: continue
        r, c = divmod(sq, 9)
        for dr, dc in sl:
            rr, cc, first = r+dr, c+dc, -1
            while 0 <= rr < 9 and 0 <= cc < 9:
                q = bd[rr*9+cc]
                if q:
                    if first < 0:
                        if (q > 0) == (side > 0) and abs(q) != KING: first = rr*9+cc
                        else: break
                    else:
                        if q == KING*(-side): blockers.add(first)
                        break
                rr += dr; cc += dc
    for b in blockers:
        p = bd[b]; r, c = divmod(b, 9)
        st, sl = MOVE_TABLE[p]
        for dr, dc in st:
            tr, tc = r+dr, c+dc
            if 0 <= tr < 9 and 0 <= tc < 9:
                q = bd[tr*9+tc]
                if q == 0 or (q > 0) != (side > 0):
                    try_move(r, c, tr, tc, p)
        for dr, dc in sl:
            tr, tc = r+dr, c+dc
            while 0 <= tr < 9 and 0 <= tc < 9:
                q = bd[tr*9+tc]
                if q and (q > 0) == (side > 0): break
                try_move(r, c, tr, tc, p)
                if q: break
                tr += dr; tc += dc
    out = []
    for fr, fc, tr, tc, promo, piece in cands:
        m = Move(fr, fc, tr, tc, piece, promo)
        ch = pos.apply(m)
        kk = ch.king(side)
        if kk is not None and ch.is_attacked(kk[0], kk[1], -side): continue
        if not ch.in_check(): continue
        out.append(m)
    return out

# ============================================================================
# 3. NOTATION / SFEN
# ============================================================================
def move_notation(m):
    s = '▲' if m.piece > 0 else '△'
    base = abs(m.piece); pr = base > KING
    if pr: base -= PROMOTED
    ch = KANJI_P[base] if pr else KANJI[base]
    sq = f'{FW[8-m.tc]}{KJ[m.tr]}'
    if m.fr < 0:  return f'{s}{sq}{ch}打'
    if m.promo:   return f'{s}{sq}{KANJI[base]}成'
    return f'{s}{sq}{ch}'

def move_english(m):
    base = abs(m.piece); pr = base > KING
    b2 = base-PROMOTED if pr else base
    name = EN_P[b2] if pr else EN[b2]
    sq = sq_usi(m.tr, m.tc)
    if m.fr < 0:  return f'{name} drops on {sq}'
    if m.promo:   return f'{name} to {sq} (promotes)'
    return f'{name} to {sq}'

SF_CH = {ROOK:'R', BISHOP:'B', GOLD:'G', SILVER:'S', KNIGHT:'N', LANCE:'L',
         PAWN:'P', KING:'K'}
SF_CODE = {v: k for k, v in SF_CH.items()}

def pos_to_sfen(pos):
    rows = []
    for r in range(9):
        s, e = '', 0
        for c in range(9):
            p = pos.bd[r*9+c]
            if not p: e += 1; continue
            if e: s += str(e); e = 0
            b = abs(p); pr = b > KING
            if pr: b -= PROMOTED
            ch = SF_CH[b]
            s += ('+' if pr else '') + (ch if p > 0 else ch.lower())
        if e: s += str(e)
        rows.append(s)
    def hand_s(side):
        return ''.join(SF_CH[t]*n if n > 1 else SF_CH[t]
                       for t in (ROOK, BISHOP, GOLD, SILVER, KNIGHT, LANCE, PAWN)
                       for n in [pos.hands[side].get(t, 0)] if n > 0)
    hs = hand_s(SENTE) + hand_s(GOTE)
    return f"{'/'.join(rows)} {'b' if pos.side == SENTE else 'w'} {hs or '-'} -"

def pos_from_sfen(s):
    parts = s.split()
    bd, r, c, promo = [0]*81, 0, 0, False
    for ch in parts[0]:
        if ch == '/': r += 1; c = 0; continue
        if ch.isdigit(): c += int(ch); continue
        if ch == '+': promo = True; continue
        code = SF_CODE[ch.upper()]
        bd[r*9+c] = (code + (PROMOTED if promo else 0)) * (1 if ch.isupper() else -1)
        promo = False; c += 1
    hands = {SENTE: {}, GOTE: {}}
    if len(parts) > 2 and parts[2] != '-':
        side = SENTE; cnt = 1
        for ch in parts[2] + ' ':
            if ch.isdigit(): cnt = int(ch); continue
            if ch == ' ': side = GOTE; continue
            t = SF_CODE[ch.upper()]
            tgt = hands[SENTE if ch.isupper() else GOTE]
            tgt[t] = tgt.get(t, 0) + cnt; cnt = 1
    return Position(bd, hands, SENTE if parts[1] == 'b' else GOTE)

def move_from_usi(pos, u):
    if '*' in u:
        t = LCODE[u[0]]
        file_no, rank = int(u[1]), ord(u[2]) - 97
        return Move(-1, -1, rank, 9-file_no, pos.side*t)
    promo = u.endswith('+')
    if promo: u = u[:-1]
    fr, ff, tr, tf = ord(u[1])-97, int(u[0]), ord(u[3])-97, int(u[2])
    return Move(fr, 9-ff, tr, 9-tf, pos.bd[fr*9+9-ff], promo)

# ============================================================================
# 3b. POSITION LEGALITY — every published board must be a *possible* position
# ============================================================================
INITIAL_PER_SIDE = {KING: 1, ROOK: 1, BISHOP: 1, GOLD: 2, SILVER: 2,
                    KNIGHT: 2, LANCE: 2, PAWN: 9}
INITIAL_TOTAL = {k: 2*v for k, v in INITIAL_PER_SIDE.items()}
FORCED_SQ = {SENTE: {PAWN: (0,), LANCE: (0,), KNIGHT: (0, 1)},
             GOTE:  {PAWN: (8,), LANCE: (8,), KNIGHT: (7, 8)}}

def base_of(code):
    b = abs(code)
    return b - PROMOTED if b > KING else b

def global_type_counts(pos):
    cnt = {}
    for p in pos.bd:
        if p: cnt[base_of(p)] = cnt.get(base_of(p), 0) + 1
    for s in (SENTE, GOTE):
        for t, n in pos.hands[s].items():
            cnt[t] = cnt.get(t, 0) + n
    return cnt

def position_violations(pos):
    v = []
    if pos.king(SENTE) is None or pos.king(GOTE) is None:
        return ['missing king']
    sr, sc = pos.king(SENTE); gr, gc = pos.king(GOTE)
    if max(abs(sr-gr), abs(sc-gc)) < 2:
        v.append('kings adjacent')
    if pos.in_check(-pos.side):
        v.append('side not to move in check')
    for t, n in global_type_counts(pos).items():
        if n > INITIAL_TOTAL[t]:
            v.append(f'inventory overflow type {t} ({n}>{INITIAL_TOTAL[t]})')
    for sq in range(81):
        p = pos.bd[sq]
        if not p: continue
        b = abs(p)
        if b > KING:
            if base_of(p) in (GOLD, KING):
                v.append(f'illegal promoted piece at {sq}')
            continue
        r, _c = divmod(sq, 9)
        side = SENTE if p > 0 else GOTE
        if r in FORCED_SQ[side].get(b, ()):
            v.append(f'unpromoted {b} on forced row {r}')
    for side in (SENTE, GOTE):
        seen = {}
        for sq in range(81):
            if pos.bd[sq] == side*PAWN:
                cc = sq % 9
                seen[cc] = seen.get(cc, 0) + 1
        for cc, n in seen.items():
            if n > 1: v.append(f'nifu file {cc} side {side}')
    for s in (SENTE, GOTE):
        if pos.hands[s].get(KING, 0) > 0:
            v.append('king in hand')
    return v

def normalize_position(pos, rng):
    if pos.king(SENTE) is None or pos.king(GOTE) is None: return None
    sr, sc = pos.king(SENTE); gr, gc = pos.king(GOTE)
    if max(abs(sr-gr), abs(sc-gc)) < 2: return None
    if pos.in_check(-pos.side): return None
    bd = list(pos.bd)
    hands = {SENTE: dict(pos.hands[SENTE]), GOTE: dict(pos.hands[GOTE])}
    for i in range(81):
        p = bd[i]
        if p and abs(p) > KING and base_of(p) in (GOLD, KING):
            bd[i] = (1 if p > 0 else -1) * base_of(p)
    for i in range(81):
        p = bd[i]
        if not p or abs(p) > KING: continue
        r, _c = divmod(i, 9)
        side = SENTE if p > 0 else GOTE
        if r in FORCED_SQ[side].get(abs(p), ()):
            bd[i] = p + PROMOTED
    for side in (SENTE, GOTE):
        seen = set()
        for i in range(81):
            if bd[i] == side*PAWN:
                cc = i % 9
                if cc in seen: bd[i] = 0
                else: seen.add(cc)
    for _pass in range(80):
        gtc = {}
        for p in bd:
            if p: gtc[base_of(p)] = gtc.get(base_of(p), 0) + 1
        for s in (SENTE, GOTE):
            for t, n in hands[s].items():
                gtc[t] = gtc.get(t, 0) + n
        over = [(t, n - INITIAL_TOTAL[t]) for t, n in gtc.items()
                if n > INITIAL_TOTAL[t]]
        if not over: break
        t, excess = over[0]
        in_hands = hands[SENTE].get(t, 0) + hands[GOTE].get(t, 0)
        if in_hands > 0:
            s = SENTE if hands[SENTE].get(t, 0) >= hands[GOTE].get(t, 0) else GOTE
            hands[s][t] -= 1
            if hands[s][t] <= 0: del hands[s][t]
        else:
            spots = [i for i in range(81) if bd[i] and base_of(bd[i]) == t
                     and abs(bd[i]) != KING]
            if not spots: return None
            bd[rng.choice(spots)] = 0
    return Position(bd, hands, pos.side)

def fill_to_full(pos, rng):
    cnt = global_type_counts(pos)
    hands = {SENTE: dict(pos.hands[SENTE]), GOTE: dict(pos.hands[GOTE])}
    for t in (ROOK, BISHOP, GOLD, SILVER, KNIGHT, LANCE, PAWN):
        need = INITIAL_TOTAL[t] - cnt.get(t, 0)
        for _ in range(max(0, need)):
            s = SENTE if rng.random() < 0.5 else GOTE
            hands[s][t] = hands[s].get(t, 0) + 1
    return Position(pos.bd[:], hands, pos.side)

# ============================================================================
# 3c. DIFFERENTIAL-TEST HELPERS (naive reference implementations)
# ============================================================================
def mkey(m): return (m.fr, m.fc, m.tr, m.tc, m.promo, m.piece)

def mirror_position(pos):
    bd = [0]*81
    for sq in range(81):
        r, c = divmod(sq, 9)
        bd[r*9 + (8-c)] = pos.bd[sq]
    return Position(bd, {SENTE: dict(pos.hands[SENTE]),
                         GOTE: dict(pos.hands[GOTE])}, pos.side)

def mirror_move(m):
    if m.fr < 0:
        return Move(-1, -1, m.tr, 8-m.tc, m.piece, m.promo)
    return Move(m.fr, 8-m.fc, m.tr, 8-m.tc, m.piece, m.promo)

def mirror_bb(bb):
    out = 0
    for sq in range(81):
        if (bb >> sq) & 1:
            r, c = divmod(sq, 9)
            out |= 1 << (r*9 + (8-c))
    return out

def naive_attacked(pos, r, c, by):
    for sq in range(81):
        p = pos.bd[sq]
        if not p or (p > 0) != (by > 0): continue
        rr, cc = divmod(sq, 9)
        st, sl = MOVE_TABLE[p]
        for dr, dc in st:
            if rr+dr == r and cc+dc == c: return True
        for dr, dc in sl:
            tr, tc = rr+dr, cc+dc
            while 0 <= tr < 9 and 0 <= tc < 9:
                if tr == r and tc == c: return True
                if pos.bd[tr*9+tc]: break
                tr += dr; tc += dc
    return False

def naive_legal_moves(pos):
    """Deliberately naive generator, independent of pseudo_moves/evastions."""
    side, out = pos.side, []
    for sq in range(81):
        p = pos.bd[sq]
        if not p or (p > 0) != (side > 0): continue
        r, c = divmod(sq, 9)
        b0 = abs(p) - PROMOTED if abs(p) > KING else abs(p)
        variants = [p] if b0 in (GOLD, KING) else [p, p+PROMOTED]
        for v in variants:
            st, sl = MOVE_TABLE[v]
            for dr, dc in st:
                tr, tc = r+dr, c+dc
                if 0 <= tr < 9 and 0 <= tc < 9:
                    q = pos.bd[tr*9+tc]
                    if q == 0 or (q > 0) != (side > 0):
                        out.extend(_emit_list(side, r, c, tr, tc, p))
            for dr, dc in sl:
                tr, tc = r+dr, c+dc
                while 0 <= tr < 9 and 0 <= tc < 9:
                    q = pos.bd[tr*9+tc]
                    if q and (q > 0) == (side > 0): break
                    out.extend(_emit_list(side, r, c, tr, tc, p))
                    if q: break
                    tr += dr; tc += dc
    hand = pos.hands.get(side) or {}
    if any(v > 0 for v in hand.values()):
        for i in range(81):
            if pos.bd[i]: continue
            tr, tc = divmod(i, 9)
            for pt, cnt in hand.items():
                if cnt <= 0: continue
                if side == SENTE:
                    if pt in (PAWN, LANCE) and tr == 0: continue
                    if pt == KNIGHT and tr <= 1:        continue
                else:
                    if pt in (PAWN, LANCE) and tr == 8: continue
                    if pt == KNIGHT and tr >= 7:        continue
                if pt == PAWN and any(pos.bd[rr*9+tc] == side*PAWN for rr in range(9)):
                    continue
                out.append(Move(-1, -1, tr, tc, side*pt))
    res = []
    for m in out:
        ch = pos.apply(m)
        k = ch.king(side)
        if k is not None and not naive_attacked(ch, k[0], k[1], -side):
            res.append(m)
    return res

def random_sparse_pos(rng):
    while True:
        bd = [0]*81
        s, g = rng.randrange(81), rng.randrange(81)
        if s == g: continue
        sr, sc = divmod(s, 9); gr, gc = divmod(g, 9)
        if max(abs(sr-gr), abs(sc-gc)) < 2: continue
        bd[s], bd[g] = KING, -KING
        break
    types = [t for t, n in ((ROOK, 2), (BISHOP, 2), (GOLD, 4), (SILVER, 4),
                            (KNIGHT, 4), (LANCE, 4), (PAWN, 18)) for _ in range(n)]
    rng.shuffle(types)
    empt = [i for i in range(81) if not bd[i]]
    rng.shuffle(empt)
    for i in empt[:rng.randint(4, 14)]:
        bd[i] = types.pop() * rng.choice((1, -1))
    return Position(bd, {SENTE: {}, GOTE: {}},
                    SENTE if rng.random() < 0.5 else GOTE)

def property_tests(rng_seed=99, rounds=12):
    rng = random.Random(rng_seed)
    for i in range(rounds):
        pos = random_sparse_pos(rng)
        mir = mirror_position(pos)
        A = {mkey(m) for m in pos.legal_moves()}
        assert A == {mkey(mirror_move(m)) for m in mir.legal_moves()}, \
            f'mirror law failed #{i}'
        assert A == {mkey(m) for m in naive_legal_moves(pos)}, \
            f'naive differential failed #{i}'
        assert attack_map(pos, SENTE) == mirror_bb(attack_map(mir, SENTE)), \
            f'attack symmetry failed #{i}'
    return True

# ============================================================================
# 4. MATE SOLVER — early-exit AND/OR (n<=2) + df-pn (n>=3), shared Zobrist TT
# ============================================================================
_INF = 1 << 40
_SHARED_TT_MAX = 1_500_000
_SHARED = {'mf': {True: {}, False: {}}, 'dfpn': {True: {}, False: {}},
           'cc': {}}

class _Budget(Exception): pass

class MateSolver:
    def __init__(self, allow_pawn_mate=False, deadline=None, node_cap=400000,
                 persistent=True, mode='fast'):
        self.allow = allow_pawn_mate; self.deadline = deadline
        self.cap = node_cap; self.nodes = 0; self.mode = mode
        if persistent:
            self.mf = _SHARED['mf'][bool(allow_pawn_mate)]
            self.dfpn = _SHARED['dfpn'][bool(allow_pawn_mate)]
            self.cc = _SHARED['cc']
        else:
            self.mf = {}; self.dfpn = {}; self.cc = {}
        self._iters = 0

    def _tick(self):
        self.nodes += 1
        if self.nodes > self.cap: raise _Budget()
        if self.deadline and (self.nodes & 63) == 0 and time.time() > self.deadline:
            raise _Budget()

    @staticmethod
    def _maybe_clear(*ds):
        for d in ds:
            if len(d) > _SHARED_TT_MAX: d.clear()

    def checks(self, pos):
        key = pos.zkey()
        got = self.cc.get(key)
        if got is None:
            got = checking_moves(pos); self.cc[key] = got
            self._maybe_clear(self.cc)
        return got

    def _ev(self, pos): return check_evasions(pos, self.mode)
    def _pawn_ok(self, m):
        return not (m.fr < 0 and abs(m.piece) == PAWN and not self.allow)

    def _mf_rec(self, pos, n):
        self._tick()
        for m in self.checks(pos):
            if not self._pawn_ok(m): continue
            ch = pos.apply(m)
            ev = self._ev(ch)
            if not ev: return True
            if n > 1 and all(not self.mates_fast(ch.apply(rp), n-1) for rp in ev):
                return True
        return False

    def mates_fast(self, pos, n):
        if n <= 0: return False
        key = (pos.zkey(), n)
        got = self.mf.get(key)
        if got is not None: return got
        res = self._dfpn(pos, n) if n >= 3 else self._mf_rec(pos, n)
        self._maybe_clear(self.mf)
        self.mf[key] = res
        return res

    def mates(self, pos, n):
        if n <= 0 or not self.mates_fast(pos, n): return []
        res = []
        for m in self.checks(pos):
            if not self._pawn_ok(m): continue
            ch = pos.apply(m)
            ev = self._ev(ch)
            if not ev: res.append(m); continue
            if n > 1 and all(not self.mates_fast(ch.apply(rp), n-1) for rp in ev):
                res.append(m)
        return res

    # ---- depth-first proof-number search (mate in <= n) ----
    def _dfpn(self, root, n):
        self._iters = 0
        key = (root.zkey(), n, True)
        e = self.dfpn.get(key)
        if e is not None and (e[0] == 0 or e[1] == 0):
            return e[0] == 0
        self._dfpn_dfs(key, root, n, _INF, _INF, frozenset())
        e = self.dfpn.get(key) or [1, 1]
        return e[0] == 0

    def _gi(self):
        self._iters += 1
        if self._iters > self.cap * 2: raise _Budget()

    def _dfpn_dfs(self, key, pos, n, phi, delta, path):
        self._tick()
        e = self.dfpn.get(key)
        if e is None:
            e = [1, 1]; self.dfpn[key] = e
        if e[0] == 0 or e[1] == 0: return
        zkey, depth, is_or = key
        if zkey in path:
            e[0], e[1] = _INF, 0; return
        if is_or:
            if depth <= 0:
                e[0], e[1] = _INF, 0; return
            kids = [m for m in self.checks(pos) if self._pawn_ok(m)]
            if not kids:
                e[0], e[1] = _INF, 0; return
            while True:
                self._gi()
                if e[0] == 0 or e[1] == 0: return
                rows = []
                for m in kids:
                    ch = pos.apply(m)
                    ck = (ch.zkey(), depth, False)
                    ce = self.dfpn.get(ck) or [1, 1]
                    rows.append(((_INF, 0) if ck[0] in path else (ce[0], ce[1]),
                                 ck, ch, m))
                if any(r[0][0] == 0 for r in rows):
                    e[0], e[1] = 0, _INF; return
                live = [r for r in rows if r[0][1] != 0]
                if not live:
                    e[0], e[1] = _INF, 0; return
                live.sort(key=lambda r: (r[0][1], r[0][0]))
                e[0] = min(r[0][0] for r in live)
                dn_sum = sum(r[0][1] for r in rows)
                e[1] = dn_sum if dn_sum < _INF else _INF
                if e[0] >= phi and e[1] >= delta: return
                c = live[0]
                dn2 = live[1][0][1] if len(live) > 1 else _INF
                phi_c = min(phi, dn2 + 1) if dn2 < _INF else phi
                self._dfpn_dfs(c[1], c[2], depth, phi_c, delta, path | {zkey})
        else:
            ev = self._ev(pos)
            if not ev:
                e[0], e[1] = 0, _INF; return
            while True:
                self._gi()
                if e[0] == 0 or e[1] == 0: return
                rows = []
                for rp in ev:
                    c2 = pos.apply(rp)
                    ck = (c2.zkey(), depth-1, True)
                    ce = self.dfpn.get(ck) or [1, 1]
                    rows.append(((_INF, 0) if ck[0] in path else (ce[0], ce[1]),
                                 ck, c2, rp))
                if any(r[0][1] == 0 for r in rows):
                    e[0], e[1] = _INF, 0; return
                live = [r for r in rows if r[0][0] != 0]
                if not live:
                    e[0], e[1] = 0, _INF; return
                live.sort(key=lambda r: (r[0][0], r[0][1]))
                e[1] = min(r[0][1] for r in live)
                pn_sum = sum(r[0][0] for r in rows)
                e[0] = pn_sum if pn_sum < _INF else _INF
                if e[0] >= phi and e[1] >= delta: return
                c = live[0]
                pn2 = live[1][0][0] if len(live) > 1 else _INF
                delta_c = min(delta, pn2 + 1) if pn2 < _INF else delta
                self._dfpn_dfs(c[1], c[2], depth-1, phi, delta_c, path | {zkey})

def resist_len(solver, pos, n):
    best = 0
    try:
        ks = solver.mates(pos, n)
    except _Budget:
        return 0
    for m in ks:
        L = 1
        ch = pos.apply(m)
        ev = check_evasions(ch)
        if ev:
            mx = 0
            try:
                for rp in ev:
                    mx = max(mx, resist_len(solver, ch.apply(rp), n-1) + 1)
            except _Budget:
                pass
            L += mx
        if L > best: best = L
        if best >= 2*n: break
    return best

def solution_tree(solver, pos, n, m):
    """{'move','after','replies':{defense:{'pos','sub'}}} — keeps ALL
    longest-resistance defenses; terminal node's 'move' is the mating move."""
    ch = pos.apply(m)
    ev = check_evasions(ch)
    node = {'move': m, 'after': ch, 'replies': {}}
    if not ev:
        return node
    scored = []
    for rp in ev:
        c2 = ch.apply(rp)
        subs = solver.mates(c2, n-1)
        if not subs: return None
        scored.append((resist_len(solver, c2, n-1), rp, subs, c2))
    top = max(s[0] for s in scored)
    for res, rp, subs, c2 in scored:
        if res < top: continue
        subs = sorted(subs, key=lambda mm: (0 if mm.fr < 0 else 1,
                      -(abs(c2.at(mm.tr, mm.tc)) if mm.fr >= 0 else 0)))
        sub = solution_tree(solver, c2, n-1, subs[0])
        if sub is None: return None
        node['replies'][rp] = {'pos': c2, 'sub': sub}
    return node

def pv_from_node(node):
    moves = []
    while node:
        moves.append(node['move'])
        if not node['replies']: break
        rp = next(iter(node['replies']))
        moves.append(rp)
        node = node['replies'][rp]['sub']
    return moves

def alt_branch(puz):
    """(pos_after_key, alt_defense, refute_moves) — 2nd longest-resistance
    root defense and its forced refutation."""
    if not puz.depth or puz.tree is None or len(puz.pv) < 3:
        return None
    replies = list(puz.tree.get('replies', {}).items())
    if len(replies) < 2: return None
    rp, entry = replies[1]
    sub = entry['sub']
    refute = pv_from_node(sub) if sub else []
    if not refute: return None
    return puz.tree['after'], rp, refute

def mate_key_kind(pos, m):
    ch = pos.apply(m)
    if not ch.in_check(): return None
    k = ch.king(GOTE)
    atts = attackers_of(ch, k[0], k[1], SENTE) if k else []
    if len(atts) >= 2: return 'double'
    dest = m.tr*9+m.tc
    if atts and all(a != dest for a in atts): return 'discovered'
    return 'normal'

# ============================================================================
# 5. TACTICS ENGINE  (alpha-beta + quiescence, honest material verification)
# ============================================================================
MATE = 100000
VAL = {PAWN:1, LANCE:3, KNIGHT:4, SILVER:5, GOLD:6, BISHOP:8, ROOK:10, KING:0}
_PBN = {PAWN:5, LANCE:4, KNIGHT:3, SILVER:2, BISHOP:3, ROOK:2}
def piece_value(code):
    b = abs(code)
    if b > KING: return VAL[b-PROMOTED] + _PBN.get(b-PROMOTED, 0)
    return VAL[b]
def material_score(pos):
    s = sum(piece_value(p)*(1 if p > 0 else -1) for p in pos.bd if p)
    for side, h in pos.hands.items():
        for t, c in h.items(): s += VAL[t]*c*(1 if side == SENTE else -1)
    return s
def cap_value(pos, m):
    return piece_value(pos.at(m.tr, m.tc)) if m.fr >= 0 else 0
def _tick(st):
    if st[0] > st[1]: raise _Budget()
    if st[2] and (st[0] & 255) == 0 and time.time() > st[2]: raise _Budget()

def qsearch(pos, alpha, beta, st, qd=6):
    st[0] += 1; _tick(st)
    stand = material_score(pos)*pos.side
    if stand >= beta or qd == 0: return stand
    best = stand
    if alpha < stand: alpha = stand
    caps = [m for m in pos.pseudo_moves()
            if (m.fr >= 0 and pos.at(m.tr, m.tc)) or m.promo]
    caps.sort(key=lambda m: -cap_value(pos, m))
    for m in caps:
        ch = pos.apply(m)
        k = ch.king(pos.side)
        if k is not None and ch.is_attacked(k[0], k[1], -pos.side): continue
        v = -qsearch(ch, -beta, -alpha, st, qd-1)
        if v > best: best = v
        if v > alpha: alpha = v
        if alpha >= beta: break
    return best

def pv_line(pos, depth, st):
    st[0] += 1; _tick(st)
    if depth == 0: return qsearch(pos, -10**9, 10**9, st), []
    moves = pos.legal_moves()
    if not moves: return ((-MATE-depth) if pos.in_check() else 0), []
    moves.sort(key=lambda m: -cap_value(pos, m))
    best, bpv, alpha = -10**9, [], -10**9
    for m in moves[:24 if depth >= 2 else 16]:
        s, pv = pv_line(pos.apply(m), depth-1, st)
        s = -s
        if s > best: best, bpv = s, [m]+pv
        if s > alpha: alpha = s
        if alpha >= beta: break
    return best, bpv

def is_pinned(pos, sq):
    p = pos.bd[sq]
    if not p or p > 0: return False
    k = pos.king(GOTE)
    if k is None: return False
    pos.bd[sq] = 0
    hit = pos.is_attacked(k[0], k[1], SENTE)
    pos.bd[sq] = p
    return hit

def fork_targets(ch, m):
    sq = m.tr*9+m.tc; p = ch.bd[sq]
    if not p or p < 0: return 0
    r, c = divmod(sq, 9); mask = 0
    st, sl = MOVE_TABLE[p]
    for dr, dc in st:
        tr, tc = r+dr, c+dc
        if 0 <= tr < 9 and 0 <= tc < 9: mask |= 1 << (tr*9+tc)
    for dr, dc in sl:
        tr, tc = r+dr, c+dc
        while 0 <= tr < 9 and 0 <= tc < 9:
            mask |= 1 << (tr*9+tc)
            if ch.bd[tr*9+tc]: break
            tr += dr; tc += dc
    n = 0
    for sqi in range(81):
        q = ch.bd[sqi]
        if q < 0 and (abs(q) == KING or piece_value(q) >= 6) and (mask >> sqi) & 1:
            n += 1
    return n

def skewer_target(ch, m):
    k = ch.king(GOTE)
    if not k: return False
    dr, dc = k[0]-m.tr, k[1]-m.tc
    if not (((dr == 0) != (dc == 0)) or (dr != 0 and abs(dr) == abs(dc))): return False
    sdr, sdc = (dr > 0)-(dr < 0), (dc > 0)-(dc < 0)
    r, c = k[0]+sdr, k[1]+sdc
    while 0 <= r < 9 and 0 <= c < 9:
        p = ch.bd[r*9+c]
        if p: return p < 0 and piece_value(p) >= 6
        r += sdr; c += sdc
    return False

def disc_unmasked(pos, m):
    if m.fr < 0: return False
    ch = pos.apply(m)
    if ch.in_check(): return False
    for sq in range(81):
        q = ch.bd[sq]
        if q >= 0 or (abs(q) != KING and piece_value(q) < 6): continue
        r, c = divmod(sq, 9)
        for a in attackers_of(ch, r, c, SENTE):
            sp = ch.bd[a]
            if sp <= 0: continue
            st, sl = MOVE_TABLE[sp]
            if not sl: continue
            ar, ac = divmod(a, 9)
            dr, dc = r-ar, c-ac
            if not ((dr == 0 and dc != 0) or (dc == 0 and dr != 0) or
                    (dr != 0 and abs(dr) == abs(dc))): continue
            sr, sc = (dr > 0)-(dr < 0), (dc > 0)-(dc < 0)
            rr, cc, pre = ar+sr, ac+sc, None
            while (rr, cc) != (r, c):
                if pos.bd[rr*9+cc]: pre = (rr, cc); break
                rr += sr; cc += sc
            if pre == (m.fr, m.fc): return True
    return False

def attack_map(pos, side):
    bd, m = pos.bd, 0
    for i in range(81):
        p = bd[i]
        if not p or (p > 0) != (side > 0): continue
        r, c = divmod(i, 9)
        st, sl = MOVE_TABLE[p]
        for dr, dc in st:
            tr, tc = r+dr, c+dc
            if 0 <= tr < 9 and 0 <= tc < 9: m |= 1 << (tr*9+tc)
        for dr, dc in sl:
            tr, tc = r+dr, c+dc
            while 0 <= tr < 9 and 0 <= tc < 9:
                m |= 1 << (tr*9+tc)
                if bd[tr*9+tc]: break
                tr += dr; tc += dc
    return m

def blocker_squares(pos):
    side, bk = pos.side, set()
    for sq in range(81):
        p = pos.bd[sq]
        if not p or (p > 0) != (side > 0): continue
        st, sl = MOVE_TABLE[p]
        if not sl: continue
        r, c = divmod(sq, 9)
        for dr, dc in sl:
            rr, cc, first = r+dr, c+dc, -1
            while 0 <= rr < 9 and 0 <= cc < 9:
                q = pos.bd[rr*9+cc]
                if q:
                    if first < 0:
                        if (q > 0) == (side > 0): first = rr*9+cc
                        else: break
                    else:
                        if q < 0 and piece_value(q) >= 6: bk.add(first)
                        break
                rr += dr; cc += dc
    return bk

def classify(pos, m):
    ch = pos.apply(m)
    k = ch.king(GOTE)
    atts = attackers_of(ch, k[0], k[1], SENTE) if k else []
    if ch.in_check():
        if len(atts) >= 2: return 'double'
        dest = m.tr*9+m.tc
        if atts and all(a != dest for a in atts): return 'discovered'
    if m.fr >= 0 and pos.at(m.tr, m.tc) and is_pinned(pos, m.tr*9+m.tc):
        return 'pin'
    if ch.in_check() and skewer_target(ch, m): return 'skewer'
    if fork_targets(ch, m) >= 2: return 'fork'
    if disc_unmasked(pos, m): return 'discattack'
    if m.promo: return 'promo'
    return 'winpiece'

TACTIC_MIN = {'fork': 3, 'skewer': 3, 'pin': 3, 'promo': 3, 'quiet': 3,
              'promoch': 3, 'discattack': 5, 'winpiece': 5}
ACCEPT = {'winpiece': {'winpiece', 'fork', 'skewer', 'pin', 'promo', 'discattack'}}

def tactic_candidates(pos):
    bk = blocker_squares(pos)
    out = []
    for m in pos.legal_moves():
        ch = pos.apply(m)
        if (m.fr >= 0 and pos.at(m.tr, m.tc)) or m.promo or ch.in_check():
            out.append(m); continue
        if m.fr >= 0 and (m.fr*9+m.fc) in bk:
            out.append(m); continue
        am = attack_map(ch, SENTE); n = 0
        for sqi in range(81):
            q = ch.bd[sqi]
            if q < 0 and (abs(q) == KING or piece_value(q) >= 6) and (am >> sqi) & 1:
                n += 1
                if n >= 2: break
        if n >= 2: out.append(m)
    return out

def find_tactic(pos, kind, opts, deadline):
    try:
        st = [0, 120000, deadline]
        cands = tactic_candidates(pos)
        if not cands: return None
        scored = []
        for m in cands:
            s, _ = pv_line(pos.apply(m), 2, st)
            if -s >= TACTIC_MIN[kind]: scored.append((-s, m))
        if not scored: return None
        scored.sort(key=lambda t: -t[0])
        if opts['unique'] and len(scored) > 1 and \
                scored[1][0] > scored[0][0] - 2 and scored[1][0] >= 3:
            return None
        g0, m = scored[0]
        cls = classify(pos, m)
        if cls not in ACCEPT.get(kind, {kind}): return None
        s, pv2 = pv_line(pos.apply(m), 2, st)
        return {'pos': pos, 'cls': cls, 'depth': 0, 'pv': [m]+pv2, 'gain': g0}
    except _Budget:
        return None

# ============================================================================
# 6. POSITION CONSTRUCTION
# ============================================================================
def _sprinkle(rng, bd, side, pool, count, forbid):
    empt = [(r, c) for r in range(9) for c in range(9)
            if bd[r*9+c] == 0 and all(max(abs(r-fr_), abs(c-fc)) > 1
                                      for fr_, fc in forbid)]
    rng.shuffle(empt)
    for r, c in empt[:count]:
        bd[r*9+c] = side*pool[rng.randrange(len(pool))]

def _far(rng, bd, rows, mind, away):
    cand = [(r, c) for r in rows for c in range(9)
            if bd[r*9+c] == 0 and all(max(abs(r-ar), abs(c-ac)) >= mind
                                      for ar, ac in away)]
    return rng.choice(cand) if cand else None

def _place_sente_king(rng, bd, away):
    for mind in (4, 3, 2):
        sq = _far(rng, bd, range(5, 9), mind, away)
        if sq: bd[sq[0]*9+sq[1]] = KING; return True
    return False

def _line_open(bd, r, c, kr, kc, max_block=0):
    dr = (kr > r)-(kr < r); dc = (kc > c)-(kc < c)
    if dr == 0 and dc == 0: return False
    if dr and dc and abs(kr-r) != abs(kc-c): return False
    blocks, rr, cc = 0, r+dr, c+dc
    while (rr, cc) != (kr, kc):
        if bd[rr*9+cc]:
            blocks += 1
            if blocks > max_block: return False
        rr += dr; cc += dc
    return True

def mate_cluster(rng, kind, depth, opts, knobs=None, tmpl=None):
    r = rng.random()
    if depth >= 2 and r < 0.25:
        pos = retrograde_cluster(rng, kind, depth, opts)
        if pos is not None: return pos
    elif depth >= 2 and r < 0.50:
        pos = constructive_cluster(rng, kind, depth, opts)
        if pos is not None: return pos
    bd = [0]*81
    if kind == 'corner':   kr, kc = 0, rng.choice((0, 8))
    elif kind == 'edge':   kr, kc = 0, rng.randint(0, 8)
    else: kr, kc = ((0, rng.randint(0, 8)) if rng.random() < 0.5
                    else (rng.randint(0, 3), rng.randint(0, 8)))
    bd[kr*9+kc] = -KING
    occupied = {(kr, kc)}
    if tmpl and rng.random() < 0.8:
        plant_template(rng, bd, kr, kc, tmpl, occupied)
    if kind in ('double', 'discovered') and rng.random() < 0.85:
        for _ in range(20):
            sgn = -1 if kc >= 3 else 1
            c_r, c_s = kc+3*sgn, kc+sgn
            lr, tgt_c = kr+1, (kc if kind == 'double' else kc+2*sgn)
            cells = [(kr, c_r), (kr, c_s), (lr, tgt_c)]
            if not all(0 <= x < 9 and 0 <= y < 9 for x, y in cells): continue
            if any(bd[x*9+y] for x, y in cells) or (lr, tgt_c) == (kr, kc): continue
            bd[kr*9+c_r], bd[kr*9+c_s], bd[lr*9+tgt_c] = ROOK, SILVER, -PAWN
            occupied.update(cells)
            break
    force = [KNIGHT] if kind == 'knight' else [PAWN] if kind == 'pawnmate' else []
    atk = [GOLD, GOLD, SILVER, ROOK, BISHOP, ROOK+PROMOTED, BISHOP+PROMOTED]
    if depth >= 2: atk += [KNIGHT, LANCE]
    if depth >= 3: atk += [GOLD, ROOK, BISHOP, SILVER]
    n_atk = min(7, 2 + depth + rng.randint(0, 1) +
                (1 if kind in ('corner', 'edge', 'sacrifice') else 0))
    if kind in ('knight', 'pawnmate'): n_atk = min(n_atk, 3)
    if len(occupied) > 1: n_atk = max(2, n_atk - 2)
    placed, guard = 0, 0
    while placed < n_atk and guard < 80:
        guard += 1
        pt = force.pop() if force else rng.choice(atk)
        if pt == KNIGHT:
            cands = [(kr+2, kc-1), (kr+2, kc+1)]
        elif pt == PAWN:
            cands = []
            if kr+2 < 9 and not bd[(kr+2)*9+kc] and not bd[(kr+1)*9+kc]:
                cands = [(kr+2, kc)]
            for dc2 in (-1, 1):
                c2 = kc+dc2
                if (kr+2 < 9 and 0 <= c2 < 9 and not bd[(kr+2)*9+c2]
                        and bd[(kr+1)*9+kc] == -PAWN):
                    cands.append((kr+2, c2))
        elif pt in (ROOK, ROOK+PROMOTED):
            cands = [(kr+a, kc) for a in (-3, -2, -1, 1, 2, 3) if 0 <= kr+a < 9] + \
                    [(kr, kc+b) for b in (-3, -2, -1, 1, 2, 3) if 0 <= kc+b < 9]
        elif pt in (BISHOP, BISHOP+PROMOTED):
            cands = [(kr+a, kc+a) for a in (-3, -2, -1, 1, 2, 3)] + \
                    [(kr+a, kc-a) for a in (-3, -2, -1, 1, 2, 3)]
        elif pt == LANCE:
            cands = [(kr+a, kc) for a in (1, 2, 3) if kr+a < 9]
        else:
            cands = [(kr+a, kc+b) for a in (-1, 0, 1) for b in (-1, 0, 1)
                     if (a, b) != (0, 0)]
        rng.shuffle(cands)
        for r, c in cands:
            if not (0 <= r < 9 and 0 <= c < 9): continue
            if (r, c) in occupied or bd[r*9+c]: continue
            if abs(pt) in (ROOK, BISHOP, LANCE):
                if not _line_open(bd, r, c, kr, kc, 0 if depth == 1 else 1):
                    continue
            bd[r*9+c] = pt
            occupied.add((r, c)); placed += 1
            break
    sup = knobs[0] if knobs else (2 if depth == 1 else 1)
    sp = [(kr+a, kc+b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    rng.shuffle(sp)
    for r, c in sp[:sup]:
        if 0 <= r < 9 and 0 <= c < 9 and not bd[r*9+c] and (r, c) not in occupied:
            bd[r*9+c] = GOLD; occupied.add((r, c))
    nd = knobs[1] if knobs else (rng.randint(0, 1) if depth == 1
                                 else rng.randint(1, 2))
    rng.shuffle(sp)
    for r, c in sp[:nd]:
        if 0 <= r < 9 and 0 <= c < 9 and not bd[r*9+c] and (r, c) not in occupied:
            bd[r*9+c] = -rng.choice((GOLD, SILVER, PAWN)); occupied.add((r, c))
    if not _place_sente_king(rng, bd, list(occupied)): return None
    hands = {SENTE: {}, GOTE: {}}
    if opts['drops']:
        pool_ = opts.get('hand_pool') or [GOLD, SILVER, BISHOP, ROOK, PAWN]
        if knobs:
            for _ in range(knobs[1]):
                t = rng.choice(pool_)
                hands[SENTE][t] = hands[SENTE].get(t, 0) + 1
        elif rng.random() < (0.9 if kind in ('dropmate', 'handpair') else
                             0.35 if kind == 'sacrifice' else 0.7):
            hands[SENTE][rng.choice(pool_)] = 1
            if depth >= 3 or rng.random() < 0.35:
                hands[SENTE][rng.choice(pool_)] = 1
    return Position(bd, hands, SENTE)

def tactic_cluster(rng, kind, opts):
    if kind == 'fork':
        for _ in range(40):
            bd = [0]*81
            r, c = rng.randint(1, 7), rng.randint(1, 7)
            deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
            rng.shuffle(deltas)
            picks = [(r+d[0], c+d[1]) for d in deltas
                     if 0 <= r+d[0] < 9 and 0 <= c+d[1] < 9]
            if len(picks) < 2: continue
            t1, t2 = picks[0], picks[1]
            if rng.random() < 0.5:
                bd[t1[0]*9+t1[1]] = -KING
                bd[t2[0]*9+t2[1]] = -rng.choice((GOLD, ROOK, BISHOP))
            else:
                bd[t1[0]*9+t1[1]] = -rng.choice((GOLD, SILVER, ROOK, BISHOP))
                bd[t2[0]*9+t2[1]] = -rng.choice((GOLD, ROOK, BISHOP, SILVER))
            bd[r*9+c] = KNIGHT
            if not _place_sente_king(rng, bd, [(r, c), t1, t2]): continue
            return Position(bd, {SENTE: {}, GOTE: {}}, SENTE)
        return None
    if kind == 'skewer':
        for _ in range(40):
            bd = [0]*81
            kr, kc = rng.randint(1, 7), rng.randint(1, 6)
            if rng.random() < 0.5:
                sgn = rng.choice((-1, 1))
                ksq, tro = (kr, kc), (kr, kc+2*sgn)
                land, start = (kr, kc-sgn), (kr-1, kc-sgn)
            else:
                sgn = rng.choice((-1, 1))
                ksq, tro = (kr, kc), (kr+2*sgn, kc)
                land, start = (kr-sgn, kc), (kr-sgn, kc-1)
            cells = [ksq, tro, land, start]
            if not all(0 <= x < 9 and 0 <= y < 9 for x, y in cells): continue
            if len(set(cells)) < 4 or any(bd[x*9+y] for x, y in cells): continue
            bd[ksq[0]*9+ksq[1]], bd[tro[0]*9+tro[1]], bd[start[0]*9+start[1]] = \
                -KING, -ROOK, ROOK
            if not _place_sente_king(rng, bd, cells): continue
            _sprinkle(rng, bd, GOTE, [PAWN], 1, cells)
            return Position(bd, {SENTE: {}, GOTE: {}}, SENTE)
        return None
    if kind == 'pin':
        for _ in range(40):
            bd = [0]*81
            kr, kc = rng.randint(0, 4), rng.randint(0, 8)
            dirs = [d for d in (_ORTH + _DIAG)
                    if 0 <= kr+2*d[0] < 9 and 0 <= kc+2*d[1] < 9]
            if not dirs: continue
            dr, dc = rng.choice(dirs)
            v1, v2 = (kr+dr, kc+dc), (kr+2*dr, kc+2*dc)
            cells = [(kr, kc), v1, v2]
            if any(bd[x*9+y] for x, y in cells): continue
            bd[kr*9+kc] = -KING
            bd[v1[0]*9+v1[1]] = -rng.choice((SILVER, GOLD))
            bd[v2[0]*9+v2[1]] = ROOK if (dr, dc) in _ORTH else BISHOP
            nbs = [(v1[0]+a, v1[1]+b) for a in (-1, 0, 1) for b in (-1, 0, 1)
                   if (a or b) and 0 <= v1[0]+a < 9 and 0 <= v1[1]+b < 9
                   and (v1[0]+a, v1[1]+b) not in cells
                   and not bd[(v1[0]+a)*9+v1[1]+b]]
            if not nbs: continue
            cr, cc = rng.choice(nbs)
            bd[cr*9+cc] = GOLD; cells.append((cr, cc))
            if not _place_sente_king(rng, bd, cells): continue
            _sprinkle(rng, bd, GOTE, [PAWN], 1, cells)
            return Position(bd, {SENTE: {}, GOTE: {}}, SENTE)
        return None
    if kind == 'promo':
        for _ in range(40):
            bd = [0]*81
            c = rng.randint(0, 8); r = rng.choice((0, 1, 2))
            bd[r*9+c] = -rng.choice((SILVER, GOLD))
            r0 = rng.choice([rr for rr in range(3, 7) if rr > r+1])
            bd[r0*9+c] = ROOK
            kc2 = rng.choice([cc for cc in range(9) if abs(cc-c) >= 2])
            kr2 = rng.choice([rr for rr in range(4, 8) if not bd[rr*9+kc2]])
            bd[kr2*9+kc2] = -KING
            cells = [(r, c), (r0, c), (kr2, kc2)]
            if not _place_sente_king(rng, bd, cells): continue
            _sprinkle(rng, bd, GOTE, [PAWN], 1, cells)
            return Position(bd, {SENTE: {}, GOTE: {}}, SENTE)
        return None
    if kind == 'discattack':
        for _ in range(40):
            bd = [0]*81
            left = rng.random() < 0.5
            c = 0 if left else 8
            r = rng.randint(0, 4)
            inw = 1 if left else -1
            g, b, s = (r, c), (r, c+2*inw), (r, c+4*inw)
            cells = [g, b, s]
            if any(not (0 <= x < 9 and 0 <= y < 9) for x, y in cells): continue
            if any(bd[x*9+y] for x, y in cells): continue
            bd[g[0]*9+g[1]], bd[b[0]*9+b[1]], bd[s[0]*9+s[1]] = -ROOK, GOLD, ROOK
            for rr in (r-1, r+1):
                if 0 <= rr < 9 and not bd[rr*9+c]:
                    bd[rr*9+c] = GOLD; cells.append((rr, c))
            si = r+1 if r+1 < 9 else r-1
            if 0 <= si < 9 and not bd[si*9+c+inw]:
                bd[si*9+c+inw] = SILVER; cells.append((si, c+inw))
            if not _place_sente_king(rng, bd, cells): continue
            _sprinkle(rng, bd, GOTE, [PAWN], 1, cells)
            return Position(bd, {SENTE: {}, GOTE: {}}, SENTE)
        return None
    if kind == 'winpiece':
        for _ in range(40):
            bd = [0]*81
            r, c = rng.randint(1, 6), rng.randint(0, 8)
            bd[r*9+c] = -rng.choice((GOLD, SILVER, ROOK, BISHOP))
            spots = [(r+a, c+b) for a in (-1, 0, 1) for b in (-1, 0, 1)
                     if (a or b) and 0 <= r+a < 9 and 0 <= c+b < 9]
            rng.shuffle(spots)
            if not spots: continue
            ar, ac = spots[0]
            bd[ar*9+ac] = rng.choice((GOLD, SILVER))
            for rr in range(9):
                if rr != r and not bd[rr*9+c] and abs(rr-r) >= 2:
                    bd[rr*9+c] = ROOK; break
            if not _place_sente_king(rng, bd, [(r, c), (ar, ac)]): continue
            _sprinkle(rng, bd, GOTE, [PAWN], 1, [(r, c)])
            return Position(bd, {SENTE: {}, GOTE: {}}, SENTE)
        return None
    return None

def quick_screen(pos, kind):
    if kind == 'discattack': return bool(blocker_squares(pos))
    if kind == 'winpiece':
        for sq in range(81):
            q = pos.bd[sq]
            if q < 0 and piece_value(q) >= 5:
                r, c = divmod(sq, 9)
                if attackers_of(pos, r, c, SENTE): return True
        return False
    return True

# ============================================================================
# 6b. GENERATION ALGORITHMS  (naturalness, rescue, bandit, MAP-Elites,
#     retrograde/constructive clusters, CP-SAT proposer, mate race)
# ============================================================================
def plausibility_score(pos, max_moves=9):
    ok = total = 0
    for sq in range(81):
        p = pos.bd[sq]
        if not p: continue
        total += 1
        side = SENTE if p > 0 else GOTE
        r, _c = divmod(sq, 9)
        b = base_of(p)
        home = 6 if side == SENTE else 2
        good = abs(r - home) <= max_moves
        if abs(p) > KING and b in (PAWN, LANCE, KNIGHT, SILVER):
            edge = 2 if side == SENTE else 6
            good = good and (r <= edge + 1 if side == SENTE else r >= edge - 1)
        if b in (GOLD, KING): good = True
        if good: ok += 1
    for s in (SENTE, GOTE):
        n = sum(pos.hands[s].values()); total += n; ok += n
    return ok / total if total else 1.0

def naturalness(pos):
    pen = 0.0
    pen += 1.2 * sum(1 for sq in range(36) if pos.bd[sq] > 0)
    for sq in range(81):
        p = pos.bd[sq]
        if p < 0 and abs(p) != KING:
            r, c = divmod(sq, 9)
            if not any(0 <= r+a < 9 and 0 <= c+b < 9
                       and pos.bd[(r+a)*9+c+b] < 0
                       for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b):
                pen += 0.8
    pen += 0.5 * sum(pos.hands[GOTE].values())
    return pen

def mutate_position(rng, pos):
    bd = pos.bd[:]
    mv = rng.random()
    if mv < 0.30:
        k = pos.king(GOTE)
        if not k: return None
        spots = [(k[0]+a, k[1]+b) for a in range(-2, 3) for b in range(-2, 3)
                 if 0 <= k[0]+a < 9 and 0 <= k[1]+b < 9
                 and not bd[(k[0]+a)*9+k[1]+b]]
        if not spots: return None
        r, c = rng.choice(spots)
        bd[r*9+c] = -rng.choice((GOLD, SILVER, PAWN))
    elif mv < 0.55:
        spots = [i for i in range(81) if bd[i] and abs(bd[i]) != KING]
        if not spots: return None
        bd[rng.choice(spots)] = 0
    elif mv < 0.75:
        spots = [i for i in range(81) if bd[i] and abs(bd[i]) != KING]
        if not spots: return None
        i = rng.choice(spots); r, c = divmod(i, 9)
        a, b = rng.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        rr, cc = r+a, c+b
        if not (0 <= rr < 9 and 0 <= cc < 9) or bd[rr*9+cc]: return None
        bd[rr*9+cc] = bd[i]; bd[i] = 0
    elif mv < 0.90:
        spots = [i for i in range(81) if bd[i] and abs(bd[i]) != KING]
        if not spots: return None
        i = rng.choice(spots)
        swaps = {GOLD: SILVER, SILVER: GOLD, ROOK: BISHOP, BISHOP: ROOK}
        b0 = base_of(bd[i])
        if b0 not in swaps: return None
        bd[i] = (1 if bd[i] > 0 else -1) * \
                (swaps[b0] + (PROMOTED if abs(bd[i]) > KING else 0))
    else:
        spots = [i for i in range(81) if bd[i] < 0 and abs(bd[i]) != KING]
        if not spots: return None
        i = rng.choice(spots); t = base_of(bd[i])
        bd[i] = 0
        h = dict(pos.hands[GOTE]); h[t] = h.get(t, 0) + 1
        return Position(bd, {SENTE: pos.hands[SENTE], GOTE: h}, pos.side)
    return Position(bd, pos.hands, pos.side)

class NearMissPool:
    def __init__(self, cap=24):
        self.items = []; self.cap = cap
    def add(self, pos, kind, d, reason):
        pr = {'dual': 3, 'deeper': 2, 'unclassified': 1, 'no_mate': 0}.get(reason, 0)
        self.items.append((pr, pos, kind, d, reason))
        self.items.sort(key=lambda t: -t[0])
        del self.items[self.cap:]

BANDIT_ARMS = [(d, h) for d in (0, 1, 2) for h in (0, 1, 2)]

class KnobBandit:
    C = 1.2
    def __init__(self): self.arms = {}; self.total = 0
    def select(self, rng):
        untried = [a for a in BANDIT_ARMS if a not in self.arms]
        if untried: return rng.choice(untried)
        self.total += 1
        best, bv = None, -1.0
        for a, (n, s) in self.arms.items():
            v = s/n + self.C*math.sqrt(math.log(max(1, self.total))/n)
            if v > bv: best, bv = a, v
        return best
    def update(self, arm, ok):
        n, s = self.arms.get(arm, (0, 0.0))
        self.arms[arm] = (n+1, s + (1.0 if ok else 0.0))

def arm_to_knobs(arm, depth):
    d, h = arm
    if depth and depth >= 2: d = max(1, d)
    return (d, h)

def template_sig(pos, pv):
    if not pv: return None
    sig = key_signature(pos, pv, '')
    return sig[1:] if sig else None

def mine_templates(entries, cap=48):
    from collections import Counter
    c = Counter(tuple(e['tmpl']) for e in entries if e.get('tmpl'))
    return [t for t, n in c.most_common(cap) if n >= 2]

def plant_template(rng, bd, kr, kc, tmpl, occupied):
    tpt, dr, dc = tmpl[0], tmpl[3], tmpl[4]
    cands = [(r, c) for r in range(9) for c in range(9)
             if not bd[r*9+c] and (r, c) not in occupied
             and abs(r-kr) in range(max(1, dr-1), dr+2)
             and abs(c-kc) in range(max(1, dc-1), dc+2)]
    rng.shuffle(cands)
    for r, c in cands[:6]:
        if abs(tpt) in (ROOK, BISHOP, LANCE) and \
                not _line_open(bd, r, c, kr, kc, 1):
            continue
        bd[r*9+c] = tpt; occupied.add((r, c))
        return True
    return False

MAP_QUOTA = 16

def niche_of(puz):
    dcount = 0
    if isinstance(puz.tree, dict) and 'replies' in puz.tree:
        dcount = len(puz.tree['replies'])
    return (puz.depth, puz.diff // 25, min(dcount, 3))

def retrograde_cluster(rng, kind, depth, opts):
    """Mate-first composition: dense mate-1 core, then un-play (depth-1)
    attacker/defender pairs; the solver does the real verification."""
    for _ in range(12):
        bd = [0]*81
        kr, kc = 0, rng.randint(0, 8)
        bd[kr*9+kc] = -KING
        sp = [(kr+a, kc+b) for a in (-1, 0, 1) for b in (-1, 0, 1)
              if (a, b) != (0, 0) and 0 <= kr+a < 9]
        rng.shuffle(sp)
        for r, c in sp:
            if not bd[r*9+c]:
                bd[r*9+c] = -rng.choice((GOLD, SILVER, PAWN))
        if kr + 1 < 9 and not bd[(kr+1)*9+kc]:
            bd[(kr+1)*9+kc] = rng.choice((GOLD, SILVER))
        for _i in range(depth - 1):
            sente = [i for i in range(81) if bd[i] > 0 and abs(bd[i]) != KING]
            if not sente: break
            src = rng.choice(sente); r, c = divmod(src, 9)
            dests = [(rr, cc) for rr in range(9) for cc in range(9)
                     if not bd[rr*9+cc]
                     and 1 <= max(abs(rr-r), abs(cc-c)) <= 4
                     and max(abs(rr-kr), abs(cc-kc)) >= 2]
            if not dests: break
            rr, cc = rng.choice(dests)
            bd[rr*9+cc] = bd[src]; bd[src] = 0
            defs = [i for i in range(81) if bd[i] < 0 and abs(bd[i]) != KING]
            if defs:
                dsrc = rng.choice(defs)
                ddest = [(a, b) for a in range(max(0, kr-2), min(9, kr+3))
                         for b in range(max(0, kc-2), min(9, kc+3))
                         if not bd[a*9+b] and (a, b) != (kr, kc)]
                if ddest:
                    a, b = rng.choice(ddest)
                    bd[a*9+b] = bd[dsrc]; bd[dsrc] = 0
        if not _place_sente_king(rng, bd, [(kr, kc)]): continue
        hands = {SENTE: {}, GOTE: {}}
        if opts['drops'] and rng.random() < 0.6:
            hands[SENTE][rng.choice((GOLD, SILVER, ROOK, BISHOP))] = 1
        pos = normalize_position(Position(bd, hands, SENTE), rng)
        if pos and not pos.in_check(GOTE) and not pos.in_check(SENTE):
            return pos
    return None

def constructive_cluster(rng, kind, depth, opts):
    """Constraint-ordered placement: lines pruned at placement time."""
    for _outer in range(6):
        bd = [0]*81
        kr, kc = 0, rng.randint(0, 8)
        bd[kr*9+kc] = -KING
        occupied = {(kr, kc)}
        pool = [GOLD, SILVER, ROOK, BISHOP] + \
               ([KNIGHT, LANCE] if depth >= 2 else [])
        placed, ok = 0, True
        for _step in range(min(7, 2 + depth + rng.randint(0, 1))):
            pt = rng.choice(pool)
            if pt == KNIGHT:
                cands = [(kr+2, kc-1), (kr+2, kc+1)]
            elif pt == LANCE:
                cands = [(kr+a, kc) for a in (1, 2, 3) if kr+a < 9]
            elif pt == ROOK:
                cands = [(kr+a, kc) for a in (-3, -2, -1, 1, 2, 3)
                         if 0 <= kr+a < 9] + \
                        [(kr, kc+b) for b in (-3, -2, -1, 1, 2, 3)
                         if 0 <= kc+b < 9]
            elif pt == BISHOP:
                cands = [(kr+a, kc+a) for a in (-3, -2, -1, 1, 2, 3)] + \
                        [(kr+a, kc-a) for a in (-3, -2, -1, 1, 2, 3)]
            else:
                cands = [(kr+a, kc+b) for a in (-1, 0, 1) for b in (-1, 0, 1)
                         if (a, b) != (0, 0)]
            rng.shuffle(cands)
            done = False
            for r, c in cands:
                if not (0 <= r < 9 and 0 <= c < 9): continue
                if (r, c) in occupied or bd[r*9+c]: continue
                if pt in (ROOK, BISHOP, LANCE) and \
                        not _line_open(bd, r, c, kr, kc, depth - 1):
                    continue
                bd[r*9+c] = pt; occupied.add((r, c)); placed += 1
                done = True; break
            if not done: ok = False; break
        if not ok or placed < 2: continue
        sup = 1 if depth >= 2 else 2
        sp = [(kr+a, kc+b) for a in (-1, 0, 1) for b in (-1, 0, 1)
              if (a, b) != (0, 0)]
        rng.shuffle(sp)
        for r, c in sp[:sup]:
            if 0 <= r < 9 and not bd[r*9+c] and (r, c) not in occupied:
                bd[r*9+c] = -rng.choice((GOLD, SILVER, PAWN))
                occupied.add((r, c))
        if not _place_sente_king(rng, bd, list(occupied)): continue
        pos = normalize_position(Position(bd, {SENTE: {}, GOTE: {}}, SENTE), rng)
        if pos and not pos.in_check(GOTE) and not pos.in_check(SENTE):
            return pos
    return None

try:
    from ortools.sat.python import cp_model as _cpm
    HAS_CP = True
except Exception:
    HAS_CP = False

def cpsat_propose(rng, depth):
    """Structured diverse placement; the mate oracle stays external."""
    if not HAS_CP: return None
    m = _cpm.CpModel()
    kr, kc = 0, rng.randint(0, 8)
    cells = [(r, c) for r in range(5) for c in range(9) if (r, c) != (kr, kc)]
    types = [GOLD, SILVER, ROOK, BISHOP] + \
            ([KNIGHT, LANCE] if depth >= 2 else [])
    x = {}
    for rc in cells:
        for t in types:
            for s in (1, -1):
                x[(rc, t, s)] = m.NewBoolVar('')
        m.Add(sum(x[(rc, t, s)] for t in types for s in (1, -1)) <= 1)
    defs = [x[(rc, t, -1)] for rc in cells for t in (GOLD, SILVER)]
    m.Add(sum(defs) >= 1); m.Add(sum(defs) <= 3)
    atk = [x[(rc, t, 1)] for rc in cells for t in types]
    m.Add(sum(atk) >= 3); m.Add(sum(atk) <= 6)
    m.Maximize(sum(rng.randint(0, 3) * v for v in x.values()))
    sol = _cpm.CpSolver()
    sol.parameters.max_time_in_seconds = 1.0
    sol.parameters.random_seed = rng.randrange(10**6)
    if sol.Solve(m) not in (_cpm.OPTIMAL, _cpm.FEASIBLE): return None
    bd = [0]*81
    bd[kr*9+kc] = -KING
    for rc in cells:
        for t in types:
            for s in (1, -1):
                if sol.Value(x[(rc, t, s)]): bd[rc[0]*9+rc[1]] = t*s
    return bd

def add_gote_threat(rng, pos, d):
    """Mate-race: give Gote a real mating threat without checking Sente."""
    bd = pos.bd[:]
    sr, sc = pos.king(SENTE)
    gtc = global_type_counts(pos)
    pool = [t for t in (ROOK, BISHOP, GOLD, SILVER)
            for _ in range(max(0, INITIAL_TOTAL[t] - gtc.get(t, 0)))]
    rng.shuffle(pool)
    spots = [(sr+a, sc+b) for a in range(-3, 4) for b in range(-3, 4)
             if 0 <= sr+a < 9 and 0 <= sc+b < 9
             and not bd[(sr+a)*9+sc+b] and (a or b)]
    rng.shuffle(spots)
    placed, want = 0, rng.randint(2, 3)
    for r, c in spots:
        if placed >= want or not pool: break
        bdt = bd[:]; bdt[r*9+c] = -pool.pop()
        t = Position(bdt, pos.hands, SENTE)
        if t.in_check(SENTE): continue
        bd = bdt; placed += 1
    return Position(bd, pos.hands, SENTE)

# ============================================================================
# 6c. GENRE GENERATORS
# ============================================================================
def vmirror_pos(pos):
    bd = [0]*81
    for sq in range(81):
        r, c = divmod(sq, 9)
        bd[sq] = -pos.bd[(8-r)*9 + c]
    return Position(bd, {SENTE: dict(pos.hands[GOTE]),
                         GOTE: dict(pos.hands[SENTE])}, -pos.side)

def vmirror_move(m):
    if m.fr < 0:
        return Move(-1, -1, 8-m.tr, m.tc, -m.piece, m.promo)
    return Move(8-m.fr, m.fc, 8-m.tr, m.tc, -m.piece, m.promo)

def vmirror_tree(node):
    if node is None: return None
    return {'move': vmirror_move(node['move']),
            'after': vmirror_pos(node['after']),
            'replies': {vmirror_move(rp): {'pos': vmirror_pos(e['pos']),
                                           'sub': vmirror_tree(e['sub'])}
                        for rp, e in node['replies'].items()}}

def make_gote_mate(rng, d, opts, deadline):
    p = make_mate_puzzle(rng, f'mate{d}', opts, deadline)
    if p is None: return None
    npos = vmirror_pos(p.pos)
    if position_violations(npos): return None
    cur, npv = npos, []
    for m in p.pv:
        vm = vmirror_move(m)
        if vm not in cur.legal_moves(): return None
        npv.append(vm); cur = cur.apply(vm)
    pz = Puzzle(0, npos, 'gote', d, npv, vmirror_tree(p.tree), p.stars,
                diff=p.diff)
    pz.plaus = plausibility_score(npos)
    return pz

PARRY_MARGIN = 3

def find_parry(pos, deadline):
    try:
        st = [0, 120000, deadline]
        moves = pos.legal_moves()
        if not moves or len(moves) > 44: return None
        scored = []
        for m in moves:
            s, _ = pv_line(pos.apply(m), 2, st)
            scored.append((s, m))
        scored.sort(key=lambda t: t[0])
        s0, m0 = scored[0]
        if s0 > 0: return None
        if len(scored) > 1 and scored[1][0] - s0 < PARRY_MARGIN: return None
        s, pv2 = pv_line(pos.apply(m0), 2, st)
        margin = scored[1][0] - s0 if len(scored) > 1 else PARRY_MARGIN
        return {'pos': pos, 'cls': 'parry', 'depth': 0, 'pv': [m0]+pv2,
                'gain': int(round(s0)), 'margin': margin}
    except _Budget:
        return None

def make_parry_puzzle(rng, opts, deadline):
    while time.time() < deadline:
        base = tactic_cluster(rng, 'winpiece', opts)
        if base is None: continue
        pos = normalize_position(Position(base.bd, base.hands, GOTE), rng)
        if pos is None or position_violations(pos): continue
        if pos.in_check(GOTE) or pos.in_check(SENTE): continue
        r = find_parry(pos, deadline)
        if r:
            pz = Puzzle(0, r['pos'], 'parry', 0, r['pv'], None, 3, gain=r['gain'])
            pz.diff = min(99, 24 + 6*int(r['margin']))
            pz.stars = stars_from_diff(pz.diff)
            pz.plaus = plausibility_score(pos)
            return pz
    return None

QUIET_MIN = 3

def find_quiet(pos, deadline):
    try:
        st = [0, 150000, deadline]
        scored = []
        for m in pos.legal_moves():
            ch = pos.apply(m)
            if ch.in_check(): continue
            s, _ = pv_line(ch, 2, st)
            scored.append((-s, m))
        if not scored: return None
        scored.sort(key=lambda t: -t[0])
        g0, m0 = scored[0]
        if g0 < QUIET_MIN: return None
        if len(scored) > 1 and scored[1][0] > 0: return None
        s, pv2 = pv_line(pos.apply(m0), 2, st)
        return {'pos': pos, 'cls': 'quiet', 'depth': 0, 'pv': [m0]+pv2, 'gain': g0}
    except _Budget:
        return None

def make_quiet_puzzle(rng, opts, deadline):
    while time.time() < deadline:
        base = tactic_cluster(rng, 'winpiece', opts)
        if base is None: continue
        pos = normalize_position(base, rng)
        if pos is None or position_violations(pos): continue
        if pos.in_check(GOTE) or pos.in_check(SENTE): continue
        if abs(material_score(pos)) > 8: continue
        r = find_quiet(pos, deadline)
        if r:
            pz = Puzzle(0, r['pos'], 'quiet', 0, r['pv'], None, 3, gain=r['gain'])
            pz.diff = min(99, 20 + 2*r['gain'])
            pz.stars = stars_from_diff(pz.diff)
            pz.plaus = plausibility_score(pos)
            return pz
    return None

PROMO_GAP = 4

def find_promo_choice(pos, deadline):
    try:
        st = [0, 150000, deadline]
        scored = []
        for m in pos.legal_moves():
            s, _ = pv_line(pos.apply(m), 2, st)
            scored.append((-s, m))
        if not scored: return None
        scored.sort(key=lambda t: -t[0])
        g0, m0 = scored[0]
        if not m0.promo: return None
        twins = [g for g, m in scored[1:]
                 if not m.promo and (m.fr, m.fc, m.tr, m.tc) ==
                    (m0.fr, m0.fc, m0.tr, m0.tc)]
        if not twins or g0 - twins[0] < PROMO_GAP: return None
        others = [g for g, m in scored[1:]
                  if (m.fr, m.fc, m.tr, m.tc) != (m0.fr, m0.fc, m0.tr, m0.tc)]
        if others and others[0] > 0: return None
        s, pv2 = pv_line(pos.apply(m0), 2, st)
        return {'pos': pos, 'cls': 'promoch', 'depth': 0, 'pv': [m0]+pv2,
                'gain': g0, 'gap': g0 - twins[0]}
    except _Budget:
        return None

def make_promoch_puzzle(rng, opts, deadline):
    while time.time() < deadline:
        base = tactic_cluster(rng, 'promo', opts)
        if base is None: continue
        pos = normalize_position(base, rng)
        if pos is None or position_violations(pos): continue
        if pos.in_check(GOTE) or pos.in_check(SENTE): continue
        r = find_promo_choice(pos, deadline)
        if r:
            pz = Puzzle(0, r['pos'], 'promoch', 0, r['pv'], None, 3, gain=r['gain'])
            pz.diff = min(99, 22 + 4*int(r['gap']))
            pz.stars = stars_from_diff(pz.diff)
            pz.plaus = plausibility_score(pos)
            return pz
    return None

def make_findall_puzzle(rng, opts, deadline):
    solver = MateSolver(opts['pawn_mate'], deadline)
    while time.time() < deadline:
        pos = mate_cluster(rng, 'mate1', 1, opts)
        if pos is None: continue
        pos = normalize_position(pos, rng)
        if pos is None or pos.in_check(GOTE) or pos.in_check(SENTE): continue
        try:
            ks = solver.mates(pos, 1)
        except _Budget:
            continue
        if not (2 <= len(ks) <= 4): continue
        k0 = ks[0]
        try:
            node = solution_tree(solver, pos, 1, k0)
        except _Budget:
            continue
        if node is None: continue
        pz = Puzzle(0, pos, 'findall', 1, [k0], node, 3, keys=list(ks[1:5]))
        pz.diff = min(99, diff_score(pz) + 8*len(pz.keys))
        pz.stars = stars_from_diff(pz.diff)
        pz.plaus = plausibility_score(pos)
        return pz
    return None

# ============================================================================
# 7. PUZZLE GENERATION (top level)
# ============================================================================
CAT_TITLE = {'mate1': 'Mate in 1', 'mate2': 'Mate in 2', 'mate3': 'Mate in 3',
             'mate4': 'Mate in 4', 'corner': 'Corner mate', 'edge': 'Edge mate',
             'double': 'Double check', 'discovered': 'Discovered check',
             'knight': 'Knight mate', 'pawnmate': 'Pawn mate',
             'dropmate': 'Drop mate', 'sacrifice': 'Sacrifice mate',
             'handpair': 'Gold & Silver hand mate', 'gote': 'Gote to mate △',
             'findall': 'Find ALL mates', 'race': 'Mate race',
             'fork': 'Fork', 'skewer': 'Skewer', 'pin': 'Pin & win',
             'promo': 'Promotion', 'promoch': 'Promotion choice',
             'quiet': 'Best move (quiet key)', 'parry': 'Find the defense',
             'discattack': 'Discovered attack', 'winpiece': 'Win a piece'}
BANNER = {**{k: 'CHECKMATE!' for k in ('mate1', 'mate2', 'mate3', 'mate4')},
          'corner': 'CORNER MATE!', 'edge': 'EDGE MATE!', 'double': 'DOUBLE CHECK!',
          'discovered': 'DISCOVERED CHECK!', 'knight': 'KNIGHT MATE!',
          'pawnmate': 'PAWN MATE!', 'dropmate': 'DROP MATE!',
          'sacrifice': 'SACRIFICE!', 'handpair': 'GOLD-SILVER MATE!',
          'gote': 'GOTE MATES!', 'findall': 'CHECKMATE!',
          'race': 'MATE RACE!', 'fork': 'DOUBLE ATTACK!',
          'skewer': 'SKEWERED!', 'pin': 'PINNED & WON!', 'promo': 'PROMOTE!',
          'promoch': 'PROMOTION WINS!', 'quiet': 'BEST MOVE!',
          'parry': 'SURVIVES!', 'discattack': 'DISCOVERED ATTACK!',
          'winpiece': 'WIN MATERIAL!'}
NOTE = {'mate1': 'Nailed it — mate in one.', 'mate2': 'The key — mate follows.',
        'mate3': 'Deep calculation pays off.', 'mate4': 'Four moves deep — brilliant.',
        'corner': 'Cornered — nowhere to run.', 'edge': 'No escape along the edge.',
        'double': 'Two pieces check at once!', 'discovered': 'The hidden piece strikes.',
        'knight': 'The knight strikes unseen.', 'pawnmate': 'Even a pawn can mate.',
        'dropmate': 'A piece from the hand ends it.',
        'sacrifice': 'The piece hangs — the mate does not.',
        'handpair': 'Only golds and silvers in hand.',
        'gote': 'Same puzzle, other side — △ mates.',
        'findall': 'Not one — every mate-in-one.',
        'race': "Both kings are under threat — Sente's is faster.",
        'fork': 'Two targets, one move.', 'skewer': 'The king must step aside.',
        'pin': 'Pinned means prey.', 'promo': 'Promote and profit.',
        'promoch': 'To promote or not — that is the puzzle.',
        'quiet': 'No check, no capture — just the best move.',
        'parry': 'One move survives — all the rest lose.',
        'discattack': 'The front piece unmasked it.',
        'winpiece': 'Material decided by force.'}
TITLES = {'fork': 'DOUBLE ATTACK', 'skewer': 'SKEWER', 'pin': 'PIN & WIN',
          'promo': 'PROMOTION', 'double': 'DOUBLE CHECK',
          'discovered': 'DISCOVERED CHECK', 'edge': 'EDGE MATE',
          'knight': 'KNIGHT MATE', 'pawnmate': 'PAWN MATE', 'dropmate': 'DROP MATE',
          'sacrifice': 'SACRIFICE MATE', 'discattack': 'DISCOVERED ATTACK',
          'winpiece': 'WIN A PIECE', 'gote': 'GOTE TO MATE', 'parry': 'ONLY DEFENSE',
          'quiet': 'BEST MOVE', 'promoch': 'PROMOTION CHOICE',
          'handpair': 'GOLD & SILVER MATE', 'findall': 'FIND ALL MATES'}
MIX = (['mate1']*3 + ['mate2']*3 + ['mate3']*2 + ['mate4'] + ['corner', 'edge'] +
       ['double', 'discovered', 'knight', 'pawnmate', 'dropmate', 'sacrifice'] +
       ['fork', 'fork', 'skewer', 'pin', 'promo', 'discattack', 'winpiece'] +
       ['gote']*4 + ['parry', 'quiet', 'promoch', 'handpair', 'findall', 'race'])
KIND_DEPTH = {'mate1': 1, 'mate2': 2, 'mate3': 3, 'mate4': 4}
TIMEOUT = {'mate1': 8, 'mate2': 14, 'mate3': 30, 'mate4': 45, 'corner': 15,
           'edge': 15, 'double': 15, 'discovered': 15, 'knight': 18,
           'pawnmate': 18, 'dropmate': 15, 'sacrifice': 15, 'handpair': 15,
           'gote': 40, 'findall': 12, 'race': 25, 'fork': 10, 'skewer': 10,
           'pin': 10, 'promo': 10, 'promoch': 15, 'quiet': 15, 'parry': 15,
           'discattack': 15, 'winpiece': 12}
TACTIC_KINDS = ('fork', 'skewer', 'pin', 'promo', 'discattack', 'winpiece')

def defside(puz):
    return GOTE if puz.pos.side == SENTE else SENTE

@dataclass
class Puzzle:
    index: int; pos: Position; cls: str; depth: int
    pv: list; tree: object; stars: int
    gain: int = 0; nodes: int = 0; secs: float = 0.0; diff: int = 0
    plaus: float = 1.0
    keys: list = None
    threat_m: int = 0
    @property
    def banner(self): return BANNER[self.cls]
    @property
    def note_en(self): return NOTE[self.cls]
    @property
    def label(self):
        base = CAT_TITLE[self.cls] if self.depth == 0 else \
            f'{CAT_TITLE[self.cls]} · {len(self.pv)} plies'
        return f'#{self.index:02d} · {base} · {"★"*self.stars} · D{self.diff}'

def stars_from_diff(d): return max(1, min(5, 1 + d // 20))

def diff_score(puz):
    if puz.depth:
        s = 10*puz.depth
        s += 6 if any(m.fr < 0 for m in puz.pv) else 0
        s += 6 if any(m.promo for m in puz.pv) else 0
        s += 8 if puz.cls == 'sacrifice' else 0
        try: s += min(12, 3*len(check_evasions(puz.pos.apply(puz.pv[0]))))
        except Exception: pass
        return min(99, s)
    s = 12 + min(24, 2*puz.gain)
    s += 8 if any(m.promo for m in puz.pv) else 0
    s += 6 if puz.cls in ('skewer', 'discattack', 'fork') else 0
    return min(99, s)

def validate_puzzle(puz):
    try:
        if position_violations(puz.pos): return False
        pos = puz.pos
        for m in puz.pv:
            if m not in pos.legal_moves(): return False
            pos = pos.apply(m)
        if puz.depth:
            if not pos.in_check() or pos.legal_moves(): return False
            s = MateSolver(False, time.time()+6)
            if s.mates_fast(puz.pos, puz.depth-1): return False
            if not s.mates_fast(puz.pos, puz.depth): return False
            if puz.depth <= 2:
                ch = puz.pos.apply(puz.pv[0])
                if not ch.in_check(): return False
                if puz.depth == 1:
                    if naive_legal_moves(ch): return False
                else:
                    for rp in naive_legal_moves(ch):
                        c2 = ch.apply(rp)
                        if not c2.in_check(): return False
                        if not any(c2.apply(mm).in_check()
                                   and not c2.apply(mm).legal_moves()
                                   for mm in c2.legal_moves()):
                            return False
            if puz.depth >= 3:
                s2 = MateSolver(False, time.time()+10, node_cap=600000,
                                persistent=False, mode=0xC0FFEE + puz.depth)
                if not s2.mates_fast(puz.pos, puz.depth): return False
                if s2.mates_fast(puz.pos, puz.depth-1): return False
        else:
            if puz.cls != 'parry':
                if material_score(pos) - material_score(puz.pos) < \
                        TACTIC_MIN.get(puz.cls, 3): return False
        if puz.cls == 'findall' and puz.keys:
            for k in puz.keys:
                ch = puz.pos.apply(k)
                if not ch.in_check() or ch.legal_moves(): return False
        return True
    except Exception:
        return False

def _mate_ok(solver, pos, d, kind, opts):
    try:
        if d > 1 and solver.mates_fast(pos, d-1): return False
        ks = solver.mates(pos, d)
    except _Budget:
        return False
    if not ks: return False
    if opts['unique'] and len(ks) > 1: return False
    if kind == 'dropmate' and not any(m.fr < 0 for m in ks): return False
    return True

def sa_minimize(solver, rng, pos, d, kind, opts, deadline):
    def energy(p):
        pc = sum(1 for x in p.bd if x and abs(x) != KING) + \
             sum(sum(h.values()) for h in p.hands.values())
        return pc + 0.35 * naturalness(p)
    best = cur = pos; best_e = cur_e = energy(pos)
    T, steps = 1.0, 0
    while steps < 60 and time.time() < deadline:
        steps += 1; T *= 0.94
        cand, mv = None, rng.random()
        if mv < 0.40:
            spots = [i for i in range(81) if cur.bd[i] and abs(cur.bd[i]) != KING]
            if spots:
                i = rng.choice(spots); nb = cur.bd[:]; nb[i] = 0
                cand = Position(nb, cur.hands, cur.side)
        elif mv < 0.60:
            pairs = {ROOK: GOLD, BISHOP: GOLD, SILVER: KNIGHT, GOLD: SILVER}
            spots = [i for i in range(81)
                     if cur.bd[i] > 0 and base_of(cur.bd[i]) in pairs]
            if spots:
                i = rng.choice(spots); nb = cur.bd[:]
                nb[i] = pairs[base_of(cur.bd[i])]
                cand = Position(nb, cur.hands, cur.side)
        elif mv < 0.80:
            spots = [i for i in range(81) if cur.bd[i] and abs(cur.bd[i]) != KING]
            if spots:
                i = rng.choice(spots); r, c = divmod(i, 9)
                a, b = rng.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
                rr, cc = r+a, c+b
                if 0 <= rr < 9 and 0 <= cc < 9 and not cur.bd[rr*9+cc]:
                    nb = cur.bd[:]; nb[rr*9+cc] = cur.bd[i]; nb[i] = 0
                    cand = Position(nb, cur.hands, cur.side)
        else:
            spots = [i for i in range(81) if cur.bd[i] < 0 and abs(cur.bd[i]) != KING]
            if spots:
                i = rng.choice(spots); t = base_of(cur.bd[i])
                nb = cur.bd[:]; nb[i] = 0
                h = dict(cur.hands[GOTE]); h[t] = h.get(t, 0) + 1
                cand = Position(nb, {SENTE: cur.hands[SENTE], GOTE: h}, cur.side)
        if cand is None: continue
        if position_violations(cand): continue
        if not _mate_ok(solver, cand, d, kind, opts): continue
        e = energy(cand)
        if e <= cur_e or rng.random() < math.exp((cur_e - e) / max(1e-6, T)):
            cur, cur_e = cand, e
            if e < best_e: best, best_e = cand, e
    return best

def minimize(solver, rng, pos, d, kind, opts, deadline):
    best, improved = pos, True
    while improved and time.time() < deadline:
        improved = False
        for sqi in sorted(range(81), key=lambda s: -piece_value(best.bd[s])):
            p = best.bd[sqi]
            if not p or abs(p) == KING: continue
            nb = best.bd[:]; nb[sqi] = 0
            cand = Position(nb, best.hands, best.side)
            if _mate_ok(solver, cand, d, kind, opts):
                best = cand; improved = True; break
        if improved: continue
        for side in (SENTE, GOTE):
            hit = False
            for t in list(best.hands[side].keys()):
                if best.hands[side].get(t, 0) > 0:
                    h = dict(best.hands); hh = dict(h[side]); hh[t] -= 1
                    if hh[t] <= 0: del hh[t]
                    h[side] = hh
                    cand = Position(best.bd[:], h, best.side)
                    if _mate_ok(solver, cand, d, kind, opts):
                        best = cand; improved = True; hit = True; break
            if hit: break
    if time.time() < deadline:
        best = sa_minimize(solver, rng, best, d, kind, opts, deadline)
        if position_violations(best): best = pos
    return best

def finalize_mate(solver, p, d, kind, opts):
    """-> ((node, line), 'ok') or (None, reason) for the near-miss pool."""
    try:
        if d > 1 and solver.mates_fast(p, d-1): return None, 'deeper'
        ks = solver.mates(p, d)
    except _Budget:
        return None, 'budget'
    if not ks: return None, 'no_mate'
    if opts['unique'] and len(ks) > 1: return None, 'dual'
    if kind == 'dropmate':
        ks = [x for x in ks if x.fr < 0]
        if not ks: return None, 'no_mate'
    k0 = ks[0]
    try:
        node = solution_tree(solver, p, d, k0)
    except _Budget:
        return None, 'budget'
    if node is None: return None, 'no_mate'
    line = pv_from_node(node)
    if kind == 'sacrifice':
        m0 = line[0]
        if m0.fr < 0 or not attackers_of(p, m0.tr, m0.tc, GOTE):
            return None, 'unclassified'
    if kind == 'handpair':
        hp = p.hands[SENTE]
        if not hp or any(t not in (GOLD, SILVER) for t in hp) \
                or sum(hp.values()) > 2: return None, 'unclassified'
        if not any(mm.fr < 0 for mm in line): return None, 'unclassified'
    if kind == 'knight' and abs(line[-1].piece) != KNIGHT: return None, 'unclassified'
    if kind == 'pawnmate' and abs(line[-1].piece) != PAWN: return None, 'unclassified'
    if kind in ('double', 'discovered'):
        pc = p
        for mm in line[:-1]: pc = pc.apply(mm)
        if mate_key_kind(pc, line[-1]) != kind: return None, 'unclassified'
    if kind == 'edge' and p.king(GOTE)[0] != 0: return None, 'unclassified'
    if kind == 'corner':
        if p.king(GOTE) not in ((0, 0), (0, 8)): return None, 'unclassified'
        if not any(mm.fr < 0 for mm in line): return None, 'unclassified'
    if kind == 'race':
        try:
            threat = Position(p.bd, p.hands, GOTE)
            if solver.mates_fast(threat, d): return None, 'unclassified'
            if not solver.mates_fast(threat, d+1): return None, 'unclassified'
        except _Budget:
            return None, 'budget'
    return (node, line), 'ok'

def rescue_near_misses(solver, rng, pool, kind, opts, deadline):
    cands = [it for it in pool.items if kind == 'mixed' or it[2] == kind]
    for pr, pos, k, d, reason in cands:
        if time.time() > deadline: return None
        for _ in range(6):
            cand = mutate_position(rng, pos)
            if cand is None: continue
            cand = normalize_position(cand, rng)
            if cand is None or position_violations(cand): continue
            if cand.in_check(GOTE) or cand.in_check(SENTE): continue
            got, _r = finalize_mate(solver, cand, d, k, opts)
            if got:
                node, line = got
                pz = Puzzle(0, cand, k, d, line, node, 3, nodes=solver.nodes)
                pz.diff = diff_score(pz); pz.stars = stars_from_diff(pz.diff)
                pz.plaus = plausibility_score(cand)
                return pz
    return None

def make_mate_puzzle(rng, kind, opts, deadline, pool=None, bandit=None,
                     templates=()):
    opts = dict(opts); t0 = time.time()
    if kind == 'handpair':
        opts['hand_pool'] = (GOLD, SILVER); opts['drops'] = True
    solver = MateSolver(opts['pawn_mate'], deadline)
    dflt = KIND_DEPTH.get(kind)
    while time.time() < deadline:
        arm = None
        if bandit:
            arm = bandit.select(rng)
            opts['knobs'] = arm_to_knobs(arm, dflt or 2)
        else:
            opts.pop('knobs', None)
        opts['tmpl'] = rng.choice(templates) if templates and rng.random() < 0.4 \
            else None
        d = dflt if dflt else (2 if kind == 'race' else rng.choice((1, 2)))
        if opts.get('cpsat') and HAS_CP and rng.random() < 0.15:
            bdc = cpsat_propose(rng, d)
            pos = normalize_position(
                Position(bdc, {SENTE: {}, GOTE: {}}, SENTE), rng) \
                if bdc is not None else None
        else:
            pos = mate_cluster(rng, kind, d, opts,
                               knobs=opts.get('knobs'), tmpl=opts.get('tmpl'))
        if pos is None: continue
        pos = normalize_position(pos, rng)
        if pos is None or pos.in_check(GOTE) or pos.in_check(SENTE): continue
        if kind == 'race':
            pos = add_gote_threat(rng, pos, d)
            pos = normalize_position(pos, rng)
            if pos is None or pos.in_check(GOTE) or pos.in_check(SENTE): continue
        if opts.get('strict40'):
            pos = fill_to_full(pos, rng)
        got, reason = finalize_mate(solver, pos, d, kind, opts)
        if got is None:
            if pool and reason in ('dual', 'deeper'):
                pool.add(pos, kind, d, reason)
            if bandit and arm: bandit.update(arm, 0.0)
            continue
        node, pv = got
        mpos = pos
        if kind in KIND_DEPTH:
            mpos = minimize(solver, rng, pos, d, kind, opts, deadline)
            if mpos is not pos and position_violations(mpos):
                mpos = pos
            if mpos is not pos:
                g2, _r2 = finalize_mate(solver, mpos, d, kind, opts)
                if g2: node, pv = g2
                else:  mpos = pos
        pz = Puzzle(0, mpos, kind, d, pv, node, 3,
                    nodes=solver.nodes, secs=time.time()-t0)
        pz.diff = diff_score(pz); pz.stars = stars_from_diff(pz.diff)
        pz.plaus = plausibility_score(mpos)
        if kind == 'race': pz.threat_m = d + 1
        if bandit and arm: bandit.update(arm, 1.0)
        return pz
    if pool:
        pz = rescue_near_misses(solver, rng, pool, kind, opts, deadline)
        if pz: pz.secs = time.time()-t0; return pz
    return None

def make_tactic_puzzle(rng, kind, opts, deadline):
    while time.time() < deadline:
        pos = tactic_cluster(rng, kind, opts)
        if pos is None: continue
        pos = normalize_position(pos, rng)
        if pos is None or pos.in_check(GOTE) or pos.in_check(SENTE): continue
        if abs(material_score(pos)) > 8: continue
        if not quick_screen(pos, kind): continue
        r = find_tactic(pos, kind, opts, deadline)
        if r:
            puz = Puzzle(0, r['pos'], r['cls'], 0, r['pv'], None, 3,
                         gain=r['gain'])
            puz.diff = diff_score(puz); puz.stars = stars_from_diff(puz.diff)
            puz.plaus = plausibility_score(r['pos'])
            return puz
    return None

def generate_one(rng, kind, opts, pool=None, bandit=None, templates=()):
    t0 = time.time()
    deadline = t0 + TIMEOUT[kind]
    if kind == 'gote':
        d = rng.choices((1, 2, 3), weights=(5, 4, 1))[0]
        puz = make_gote_mate(rng, d, opts, deadline)
    elif kind == 'parry':
        puz = make_parry_puzzle(rng, opts, deadline)
    elif kind == 'quiet':
        puz = make_quiet_puzzle(rng, opts, deadline)
    elif kind == 'promoch':
        puz = make_promoch_puzzle(rng, opts, deadline)
    elif kind == 'findall':
        puz = make_findall_puzzle(rng, opts, deadline)
    elif kind in TACTIC_KINDS:
        puz = make_tactic_puzzle(rng, kind, opts, deadline)
    else:
        puz = make_mate_puzzle(rng, kind, opts, deadline, pool, bandit, templates)
    if puz: puz.secs = time.time() - t0
    return puz

def _gen_task(args):
    kind, seed, opts = args
    try:
        return (kind, generate_one(random.Random(seed), kind, opts), '')
    except Exception as e:
        return (kind, None, repr(e))

class ShuffleBag:
    def __init__(self, rng, items):
        self.rng, self.items = rng, list(items); self.pool = []
    def draw(self):
        if not self.pool:
            self.pool = self.items[:]; self.rng.shuffle(self.pool)
        return self.pool.pop()

# ============================================================================
# 7c. MANIFEST — persistent dedupe across sessions
# ============================================================================
MANIFEST_VERSION = 1

def default_manifest_path():
    try:
        d = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        d = os.path.expanduser('~')
    return os.path.join(d, 'manifest.json')

def canonical_sfen(pos):
    return min(pos_to_sfen(pos), pos_to_sfen(mirror_position(pos)))

def key_signature(pos, pv, cls):
    if not pv: return None
    m = pv[0]
    k = pos.king(GOTE)
    kr, kc = k if k else (0, 0)
    return (cls, base_of(abs(m.piece)), bool(m.promo), m.fr < 0,
            min(abs(m.tr - kr), 3), min(abs(m.tc - kc), 3))

def manifest_entry(puz):
    return {'csfen': canonical_sfen(puz.pos),
            'sig': key_signature(puz.pos, puz.pv, puz.cls),
            'tmpl': list(template_sig(puz.pos, puz.pv) or ()),
            'niche': list(niche_of(puz)),
            'cls': puz.cls, 'stars': puz.stars,
            'key': puz.pv[0].usi() if puz.pv else '',
            't': int(time.time())}

def load_manifest(path, log=None):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data.get('version') != MANIFEST_VERSION:
            raise ValueError('version mismatch')
        entries = [e for e in data.get('puzzles', [])
                   if isinstance(e, dict) and 'csfen' in e]
        if log is not None:
            log(f'manifest: {len(entries)} puzzle(s) on record')
        return entries
    except FileNotFoundError:
        if log is not None: log('manifest: none yet (starting fresh)')
        return []
    except Exception as e:
        if log is not None: log(f'manifest: unreadable ({e}) — starting fresh')
        return []

def save_manifest(path, entries):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'app': 'shogi-shorts', 'version': MANIFEST_VERSION,
                   'puzzles': entries}, f, indent=1)
    os.replace(tmp, path)

# ============================================================================
# 8. CANVAS LAYOUTS
# ============================================================================
LAYOUTS = {'Shorts 9:16': (1080, 1920), 'Portrait 4:5': (1080, 1350),
           'Square 1:1': (1080, 1080)}
W, H = 1080, 1920
CELL, BOARD, BX, BY = 96, 864, 60, 460
STAND_H, Y_GOTE, Y_STATUS, Y_TIMER, Y_SENTE = 92, 1354, 1466, 1594, 1624
SC = 1.0

def set_layout(name):
    global W, H, CELL, BOARD, BX, BY, STAND_H
    global Y_GOTE, Y_STATUS, Y_TIMER, Y_SENTE, SC
    W, H = LAYOUTS.get(name, LAYOUTS['Shorts 9:16'])
    SC = max(0.55, min(1.0, H/1920))
    BY = 460 if H >= 1600 else 380 if H >= 1300 else 260
    CELL = min(96, (W-140)//9, (H-BY-int(430*SC))//9)
    CELL = max(40, CELL)
    BOARD = CELL*9
    BX = (W-BOARD)//2
    STAND_H = int(92*SC)
    Y_GOTE = BY + BOARD + int(30*SC)
    Y_STATUS = Y_GOTE + STAND_H + int(20*SC)
    Y_TIMER = Y_STATUS + int(128*SC)
    Y_SENTE = Y_TIMER + int(30*SC)

set_layout('Shorts 9:16')

# ============================================================================
# 9. RENDERER
# ============================================================================
@dataclass
class Theme:
    bg1: str; bg2: str; text: str; sub: str; accent: str; accent2: str; danger: str
    board_frame: str; board_face: str; line: str; coord: str
    piece_hi: str; piece_lo: str; piece_edge: str; piece_text: str; promo_text: str
    hand: str; hand_edge: str; hand_text: str
    track: str; panel: str; panel_edge: str

THEMES = {
 'Midnight': Theme('#0a0f1e', '#1b2547', '#f2f5ff', '#8fa0c9', '#ffc94d',
   '#59c8e8', '#ff4d5e', '#3a2a18', '#c89b62', '#5b3d21', '#8fa0c9',
   '#fcf1d4', '#e6d0a0', '#6b4a26', '#232733', '#b3272d',
   '#131b33', '#2b3965', '#c7d2ee', '#253154', '#121a31', '#33406b'),
 'Sumi': Theme('#0c0c0e', '#26262b', '#f5f1e8', '#a09b8f', '#e2483d',
   '#d9b56b', '#ff5347', '#2c2318', '#d3ab68', '#54391f', '#a09b8f',
   '#f8eed1', '#e2cd9b', '#5f4322', '#26262b', '#c03327',
   '#1a1a1e', '#38383f', '#d8d4c8', '#333339', '#191a1f', '#3c3c45'),
 'Paper': Theme('#efe8d8', '#d8ccb0', '#2c2c34', '#6f6a5c', '#c0392b',
   '#2e7d6e', '#d64545', '#7a5a33', '#c89f63', '#5b3d21', '#6f6a5c',
   '#fdf5df', '#ecd9ab', '#6b4a26', '#2c2c34', '#b3272d',
   '#f8f3e7', '#c6b99a', '#4a463c', '#c9bda0', '#faf6ec', '#c6b99a'),
}
FONTS = {}
def resolve_fonts():
    fams = set(QFontDatabase.families())
    def pick(c):
        for x in c:
            if x in fams: return x
        return ''
    FONTS['cjk'] = pick(['Noto Sans JP', 'Noto Sans CJK JP', 'Yu Gothic UI',
                         'Yu Gothic', 'Meiryo', 'Hiragino Sans', 'Malgun Gothic'])
    FONTS['serif'] = pick(['Noto Serif JP', 'Noto Serif CJK JP', 'Yu Mincho',
                           'MS Mincho', 'Hiragino Mincho ProN'])
    FONTS['latin'] = pick(['Montserrat', 'Segoe UI', 'Inter', 'Arial'])

def F(px, kind='latin', bold=False, spacing=0.0):
    f = QFont()
    if FONTS.get(kind): f.setFamilies([FONTS[kind]])
    f.setPixelSize(max(9, int(px))); f.setBold(bold)
    if spacing: f.setLetterSpacing(QFont.AbsoluteSpacing, float(spacing))
    return f

@dataclass
class FrameSpec:
    pos: Position; theme: Theme
    puzzle_no: int = 0; handle: str = ''; stars: int = 0
    title: str = 'MATE IN 1'; subtitle: str = ''
    status: str = ''; status_en: str = ''
    timer: float = -1.0; countdown: int = -1
    dots: tuple = None
    pop: float = -1.0
    hook_txt: str = ''
    recap_moves: list = None
    hl_from: tuple = None; hl_to: tuple = None
    pulse: float = 0.0; check_sq: tuple = None; arrow: tuple = None
    skip_sq: tuple = None; glide: tuple = None
    cap_arc: tuple = None
    skip_cap: tuple = None
    flash: float = -1.0; banner: str = 'CHECKMATE!'
    flash_col: str = None
    outro: float = -1.0; zoom: float = 1.0; ascii: bool = False

def sq_center(r, c):
    return (BX+c*CELL+CELL/2, BY+r*CELL+CELL/2)

def stand_slot(pos_after, side, ptype):
    hand = pos_after.hands[side]
    types = sorted((t for t, n in hand.items() if n > 0), key=lambda t: -t)
    idx = types.index(ptype) if ptype in types else 0
    y = (Y_SENTE if side == SENTE else Y_GOTE) + STAND_H/2
    return BX + int(250*SC) + idx*int(68*SC) + int(29*SC), y

def draw_piece(p, code, cx, cy, w, h, th, ascii_l=False, alpha=1.0, rot=None):
    p.save(); p.translate(cx, cy)
    p.rotate((180.0 if code < 0 else 0.0) if rot is None else rot)
    p.setOpacity(alpha)
    path = QPainterPath()
    path.moveTo(0, -h*0.52); path.lineTo(w*0.44, -h*0.16); path.lineTo(w*0.35, h*0.46)
    path.lineTo(-w*0.35, h*0.46); path.lineTo(-w*0.44, -h*0.16); path.closeSubpath()
    p.save(); p.translate(2.5, 4.5)
    p.setPen(Qt.NoPen); p.setBrush(QColor(0, 0, 0, 80)); p.drawPath(path)
    p.restore()
    g = QLinearGradient(0, -h*0.5, 0, h*0.55)
    g.setColorAt(0, QColor(th.piece_hi)); g.setColorAt(1, QColor(th.piece_lo))
    p.setPen(QPen(QColor(th.piece_edge), 2)); p.setBrush(QBrush(g)); p.drawPath(path)
    base = abs(code); pr = base > KING
    if pr: base -= PROMOTED
    if ascii_l: ch = ('+' if pr else '')+LETTER[base]
    elif base == KING: ch = '王' if code < 0 else '玉'
    else: ch = KANJI_P[base] if pr else KANJI[base]
    p.setFont(F(h*0.40, 'serif'))
    p.setPen(QColor(th.promo_text if pr else th.piece_text))
    p.drawText(QRectF(-w*0.5, -h*0.36, w, h*0.78),
               Qt.AlignHCenter | Qt.AlignVCenter, ch)
    p.restore()

def draw_board(p, th):
    p.setPen(QPen(QColor(th.board_frame), 4)); p.setBrush(QColor(th.board_face))
    p.drawRoundedRect(QRectF(BX-16, BY-16, BOARD+32, BOARD+32), 12, 12)
    p.setPen(QPen(QColor(th.line), 2))
    for i in range(10):
        p.drawLine(QPointF(BX+i*CELL, BY), QPointF(BX+i*CELL, BY+BOARD))
        p.drawLine(QPointF(BX, BY+i*CELL), QPointF(BX+BOARD, BY+i*CELL))
    p.setPen(QPen(QColor(th.line), 5)); p.drawRect(QRectF(BX, BY, BOARD, BOARD))
    p.setPen(Qt.NoPen); p.setBrush(QColor(th.line))
    for gx, gy in ((3, 3), (3, 6), (6, 3), (6, 6)):
        p.drawEllipse(QPointF(BX+gx*CELL, BY+gy*CELL), 6, 6)
    p.setPen(QColor(th.coord)); p.setFont(F(22*SC, 'cjk'))
    for c in range(9):
        p.drawText(QRectF(BX+c*CELL, BY-40*SC, CELL, 30*SC),
                   Qt.AlignHCenter | Qt.AlignVCenter, FW[8-c])
    for r in range(9):
        p.drawText(QRectF(BX+BOARD+6, BY+r*CELL, 46, CELL),
                   Qt.AlignHCenter | Qt.AlignVCenter, KJ[r])

def draw_check_glow(p, s):
    if not s.check_sq: return
    r, c = s.check_sq
    cx, cy = sq_center(r, c)
    a = max(0, min(255, int(150+60*math.sin(s.pulse*8))))
    rg = QRadialGradient(cx, cy, CELL*0.95)
    rg.setColorAt(0, QColor(255, 70, 90, a))
    rg.setColorAt(0.7, QColor(255, 70, 90, a*0.35))
    rg.setColorAt(1, QColor(255, 70, 90, 0))
    p.setPen(Qt.NoPen); p.setBrush(QBrush(rg))
    p.drawEllipse(QPointF(cx, cy), CELL*0.95, CELL*0.95)

def draw_arrow(p, s):
    if not s.arrow: return
    th = s.theme
    fr, fc, tr, tc = s.arrow
    x0, y0 = sq_center(fr, fc); x1, y1 = sq_center(tr, tc)
    dx, dy = x1-x0, y1-y0
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx/L, dy/L
    col = QColor(th.accent); col.setAlpha(215)
    pen = QPen(col, 13); pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen); p.setBrush(Qt.NoBrush)
    p.drawLine(QPointF(x0+ux*CELL*0.3, y0+uy*CELL*0.3),
               QPointF(x1-ux*CELL*0.44, y1-uy*CELL*0.44))
    p.setPen(Qt.NoPen)
    head = QColor(th.accent); head.setAlpha(235); p.setBrush(head)
    px, py = -uy, ux
    p.drawPolygon([QPointF(x1, y1),
                   QPointF(x1-ux*30+px*16, y1-uy*30+py*16),
                   QPointF(x1-ux*30-px*16, y1-uy*30-py*16)])

def draw_hand(p, side, hand, y, s):
    th = s.theme
    p.setBrush(QColor(th.hand)); p.setPen(QPen(QColor(th.hand_edge), 2))
    p.drawRoundedRect(QRectF(BX, y, BOARD, STAND_H), 14, 14)
    p.setFont(F(34*SC, 'cjk')); p.setPen(QColor(th.hand_text))
    p.drawText(QRectF(BX+22*SC, y, 200*SC, STAND_H),
               Qt.AlignLeft | Qt.AlignVCenter,
               f'{"▲" if side == SENTE else "△"} {"Sente" if side == SENTE else "Gote"}')
    x = BX+int(250*SC)
    for pt, cnt in sorted(((t, n) for t, n in hand.items() if n > 0),
                          key=lambda kv: -kv[0]):
        draw_piece(p, side*pt, x+int(29*SC), y+STAND_H/2, 56*SC, 64*SC, th, s.ascii)
        if cnt > 1:
            p.setPen(Qt.NoPen); p.setBrush(QColor(th.danger))
            p.drawEllipse(QPointF(x+int(52*SC), y+int(20*SC)), 15*SC, 15*SC)
            p.setFont(F(22*SC, 'latin', True)); p.setPen(QColor('#ffffff'))
            p.drawText(QRectF(x+int(37*SC), y+int(5*SC), 30*SC, 30*SC),
                       Qt.AlignCenter, str(cnt))
        x += int(68*SC)

def draw_dots(p, s):
    if not s.dots: return
    done, total = s.dots
    if total <= 0: return
    r, gap = 7*SC, 24*SC
    x0 = W/2 - (total-1)*gap/2
    y = BY - 64*SC
    for i in range(total):
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(s.theme.accent) if i < done else QColor(s.theme.track))
        p.drawEllipse(QPointF(x0+i*gap, y), r, r)

def draw_countdown(p, s):
    if s.countdown < 0: return
    th = s.theme
    hot = s.countdown <= 3
    if hot:
        a = int(120+90*abs(math.sin(s.pulse*10)))
        p.setPen(Qt.NoPen); p.setBrush(QColor(255, 60, 70, a//3))
        p.drawRoundedRect(QRectF(BX+BOARD+14*SC, Y_TIMER-24*SC, 108*SC, 64*SC),
                          12, 12)
    p.setFont(F(52*SC, 'latin', True))
    p.setPen(QColor(th.danger if hot else th.text))
    p.drawText(QRectF(BX+BOARD+14*SC, Y_TIMER-24*SC, 108*SC, 64*SC),
               Qt.AlignCenter, str(s.countdown))

def draw_hook(p, s):
    if not s.hook_txt: return
    th = s.theme
    band = QRectF(50, BY+BOARD*0.16, W-100, 150*SC)
    p.setPen(Qt.NoPen); p.setBrush(QColor(0, 0, 0, 150))
    p.drawRoundedRect(band, 20, 20)
    sc = 1.0 + 0.35*math.exp(-5*max(0.0, s.pulse))
    p.save(); p.translate(band.center().x(), band.center().y()); p.scale(sc, sc)
    p.setFont(F(62*SC, 'latin', True))
    p.setPen(QColor(0, 0, 0, 160))
    p.drawText(QRectF(-W/2+4, -56*SC, W, 120*SC), Qt.AlignCenter, s.hook_txt)
    p.setPen(QColor(th.accent))
    p.drawText(QRectF(-W/2, -60*SC, W, 120*SC), Qt.AlignCenter, s.hook_txt)
    p.restore()

def draw_recap(p, s):
    if not s.recap_moves: return
    th = s.theme
    top = Y_STATUS - 8*SC
    bot = Y_SENTE + STAND_H + 8*SC
    rect = QRectF(BX-8, top, BOARD+16, bot-top)
    p.setBrush(QColor(th.panel)); p.setPen(QPen(QColor(th.panel_edge), 3))
    p.drawRoundedRect(rect, 18, 18)
    p.setPen(QColor(th.accent)); p.setFont(F(34*SC, 'latin', True, 4*SC))
    p.drawText(QRectF(rect.x(), rect.y()+10*SC, rect.width(), 40*SC),
               Qt.AlignHCenter, 'SOLUTION')
    p.setFont(F(40*SC, 'cjk'))
    fm = QFontMetrics(p.font())
    x, y = rect.x()+22*SC, rect.y()+82*SC
    line_h = 54*SC
    for i, m in enumerate(s.recap_moves):
        tok = move_notation(m)
        w = fm.horizontalAdvance(tok)
        if x+w > rect.right()-20*SC:
            x = rect.x()+22*SC; y += line_h
        p.setPen(QColor(th.accent) if i == 0 else QColor(th.text))
        p.drawText(QPointF(x, y), tok)
        x += w + 20*SC

def render_frame(spec: FrameSpec) -> QImage:
    th = spec.theme
    img = QImage(W, H, QImage.Format_ARGB32_Premultiplied)
    img.fill(QColor(th.bg1))
    p = QPainter(img)
    p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
    g = QLinearGradient(0, 0, 0, H)
    g.setColorAt(0, QColor(th.bg1)); g.setColorAt(1, QColor(th.bg2))
    p.fillRect(0, 0, W, H, QBrush(g))
    rg = QRadialGradient(W/2, 260*SC, 700*SC)
    rg.setColorAt(0, QColor(255, 255, 255, 18)); rg.setColorAt(1, QColor(0, 0, 0, 0))
    p.fillRect(0, 0, W, H, QBrush(rg))
    top = f'SHOGI TSUME  ·  PUZZLE #{spec.puzzle_no:02d}'
    if spec.stars > 0: top += f'  ·  {"★"*spec.stars}{"☆"*(5-spec.stars)}'
    p.setFont(F(28*SC, 'latin', True, 6*SC)); p.setPen(QColor(th.sub))
    p.drawText(QRectF(0, 92*SC, W, 40*SC), Qt.AlignHCenter, top)
    p.setFont(F(112*SC, 'latin', True)); p.setPen(QColor(th.text))
    p.drawText(QRectF(0, 138*SC, W, 160*SC), Qt.AlignHCenter | Qt.AlignVCenter,
               spec.title)
    if spec.subtitle:
        p.setFont(F(42*SC, 'cjk')); p.setPen(QColor(th.sub))
        p.drawText(QRectF(0, 314*SC, W, 56*SC), Qt.AlignHCenter | Qt.AlignVCenter,
                   spec.subtitle)
    p.save()
    cx, cy = BX+BOARD/2, BY+BOARD/2
    p.translate(cx, cy); p.scale(spec.zoom, spec.zoom); p.translate(-cx, -cy)
    draw_board(p, th)
    if spec.hl_from:
        r, c = spec.hl_from
        p.fillRect(QRectF(BX+c*CELL+2, BY+r*CELL+2, CELL-4, CELL-4),
                   QColor(255, 209, 102, 120))
    if spec.hl_to:
        r, c = spec.hl_to
        a = int(150+45*math.sin(spec.pulse*7))
        p.fillRect(QRectF(BX+c*CELL+2, BY+r*CELL+2, CELL-4, CELL-4),
                   QColor(61, 220, 132, a))
    for sq in range(81):
        code = spec.pos.bd[sq]
        if not code: continue
        r, c = divmod(sq, 9)
        if spec.skip_sq == (r, c) and spec.glide: continue
        if spec.skip_cap == (r, c) and spec.cap_arc: continue
        draw_piece(p, code, *sq_center(r, c), CELL*0.86, CELL*0.98, th, spec.ascii)
    if spec.glide:
        code, x0, y0, x1, y1, q = spec.glide
        draw_piece(p, code, x0+(x1-x0)*q, y0+(y1-y0)*q, CELL*0.86, CELL*0.98,
                   th, spec.ascii)
    draw_check_glow(p, spec)
    draw_arrow(p, spec)
    p.restore()
    draw_hand(p, GOTE, spec.pos.hands[GOTE], Y_GOTE, spec)
    draw_hand(p, SENTE, spec.pos.hands[SENTE], Y_SENTE, spec)
    if spec.cap_arc:
        code, ax, ay, aw, ah, rot, al = spec.cap_arc
        draw_piece(p, code, ax, ay, aw, ah, th, spec.ascii, al, rot)
    draw_dots(p, spec)
    if spec.status:
        mv = spec.status[:1] in '▲△'
        sc = 1.0; al = 1.0
        if spec.pop >= 0:
            q = min(1.0, spec.pop)
            sc = 1.0 + 0.22*(1-q)**2
            al = min(1.0, q*2)
        p.save(); p.translate(W/2, Y_STATUS+39*SC); p.scale(sc, sc)
        p.setOpacity(al)
        p.setFont(F(60*SC, 'cjk', bold=mv))
        p.setPen(QColor(th.accent if mv else th.text))
        p.drawText(QRectF(-W/2, -39*SC, W, 78*SC),
                   Qt.AlignHCenter | Qt.AlignVCenter, spec.status)
        p.restore()
    if spec.status_en:
        p.setFont(F(30*SC, 'latin')); p.setPen(QColor(th.sub))
        p.drawText(QRectF(0, Y_STATUS+80*SC, W, 38*SC),
                   Qt.AlignHCenter | Qt.AlignVCenter, spec.status_en)
    if spec.timer >= 0:
        frac = max(0.0, min(1.0, spec.timer))
        p.setPen(Qt.NoPen); p.setBrush(QColor(th.track))
        p.drawRoundedRect(QRectF(BX, Y_TIMER, BOARD, 16*SC), 8*SC, 8*SC)
        if frac > 0:
            p.setBrush(QColor(th.danger if frac < 0.28 else th.accent))
            p.drawRoundedRect(QRectF(BX, Y_TIMER, BOARD*frac, 16*SC), 8*SC, 8*SC)
    draw_countdown(p, spec)
    draw_hook(p, spec)
    if spec.flash >= 0:
        t = spec.flash
        if t < 0.15:
            p.fillRect(0, 0, W, H, QColor(255, 255, 255, int(160*(1-t/0.15))))
        sc = 1.0+0.55*math.exp(-6.0*t)
        p.save(); p.translate(W/2, BY+BOARD/2); p.scale(sc, sc)
        p.setOpacity(min(1.0, t*10)); p.setFont(F(96*SC, 'latin', True))
        p.setPen(QColor(0, 0, 0, 170))
        p.drawText(QRectF(-W/2+6, -52*SC, W, 120*SC), Qt.AlignHCenter, spec.banner)
        p.setPen(QColor(spec.flash_col or th.danger))
        p.drawText(QRectF(-W/2, -60*SC, W, 120*SC), Qt.AlignHCenter, spec.banner)
        p.restore()
    draw_recap(p, spec)
    if spec.outro >= 0:
        t = spec.outro
        p.setOpacity(min(1.0, t*2.5))
        rect = QRectF(150, H-int(500*SC), W-300, int(320*SC))
        p.setBrush(QColor(th.panel)); p.setPen(QPen(QColor(th.panel_edge), 3))
        p.drawRoundedRect(rect, 24, 24)
        p.setPen(QColor(th.accent)); p.setFont(F(72*SC, 'latin', True))
        p.drawText(QRectF(rect.x(), rect.y()+28*SC, rect.width(), 90*SC),
                   Qt.AlignHCenter, 'FOLLOW FOR MORE')
        p.setPen(QColor(th.text)); p.setFont(F(46*SC, 'cjk'))
        p.drawText(QRectF(rect.x(), rect.y()+130*SC, rect.width(), 60*SC),
                   Qt.AlignHCenter, spec.handle)
        p.setPen(QColor(th.sub)); p.setFont(F(32*SC, 'latin'))
        p.drawText(QRectF(rect.x(), rect.y()+200*SC, rect.width(), 50*SC),
                   Qt.AlignHCenter, 'daily shogi tsumeshogi  ▶')
        p.setOpacity(1.0)
    if spec.handle:
        p.setOpacity(0.55); p.setFont(F(30*SC, 'latin'))
        p.setPen(QColor(th.sub))
        p.drawText(QRectF(0, H-64*SC, W, 44*SC), Qt.AlignHCenter, spec.handle)
        p.setOpacity(1.0)
    p.end()
    return img

# ============================================================================
# 10. TIMELINE
# ============================================================================
PACING = {
 'Snappy':    {'hook': 0.9, 'think': 3.0, 'answer': 0.5, 'move': 1.05,
               'between': 0.55, 'mate': 1.4, 'recap': 1.3, 'outro': 1.3},
 'Normal':    {'hook': 1.2, 'think': 5.0, 'answer': 0.6, 'move': 1.40,
               'between': 0.75, 'mate': 1.6, 'recap': 1.6, 'outro': 1.5},
 'Cinematic': {'hook': 1.6, 'think': 7.0, 'answer': 0.8, 'move': 1.80,
               'between': 1.00, 'mate': 2.2, 'recap': 2.0, 'outro': 1.9},
 'Auto':      None,
}
HOOKS = ['Can you solve this?', '99% miss this move', 'Only 1 move saves it',
         'Find the best move!', 'Pros see it instantly', 'Beat the clock ⏱']

def hook_for(puz, salt=0):
    r = random.Random(f'{puz.index}:{puz.cls}:{salt}')
    return r.choice(HOOKS if puz.depth else HOOKS[:5])

def ease_out(t): return 1-(1-t)**3

def _anim_move(s, pos_before, move, pos_after, t, zoom=1.0, atk=SENTE):
    cap = pos_before.at(move.tr, move.tc) if move.fr >= 0 else 0
    if move.fr >= 0:
        s.skip_sq = (move.fr, move.fc); s.hl_from = (move.fr, move.fc)
        x0, y0 = sq_center(move.fr, move.fc)
        x1, y1 = sq_center(move.tr, move.tc)
        s.arrow = (move.fr, move.fc, move.tr, move.tc)
    else:
        x1, y1 = sq_center(move.tr, move.tc)
        x0, y0 = x1, max(y1-CELL*1.6, BY-30)
    s.glide = (move.piece, x0, y0, x1, y1, ease_out(min(1.0, t/0.5)))
    s.hl_to = (move.tr, move.tc)
    if cap:
        ce = min(1.0, max(0.0, (t-0.32)/0.68))
        ce = ce*ce*(3-2*ce)
        if ce > 0:
            s.skip_cap = (move.tr, move.tc)
            side_cap = SENTE if move.piece > 0 else GOTE
            tx, ty = stand_slot(pos_after, side_cap, base_of(cap))
            z = zoom
            bcx, bcy = BX+BOARD/2, BY+BOARD/2
            sx = bcx + z*(x1-bcx); sy = bcy + z*(y1-bcy)
            sw, sh = CELL*0.86*z, CELL*0.98*z
            ew, eh = 56*SC, 64*SC
            w = sw + (ew-sw)*ce; h = sh + (eh-sh)*ce
            ax = sx + (tx-sx)*ce
            ay = sy + (ty-sy)*ce - CELL*1.1*SC*math.sin(math.pi*ce)
            rot0 = 180.0 if cap < 0 else 0.0
            rot1 = 0.0 if side_cap == SENTE else 180.0
            alpha = 1.0 if ce < 0.85 else max(0.0, 1.0-(ce-0.85)/0.15)
            s.cap_arc = (cap, ax, ay, w, h, rot0+(rot1-rot0)*ce, alpha)
    if (move.piece > 0) == (atk > 0) and t > 0.5 and pos_after.in_check():
        msd = GOTE if atk == SENTE else SENTE
        s.check_sq = pos_after.king(msd); s.pulse = t

def timeline(puz, opts):
    n_moves = len(puz.pv)
    alt = alt_branch(puz) if getattr(opts, 'alt', True) else None
    n_extra = len(puz.keys) if (puz.cls == 'findall' and puz.keys) else 0
    if opts.pacing == 'Auto':
        target = float(getattr(opts, 'target_s', 35.0))
        fixed = 1.2 + 0.6 + 1.6 + 1.6 + 1.4 + (0.8 if alt else 0)
        slots = 2*n_moves - 1 + n_extra
        mv = max(0.9, min(2.0, (target-fixed)/max(1, slots)))
        think = target - (fixed + mv*slots)
        p = {'hook': 1.2, 'think': max(1.5, min(8.0, think)), 'answer': 0.6,
             'move': mv, 'between': mv*0.55, 'mate': 1.6, 'recap': 1.6,
             'outro': 1.4}
    else:
        p = dict(PACING[opts.pacing])
        if getattr(opts, 'think_s', 0) > 0: p['think'] = opts.think_s
    segs, t = [], 0.0
    def add(name, dur, data=None):
        nonlocal t
        segs.append((name, dur, data, t)); t += dur
    add('hook', p['hook']); add('think', p['think']); add('answer', p['answer'])
    for i in range(n_moves):
        add('move', p['move'], i)
        if i < n_moves-1: add('between', p['between'], i)
    add('mate', p['mate'])
    if puz.cls == 'findall' and puz.keys:
        for i in range(len(puz.keys)):
            add('reveal', p['move'], i)
        add('allmate', 1.4)
    elif alt:
        add('altprep', 0.8)
        add('altmove', p['move'], 0)
        for i in range(len(alt[2])):
            add('altrefute', p['move'], i)
        add('altmate', p['mate'])
    add('recap', p['recap']); add('outro', p['outro'])
    return segs

def _base_spec(puz, opts, poses):
    if puz.cls in ('gote', 'parry', 'quiet', 'promoch', 'handpair',
                   'findall', 'race'):
        title = TITLES[puz.cls]
    elif puz.depth:
        title = f'MATE IN {puz.depth}'
    else:
        title = TITLES.get(puz.cls, 'TACTIC')
    return FrameSpec(pos=poses[0], theme=THEMES[opts.theme], puzzle_no=puz.index,
                     handle=opts.handle, title=title, ascii=opts.ascii,
                     stars=puz.stars)

def iter_video_frames(puz, opts):
    fps = max(1, opts.fps)
    moves = list(puz.pv)
    poses = [puz.pos]
    for m in moves: poses.append(poses[-1].apply(m))
    mside = defside(puz)
    gk = poses[-1].king(mside) if poses[-1].in_check() else None
    hk = hook_for(puz, puz.index)
    total = len(moves)
    movename = 'Sente' if puz.pos.side == SENTE else 'Gote'
    altinfo = alt_branch(puz) if getattr(opts, 'alt', True) else None
    alt_moves, alt_poses = [], []
    if altinfo:
        pak, arp, refute = altinfo
        alt_moves = [arp] + list(refute)
        alt_poses = [pak]
        for mm in alt_moves: alt_poses.append(alt_poses[-1].apply(mm))
    reveal_poses = []
    if puz.cls == 'findall' and puz.keys:
        reveal_poses = [poses[0].apply(k) for k in puz.keys]
    nkeys = 1 + len(reveal_poses)
    for name, dur, data, _t0 in timeline(puz, opts):
        n = max(1, round(dur*fps))
        for f in range(n):
            t = (f+0.5)/n
            s = _base_spec(puz, opts, poses)
            if name == 'hook':
                s.hook_txt = hk; s.pulse = t
                s.zoom = 1.0 + 0.04*t
            elif name == 'think':
                s.timer = 1.0-t; s.zoom = 1.0+0.05*t
                s.subtitle = f'{movename} to move'
                s.status = 'Find the best move…'; s.pop = min(1.0, t*4)
                s.countdown = max(0, math.ceil((1-t)*dur - 1e-6))
                if puz.cls == 'race' and puz.threat_m:
                    s.status_en = f'⚠ Gote mates in {puz.threat_m} — hurry!'
            elif name == 'answer':
                s.subtitle = 'The answer'; s.status = 'Watch ↓'
                s.pop = min(1.0, t*4); s.zoom = 1.05
            elif name == 'move':
                m = moves[data]; s.pos = poses[data]; s.zoom = 1.05
                _anim_move(s, poses[data], m, poses[data+1], t, s.zoom,
                           atk=puz.pos.side)
                s.dots = (data, total)
                s.status = (f'{move_notation(m)}  ·  '
                            f'{"Sente" if m.piece > 0 else "Gote"}')
                s.status_en = move_english(m)
                s.pop = min(1.0, t*4)
            elif name == 'between':
                s.pos = poses[data+1]
                prev = moves[data]
                s.hl_to = (prev.tr, prev.tc); s.zoom = 1.05
                s.dots = (data+1, total)
                s.status = ('And now, the finish!' if data+1 == total-1
                            else 'The reply is forced…')
                s.pop = min(1.0, t*4)
                if poses[data+1].in_check():
                    s.check_sq = poses[data+1].king(mside); s.pulse = t
            elif name == 'mate':
                s.pos = poses[-1]; s.flash = t
                s.banner = 'FIRST MATE!' if puz.cls == 'findall' else puz.banner
                s.zoom = 1.05+0.04*t; s.subtitle = 'Solved!'
                s.status = puz.note_en; s.pop = min(1.0, t*6)
                s.status_en = 'Comment your solve time ⏱'
                if gk: s.check_sq = gk; s.pulse = t
            elif name == 'reveal':
                k = puz.keys[data]; s.pos = poses[0]; s.zoom = 1.05
                _anim_move(s, poses[0], k, reveal_poses[data], t, s.zoom,
                           atk=puz.pos.side)
                s.dots = (data+1, len(reveal_poses))
                s.status = f'Mate #{data+2} of {nkeys}: {move_notation(k)}'
                s.status_en = move_english(k)
                s.pop = min(1.0, t*4)
                if t > 0.5 and reveal_poses[data].in_check():
                    s.check_sq = reveal_poses[data].king(mside); s.pulse = t
            elif name == 'allmate':
                s.pos = poses[0]; s.flash = t
                s.banner = 'ALL MATES FOUND!'; s.flash_col = s.theme.accent2
                s.zoom = 1.05+0.04*t
                s.subtitle = f'{nkeys} key moves!'
                s.status = 'Every mate-in-one was found.'
                s.pop = min(1.0, t*6)
            elif name == 'altprep':
                s.pos = alt_poses[0]; s.zoom = 1.0
                s.subtitle = 'Solved!  ·  bonus line'
                s.status = 'What if the defender tries differently?'
                s.status_en = 'Longest-resistance alternative'
                s.pop = min(1.0, t*4)
            elif name == 'altmove':
                s.pos = alt_poses[0]; s.zoom = 1.05
                _anim_move(s, alt_poses[0], alt_moves[0], alt_poses[1], t,
                           s.zoom, atk=puz.pos.side)
                s.status = (f'{move_notation(alt_moves[0])}  ·  '
                            f'{"Sente" if alt_moves[0].piece > 0 else "Gote"}')
                s.status_en = 'Alternative defense — the toughest try'
                s.pop = min(1.0, t*4)
            elif name == 'altrefute':
                mv2 = alt_moves[1+data]
                s.pos = alt_poses[1+data]; s.zoom = 1.05
                _anim_move(s, alt_poses[1+data], mv2, alt_poses[2+data], t,
                           s.zoom, atk=puz.pos.side)
                last = data == len(alt_moves)-2 and \
                    (mv2.piece > 0) == (puz.pos.side > 0)
                s.status = (f'{move_notation(mv2)}  ·  '
                            f'{"Sente" if mv2.piece > 0 else "Gote"}')
                s.status_en = 'Mate!' if last else 'The refutation'
                s.pop = min(1.0, t*4)
            elif name == 'altmate':
                s.pos = alt_poses[-1]; s.flash = t
                s.banner = 'STILL MATE!'; s.flash_col = s.theme.accent2
                s.subtitle = 'No escape'; s.zoom = 1.05+0.04*t
                s.status = 'The alternative defense loses too.'
                s.pop = min(1.0, t*6)
                if alt_poses[-1].in_check():
                    s.check_sq = alt_poses[-1].king(mside); s.pulse = t
            elif name == 'recap':
                s.pos = poses[-1]; s.zoom = 1.0
                s.subtitle = 'Solved!'
                s.recap_moves = list(moves) + \
                    (list(puz.keys) if puz.cls == 'findall' and puz.keys else [])
                if gk: s.check_sq = gk; s.pulse = t
            else:
                s.pos = poses[-1]; s.outro = t; s.zoom = 1.09
            yield render_frame(s)

def video_frames_count(puz, opts):
    return sum(max(1, round(d*opts.fps)) for _n, d, _x, _t in timeline(puz, opts))

# ============================================================================
# 11. AUDIO
# ============================================================================
SR = 44100

def _tone(dur, f0, f1=None, vol=0.5, decay=8.0):
    t = np.arange(int(dur*SR))/SR
    f = f0 if f1 is None else f0+(f1-f0)*t/max(dur, 1e-9)
    return np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-decay*t)*vol

def _click():
    n = np.random.RandomState(1).uniform(-1, 1, int(0.03*SR))
    return n*np.exp(-60*np.arange(len(n))/SR)*0.5 + _tone(0.06, 1700, 900, 0.35, 30)

def _thud(): return _tone(0.13, 170, 70, 0.7, 18)
def _ding(): return _tone(0.5, 1320, None, 0.4, 7) + _tone(0.5, 1980, None, 0.15, 9)

def _whoosh(dur=0.3, vol=0.25):
    n = int(dur*SR); t = np.arange(n)/SR
    noise = np.random.RandomState(3).uniform(-1, 1, n)
    sweep = np.sin(2*np.pi*(220+700*t/dur)*t)
    env = np.sin(np.pi*t/dur)**2
    return (0.6*noise+0.4*sweep)*env*vol

def _chime():
    out = np.zeros(int(1.3*SR))
    for i, f in enumerate((880, 1108.7, 1318.5)):
        s = _tone(1.1, f, None, 0.33, 4.5)
        out[i*int(0.13*SR):i*int(0.13*SR)+len(s)] += s
    return out

def _bgm(total):
    chords = [(220.0, 261.63, 329.63), (174.61, 220.0, 261.63),
              (196.0, 246.94, 293.66), (164.81, 196.0, 246.94)]
    out = np.zeros(int((total+0.5)*SR))
    seg, tpos, i = 2.0, 0.35, 0
    while tpos < total:
        ch = chords[i % 4]
        dur = min(seg, total+0.4-tpos)
        n = int(dur*SR); t = np.arange(n)/SR
        env = np.clip(np.minimum(t/0.4, (dur-t)/0.4), 0, 1)
        sig = sum(np.sin(2*np.pi*f*t + j*0.7) for j, f in enumerate(ch))/3
        s = int(tpos*SR)
        out[s:s+n] += sig*env*0.09
        tpos += seg; i += 1
    return out[:int(total*SR)]

def build_audio(puz, opts, path_wav):
    segs = timeline(puz, opts)
    total = segs[-1][3] + segs[-1][1] + 0.3
    buf = np.zeros(int(total*SR))
    def put(sig, at):
        i = int(at*SR)
        j = min(len(buf), i+len(sig))
        if j > i: buf[i:j] += sig[:j-i]
    moves = list(puz.pv)
    poses = [puz.pos]
    for m in moves: poses.append(poses[-1].apply(m))
    altinfo = alt_branch(puz) if getattr(opts, 'alt', True) else None
    alt_moves, alt_poses = [], []
    if altinfo:
        pak, arp, refute = altinfo
        alt_moves = [arp] + list(refute)
        alt_poses = [pak]
        for mm in alt_moves: alt_poses.append(alt_poses[-1].apply(mm))
    for name, dur, data, t0 in segs:
        if name == 'hook':
            put(_whoosh(0.35, 0.30), t0)
        elif name == 'think':
            put(_tone(0.5, 520, 640, 0.12, 5), t0)
            for k in (1, 2, 3):
                at = t0+dur-k
                if at > t0+0.15:
                    put(_tone(0.07, 990 if k < 3 else 1180, None, 0.30, 26), at)
        elif name == 'move':
            m = moves[data]
            cap = m.fr >= 0 and bool(poses[data].at(m.tr, m.tc))
            put(_thud() if cap else _click(), t0)
            if not cap: put(_whoosh(0.22, 0.14), t0+0.02)
            if poses[data+1].in_check(): put(_ding()*0.8, t0+0.30)
        elif name == 'mate':
            put(_chime(), t0)
        elif name == 'altprep':
            put(_tone(0.4, 470, 560, 0.14, 6), t0)
        elif name == 'altmove':
            m = alt_moves[0]
            cap = m.fr >= 0 and bool(alt_poses[0].at(m.tr, m.tc))
            put(_thud() if cap else _click(), t0)
            if not cap: put(_whoosh(0.22, 0.14), t0+0.02)
        elif name == 'altrefute':
            m = alt_moves[1+data]
            cap = m.fr >= 0 and bool(alt_poses[1+data].at(m.tr, m.tc))
            put(_thud() if cap else _click(), t0)
            if alt_poses[2+data].in_check(): put(_ding()*0.8, t0+0.30)
        elif name == 'altmate':
            put(_chime()*0.7, t0)
        elif name == 'reveal':
            put(_click(), t0)
        elif name == 'allmate':
            put(_chime()*0.8, t0)
        elif name == 'recap':
            put(_ding()*0.5, t0)
    if getattr(opts, 'bgm', True):
        buf += _bgm(total)
    write_wav(path_wav, np.clip(buf, -1, 1))
    return total

def write_wav(path, samples):
    s = np.clip(samples, -1, 1)
    data = (s*32767).astype('<i2').tobytes()
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(data)

def mux_audio(video, wav, log):
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        log('audio: ffmpeg not found — kept silent video'); return False
    tmp = video + '.mux.mp4'
    try:
        subprocess.run([ff, '-y', '-loglevel', 'error', '-i', video, '-i', wav,
                        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k',
                        '-shortest', tmp], check=True, timeout=300)
        os.replace(tmp, video)
        return True
    except Exception as e:
        log(f'audio mux failed: {e}')
        try: os.remove(tmp)
        except OSError: pass
        return False

# ============================================================================
# 12. MP4 WRITER + EXPORT PIPELINE
# ============================================================================
def qimage_to_array(img):
    im = img.convertToFormat(QImage.Format_RGB888)
    h, w, bpl = im.height(), im.width(), im.bytesPerLine()
    arr = np.frombuffer(im.constBits(), np.uint8)[:h*bpl].reshape(h, bpl)
    return np.ascontiguousarray(arr[:, :w*3].reshape(h, w, 3))

def open_writer(path, fps):
    try:
        import imageio.v2 as imageio
        return 'imageio', imageio.get_writer(
            path, fps=fps, codec='libx264', pixelformat='yuv420p',
            macro_block_size=8, ffmpeg_log_level='error',
            output_params=['-movflags', '+faststart', '-crf', '19'])
    except Exception:
        pass
    import cv2
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    if not w.isOpened(): raise RuntimeError('cv2 writer failed')
    return 'cv2', w

def close_writer(backend, w):
    try: (w.close if backend == 'imageio' else w.release)()
    except Exception: pass

def export_video(puz, outdir, opts, log, stop_fn=None, progress=None):
    set_layout(opts.layout)
    hk = hook_for(puz, puz.index)
    base = os.path.join(outdir, f'shogi_short_{puz.index:02d}_{puz.cls}')
    path = base + '.mp4'
    backend, w = open_writer(path, opts.fps)
    done = 0
    try:
        for frame in iter_video_frames(puz, opts):
            if stop_fn(): break
            w.append_data(qimage_to_array(frame))
            done += 1
            if progress: progress()
    finally:
        close_writer(backend, w)
    log(f'  ✔ {path} ({done} frames)')
    s = _base_spec(puz, opts, [puz.pos])
    s.timer = 0.4
    s.subtitle = 'Sente to move' if puz.pos.side == SENTE else 'Gote to move'
    s.status = 'Find the best move…'; s.hook_txt = hk
    render_frame(s).save(base + '.png')
    sol = ' '.join(m.usi() for m in puz.pv)
    sol_jp = ' '.join(move_notation(m) for m in puz.pv)
    cat = TITLES.get(puz.cls, f'Mate in {puz.depth}')
    title = f'{cat} — {hk} | Shogi Puzzle #{puz.index} {"★"*puz.stars}'
    alt = alt_branch(puz) if getattr(opts, 'alt', True) else None
    alt_txt = ''
    if alt:
        alt_txt = (f'\nAlt defense: {move_notation(alt[1])} → '
                   f'{" ".join(move_notation(m) for m in alt[2])}')
    genre_txt = ''
    if puz.cls == 'race' and puz.threat_m:
        genre_txt = f'\n⚠ Gote threatens mate in {puz.threat_m} — Sente is faster.'
    elif puz.cls == 'findall' and puz.keys:
        genre_txt = ('\nAll keys: ' +
                     ' '.join(m.usi() for m in [puz.pv[0]] + puz.keys))
    desc = (f'{hk}\n\n{puz.note_en}\n\n'
            f'Category: {cat} · Difficulty: {"★"*puz.stars} ({puz.diff}/99) · '
            f'{len(puz.pv)} plies · Plausibility {puz.plaus:.2f}\n\n'
            f'⬇️ ⬇️ ⬇️  ANSWER BELOW  ⬇️ ⬇️ ⬇️\n\n\n\n\n\n\n\n'
            f'ANSWER: {sol}\n{sol_jp}{alt_txt}{genre_txt}\n\n'
            f'SFEN: {pos_to_sfen(puz.pos)}\n\n'
            f'{opts.handle} · daily shogi tsumeshogi\n'
            '#shogi #puzzle #checkmate #tsumeshogi #shorts #将棋 #詰将棋')
    with open(base + '.txt', 'w', encoding='utf-8') as f:
        f.write(f'TITLE: {title}\n\nDESCRIPTION:\n{desc}\n')
    if opts.sound:
        try:
            wav = base + '.wav'
            build_audio(puz, opts, wav)
            if mux_audio(path, wav, log):
                log('  ♪ audio muxed')
            os.remove(wav)
        except Exception as e:
            log(f'  audio skipped: {e}')
    return path

def export_pack(puzzles, outdir):
    pack = {'app': 'shogi-shorts', 'version': 4, 'puzzles': []}
    for p in puzzles:
        pack['puzzles'].append({
            'sfen': pos_to_sfen(p.pos), 'category': p.cls,
            'pv': [m.usi() for m in p.pv], 'stars': p.stars, 'diff': p.diff})
    path = os.path.join(outdir, 'pack.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(pack, f, indent=1)
    return path

def import_pack(path, log):
    with open(path, 'r', encoding='utf-8') as f:
        pack = json.load(f)
    out = []
    for i, rec in enumerate(pack.get('puzzles', []), 1):
        try:
            pos = pos_from_sfen(rec['sfen'])
            pos = normalize_position(pos, random.Random(4242))
            if pos is None or position_violations(pos):
                log(f'  ⚠ puzzle {i}: illegal position — skipped'); continue
            pv, cur, ok = [], pos, True
            for u in rec.get('pv', []):
                m = move_from_usi(cur, u)
                if m not in cur.legal_moves(): ok = False; break
                pv.append(m); cur = cur.apply(m)
            if not ok or not pv:
                log(f'  ⚠ puzzle {i}: pv invalid — skipped'); continue
            cls = rec.get('category', 'mate1')
            pz = Puzzle(0, pos, cls, KIND_DEPTH.get(cls, 0), pv, None,
                        int(rec.get('stars', 3)), diff=int(rec.get('diff', 30)))
            out.append(pz)
        except Exception as e:
            log(f'  ⚠ puzzle {i}: {e}')
    return out

# ============================================================================
# 13. WORKERS
# ============================================================================
class Opts:
    def __init__(self):
        self.theme = 'Midnight'; self.handle = '@shogi.short'; self.fps = 30
        self.think_s = 5.0; self.ascii = False; self.pacing = 'Normal'
        self.layout = 'Shorts 9:16'; self.sound = True
        self.bgm = True; self.target_s = 35.0; self.alt = True

class GenWorker(QThread):
    found = Signal(object); log = Signal(str); done = Signal(int)
    def __init__(self, kind, count, seed, unique, drops, pawn_mate,
                 min_stars=1, workers=1, strict40=False,
                 manifest_path='', use_manifest=True, block_patterns=True,
                 mapelites=True, cpsat=False):
        super().__init__()
        self.cfg = (kind, count, seed, unique, drops, pawn_mate,
                    min_stars, workers, strict40)
        self.manifest_path = manifest_path
        self.use_manifest = use_manifest
        self.block_patterns = block_patterns
        self.mapelites = mapelites
        self.cpsat = cpsat
        self._stop = False
    def stop(self): self._stop = True

    def run(self):
        (kind, count, seed, unique, drops, pawn_mate,
         min_stars, workers, strict40) = self.cfg
        opts = {'unique': unique, 'drops': drops, 'pawn_mate': pawn_mate,
                'strict40': strict40, 'cpsat': self.cpsat}
        rng = random.Random(seed)
        bag = ShuffleBag(rng, MIX)
        entries = load_manifest(self.manifest_path, self.log.emit) \
                  if (self.use_manifest and self.manifest_path) else []
        manifest_sfen = {e['csfen'] for e in entries}
        if manifest_sfen:
            self.log.emit(f'manifest: skipping {len(manifest_sfen)} '
                          f'recorded position(s)')
        from collections import Counter
        niche_counts = Counter(tuple(e['niche']) for e in entries if e.get('niche'))
        batch_sfen, batch_sig, batch_niche, new_entries = set(), set(), {}, []
        pool, bandit = NearMissPool(), KnobBandit()
        templates = mine_templates(entries)
        if templates:
            self.log.emit(f'templates: {len(templates)} mined pattern(s) active')
        if self.cpsat and not HAS_CP:
            self.log.emit('CP-SAT requested but ortools missing — disabled')
        made = 0
        if strict40:
            self.log.emit('strict 40-piece fill enabled (lower yield)')

        def accept(puz):
            nonlocal made
            cs = canonical_sfen(puz.pos)
            sig = key_signature(puz.pos, puz.pv, puz.cls)
            if cs in manifest_sfen:
                self.log.emit('[x] duplicate (manifest) — re-rolling'); return False
            if cs in batch_sfen:
                self.log.emit('[x] duplicate (batch) — re-rolling'); return False
            if self.block_patterns and sig in batch_sig:
                self.log.emit('[x] duplicate key pattern — re-rolling'); return False
            if not validate_puzzle(puz):
                self.log.emit('[x] failed validation — skipped'); return False
            if puz.stars < min_stars:
                self.log.emit(f'[x] {puz.label} below min stars — skipped')
                return False
            nic = niche_of(puz)
            if self.mapelites:
                tot = niche_counts.get(nic, 0) + batch_niche.get(nic, 0)
                if tot >= MAP_QUOTA:
                    self.log.emit('[x] MAP niche saturated — re-rolling')
                    return False
                batch_niche[nic] = batch_niche.get(nic, 0) + 1
            batch_sfen.add(cs); batch_sig.add(sig)
            new_entries.append(manifest_entry(puz))
            made += 1; puz.index = made
            extra = (f' ({puz.gain:+d} mat)' if puz.depth == 0
                     else f' ({puz.nodes} nodes)')
            self.log.emit(f'[#{made}] ✔ {puz.label}{extra} — {puz.secs:.1f}s')
            self.found.emit(puz)
            return True

        def serial_loop(remaining):
            tries, max_tries = 0, remaining*4 + 8
            while remaining > 0 and tries < max_tries and not self._stop:
                tries += 1
                k = kind if kind != 'mixed' else bag.draw()
                self.log.emit(f'[try {tries}] searching {CAT_TITLE.get(k, k)} …')
                puz = generate_one(rng, k, opts, pool=pool, bandit=bandit,
                                   templates=templates)
                if puz is None: continue
                if accept(puz): remaining -= 1
            return remaining

        if workers > 1:
            tasks = [(kind if kind != 'mixed' else bag.draw(),
                      seed + 7919*(i+1), opts) for i in range(count)]
            try:
                self.log.emit(f'parallel generation × {workers} …')
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futs = {ex.submit(_gen_task, t): t for t in tasks}
                    for fut in as_completed(futs):
                        if self._stop: break
                        k, puz, err = fut.result()
                        if err:
                            self.log.emit(f'  worker error: {err}'); continue
                        if puz is None:
                            self.log.emit(f'[·] {CAT_TITLE.get(k, k)}: not found')
                            continue
                        accept(puz)
            except Exception as e:
                self.log.emit(f'parallel failed ({e}) — falling back to serial')
            left = max(0, count - made)
            if left and not self._stop:
                self.log.emit(f'topping up {left} more (serial, dedupe-aware)…')
                serial_loop(left)
        else:
            serial_loop(count)

        if self.use_manifest and self.manifest_path and new_entries:
            try:
                save_manifest(self.manifest_path, entries + new_entries)
                self.log.emit(f'manifest: +{len(new_entries)} → '
                              f'{len(entries) + len(new_entries)} total')
            except Exception as e:
                self.log.emit(f'manifest save failed: {e}')
        self.done.emit(made)

class ExportWorker(QThread):
    progress = Signal(int, int); file_done = Signal(str); log = Signal(str)
    finished_all = Signal(str)
    def __init__(self, puzzles, outdir, opts):
        super().__init__()
        self.puzzles, self.outdir, self.opts = puzzles, outdir, opts
        self._stop = False
    def stop(self): self._stop = True
    def run(self):
        total = sum(video_frames_count(p, self.opts) for p in self.puzzles)
        done, paths = 0, []
        for p in self.puzzles:
            if self._stop: break
            self.log.emit(f'exporting #{p.index:02d} ({p.label}) …')
            def bump():
                nonlocal done
                done += 1
                self.progress.emit(done, max(1, total))
            try:
                paths.append(export_video(p, self.outdir, self.opts,
                                          self.log, lambda: self._stop, bump))
                self.file_done.emit(paths[-1])
            except Exception as e:
                self.log.emit(f'  export error: {e}')
        if paths:
            try:
                pk = export_pack(self.puzzles, self.outdir)
                self.log.emit(f'  ✔ {pk}')
            except Exception as e:
                self.log.emit(f'  pack error: {e}')
        self.progress.emit(done, max(1, total))
        self.finished_all.emit(self.outdir)

# ============================================================================
# 14. PREVIEW CANVAS
# ============================================================================
class PreviewCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.img = None
        self.setMinimumHeight(380)
    def set_image(self, img):
        self.img = img; self.update()
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor('#0d0f16'))
        if self.img is None: return
        pm = QPixmap.fromImage(self.img)
        sc = min(self.width()/pm.width(), self.height()/pm.height())
        w, h = pm.width()*sc, pm.height()*sc
        p.drawPixmap(QRectF((self.width()-w)/2, (self.height()-h)/2, w, h),
                     pm, QRectF(pm.rect()))

# ============================================================================
# 15. MAIN WINDOW
# ============================================================================
CAT_LIST = ['Mixed', 'Mate in 1', 'Mate in 2', 'Mate in 3', 'Mate in 4',
            'Corner mate', 'Edge mate', 'Double check', 'Discovered check',
            'Knight mate', 'Pawn mate', 'Drop mate', 'Sacrifice mate',
            'Gold & Silver hand mate', 'Gote to mate △', 'Find ALL mates',
            'Mate race', 'Fork', 'Skewer', 'Pin & win', 'Promotion tactic',
            'Promotion choice', 'Best move (quiet key)', 'Find the defense',
            'Discovered attack', 'Win a piece']
CAT_KIND = {'Mixed': 'mixed', 'Mate in 1': 'mate1', 'Mate in 2': 'mate2',
            'Mate in 3': 'mate3', 'Mate in 4': 'mate4', 'Corner mate': 'corner',
            'Edge mate': 'edge', 'Double check': 'double',
            'Discovered check': 'discovered', 'Knight mate': 'knight',
            'Pawn mate': 'pawnmate', 'Drop mate': 'dropmate',
            'Sacrifice mate': 'sacrifice',
            'Gold & Silver hand mate': 'handpair', 'Gote to mate △': 'gote',
            'Find ALL mates': 'findall', 'Mate race': 'race', 'Fork': 'fork',
            'Skewer': 'skewer', 'Pin & win': 'pin', 'Promotion tactic': 'promo',
            'Promotion choice': 'promoch', 'Best move (quiet key)': 'quiet',
            'Find the defense': 'parry', 'Discovered attack': 'discattack',
            'Win a piece': 'winpiece'}

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Shogi Shorts Studio')
        self.puzzles, self.gen, self.exp = [], None, None
        self.player_frames = None; self.player_timer = QTimer(self)
        self.player_timer.timeout.connect(self.player_tick)
        self.settings = QSettings('shogi', 'shorts')
        root = QWidget(); self.setCentralWidget(root)
        lay = QHBoxLayout(root)
        left = QVBoxLayout(); lay.addLayout(left, 0)
        form = QFormLayout()
        self.category = QComboBox(); self.category.addItems(CAT_LIST)
        self.count = QSpinBox(); self.count.setRange(1, 50); self.count.setValue(5)
        self.workers = QSpinBox(); self.workers.setRange(1, 8); self.workers.setValue(1)
        self.minstars = QSpinBox(); self.minstars.setRange(1, 5); self.minstars.setValue(1)
        self.seed = QLineEdit(str(random.randrange(10**9)))
        self.unique = QCheckBox('unique solution'); self.unique.setChecked(True)
        self.drops = QCheckBox('allow drops'); self.drops.setChecked(True)
        self.pawn = QCheckBox('pawn-drop mates ok')
        self.strict40 = QCheckBox('strict 40-piece fill')
        self.useman = QCheckBox('skip puzzles in manifest (no repeats)')
        self.useman.setChecked(self.settings.value('useman', 'true') == 'true')
        self.blockpat = QCheckBox('block repeat key patterns')
        self.blockpat.setChecked(self.settings.value('blockpat', 'true') == 'true')
        form.addRow('Category', self.category); form.addRow('Count', self.count)
        form.addRow('Parallel', self.workers); form.addRow('Min stars', self.minstars)
        form.addRow('Seed', self.seed)
        form.addRow(self.unique); form.addRow(self.drops); form.addRow(self.pawn)
        form.addRow(self.strict40); form.addRow(self.useman)
        form.addRow(self.blockpat)
        left.addLayout(form)
        self.manifest_path = self.settings.value('manifest_path',
                                                 default_manifest_path())
        mrow = QHBoxLayout()
        n_man = len(load_manifest(self.manifest_path))
        self.man_label = QLabel(f'manifest: {n_man} · '
                                f'{os.path.basename(self.manifest_path)}')
        btn_mset = QPushButton('Set…'); btn_mclear = QPushButton('Clear…')
        btn_mset.clicked.connect(self.set_manifest_path)
        btn_mclear.clicked.connect(self.clear_manifest)
        mrow.addWidget(self.man_label, 1); mrow.addWidget(btn_mset)
        mrow.addWidget(btn_mclear)
        left.addLayout(mrow)
        vgo = QGroupBox('Video'); vf = QFormLayout(vgo)
        self.layout_c = QComboBox(); self.layout_c.addItems(list(LAYOUTS))
        self.pacing = QComboBox(); self.pacing.addItems(list(PACING))
        self.target = QDoubleSpinBox(); self.target.setRange(15, 90)
        self.target.setValue(35); self.target.setSuffix(' s')
        self.theme = QComboBox(); self.theme.addItems(list(THEMES))
        self.handle = QLineEdit('@shogi.short')
        self.fps = QSpinBox(); self.fps.setRange(10, 60); self.fps.setValue(30)
        self.think = QDoubleSpinBox(); self.think.setRange(0, 15)
        self.think.setValue(0); self.think.setSpecialValueText('preset')
        self.sound = QCheckBox('sound effects'); self.sound.setChecked(True)
        self.bgmchk = QCheckBox('background music'); self.bgmchk.setChecked(True)
        self.altchk = QCheckBox('alt-defense segment'); self.altchk.setChecked(True)
        self.mapchk = QCheckBox('MAP-Elites niche variety'); self.mapchk.setChecked(True)
        self.cpchk = QCheckBox('CP-SAT proposer' +
                               ('' if HAS_CP else ' (needs ortools)'))
        self.cpchk.setEnabled(HAS_CP)
        self.asciichk = QCheckBox('ASCII pieces')
        vf.addRow('Canvas', self.layout_c); vf.addRow('Pacing', self.pacing)
        vf.addRow('Target (Auto)', self.target)
        vf.addRow('Theme', self.theme); vf.addRow('Handle', self.handle)
        vf.addRow('FPS', self.fps); vf.addRow('Think sec', self.think)
        vf.addRow(self.sound); vf.addRow(self.bgmchk)
        vf.addRow(self.altchk); vf.addRow(self.mapchk)
        vf.addRow(self.cpchk); vf.addRow(self.asciichk)
        left.addWidget(vgo)
        self.btn_gen = QPushButton('Generate puzzles')
        self.btn_export = QPushButton('Export all → MP4+thumb+meta')
        self.btn_export.setEnabled(False)
        self.btn_cancel = QPushButton('Cancel'); self.btn_cancel.setEnabled(False)
        self.btn_play = QPushButton('▶ Play preview')
        self.btn_sfen = QPushButton('Copy SFEN')
        self.btn_imp = QPushButton('Import pack…')
        for b in (self.btn_gen, self.btn_export, self.btn_play, self.btn_sfen,
                  self.btn_imp, self.btn_cancel):
            left.addWidget(b)
        self.progress = QProgressBar(); left.addWidget(self.progress)
        self.listw = QListWidget(); lay.addWidget(self.listw, 1)
        right = QVBoxLayout(); lay.addLayout(right, 1)
        self.canvas = PreviewCanvas()
        right.addWidget(self.canvas, 1)
        self.logw = QPlainTextEdit(); self.logw.setReadOnly(True)
        self.logw.setMaximumHeight(150)
        right.addWidget(self.logw)
        self.btn_gen.clicked.connect(self.generate)
        self.btn_export.clicked.connect(self.export)
        self.btn_cancel.clicked.connect(self.cancel)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_sfen.clicked.connect(self.copy_sfen)
        self.btn_imp.clicked.connect(self.import_pack)
        self.listw.currentRowChanged.connect(self.on_select)
        for wdg, key in ((self.theme, 'theme'), (self.handle, 'handle'),
                         (self.layout_c, 'layout'), (self.pacing, 'pacing')):
            v = self.settings.value(key)
            if v: wdg.setCurrentText(v)
        self.sound.setChecked(self.settings.value('sound', 'true') == 'true')
        self.bgmchk.setChecked(self.settings.value('bgm', 'true') == 'true')
        self.altchk.setChecked(self.settings.value('altseg', 'true') == 'true')
        self.mapchk.setChecked(self.settings.value('mapelites', 'true') == 'true')
        self.strict40.setChecked(self.settings.value('strict40', 'false') == 'true')
        try: self.target.setValue(float(self.settings.value('target', 35.0)))
        except (TypeError, ValueError): pass
        self.layout_c.currentTextChanged.connect(self.relayout)
        self.theme.currentTextChanged.connect(lambda _: self.refresh_preview())
        self.resize(1280, 860)
        self.relayout(self.layout_c.currentText())

    def vopts(self):
        o = Opts()
        o.theme = self.theme.currentText(); o.handle = self.handle.text()
        o.fps = self.fps.value()
        o.think_s = self.think.value() if self.think.value() > 0 else 0
        o.ascii = self.asciichk.isChecked()
        o.pacing = self.pacing.currentText()
        o.layout = self.layout_c.currentText()
        o.sound = self.sound.isChecked()
        o.bgm = self.bgmchk.isChecked()
        o.alt = self.altchk.isChecked()
        o.target_s = self.target.value()
        return o

    def log(self, s): self.logw.appendPlainText(s)
    def cur(self):
        r = self.listw.currentRow()
        return self.puzzles[r] if 0 <= r < len(self.puzzles) else None

    def relayout(self, name):
        set_layout(name); self.refresh_preview()

    def closeEvent(self, ev):
        self.cancel()
        for key, val in (('theme', self.theme.currentText()),
                         ('handle', self.handle.text()),
                         ('layout', self.layout_c.currentText()),
                         ('pacing', self.pacing.currentText()),
                         ('target', self.target.value()),
                         ('sound', 'true' if self.sound.isChecked() else 'false'),
                         ('bgm', 'true' if self.bgmchk.isChecked() else 'false'),
                         ('altseg', 'true' if self.altchk.isChecked() else 'false'),
                         ('mapelites',
                          'true' if self.mapchk.isChecked() else 'false'),
                         ('useman', 'true' if self.useman.isChecked() else 'false'),
                         ('blockpat',
                          'true' if self.blockpat.isChecked() else 'false'),
                         ('strict40',
                          'true' if self.strict40.isChecked() else 'false')):
            self.settings.setValue(key, val)
        super().closeEvent(ev)

    def generate(self):
        if self.gen: return
        self.listw.clear(); self.puzzles.clear()
        self.btn_export.setEnabled(False)
        kind = CAT_KIND[self.category.currentText()]
        try: seed = int(self.seed.text())
        except ValueError: seed = random.randrange(10**9)
        self.gen = GenWorker(kind, self.count.value(), seed,
                             self.unique.isChecked(), self.drops.isChecked(),
                             self.pawn.isChecked(), self.minstars.value(),
                             self.workers.value(), self.strict40.isChecked(),
                             self.manifest_path, self.useman.isChecked(),
                             self.blockpat.isChecked(), self.mapchk.isChecked(),
                             self.cpchk.isChecked())
        self.gen.found.connect(self.add_puzzle)
        self.gen.log.connect(self.log)
        self.gen.done.connect(self.gen_done)
        self.btn_gen.setEnabled(False); self.btn_cancel.setEnabled(True)
        self.gen.start()

    def add_puzzle(self, puz):
        self.puzzles.append(puz)
        self.listw.addItem(puz.label)
        self.listw.setCurrentRow(len(self.puzzles)-1)
        self.btn_export.setEnabled(True)

    def gen_done(self, n):
        self.log(f'— done: {n} puzzle(s) —')
        self.btn_gen.setEnabled(True); self.btn_cancel.setEnabled(False)
        self.gen = None

    def cancel(self):
        if self.gen: self.gen.stop()
        if self.exp: self.exp.stop()

    def on_select(self, _row):
        self.player_timer.stop(); self.player_frames = None
        self.refresh_preview()

    def refresh_preview(self):
        puz = self.cur()
        if not puz: return
        s = _base_spec(puz, self.vopts(), [puz.pos])
        s.timer = 0.5
        s.subtitle = 'Sente to move' if puz.pos.side == SENTE else 'Gote to move'
        s.status = 'Find the best move…'; s.pop = 1.0
        self.canvas.set_image(render_frame(s))

    def toggle_play(self):
        if self.player_timer.isActive():
            self.player_timer.stop(); self.btn_play.setText('▶ Play preview'); return
        puz = self.cur()
        if not puz: return
        set_layout(self.vopts().layout)
        self.player_frames = iter_video_frames(puz, self.vopts())
        self.player_timer.start(int(1000/max(10, self.vopts().fps)))
        self.btn_play.setText('⏸ Stop')

    def player_tick(self):
        try:
            self.canvas.set_image(next(self.player_frames))
        except StopIteration:
            self.player_timer.stop()
            self.btn_play.setText('▶ Play preview')

    def copy_sfen(self):
        puz = self.cur()
        if not puz: return
        QApplication.clipboard().setText(pos_to_sfen(puz.pos))
        self.log('SFEN copied: ' + pos_to_sfen(puz.pos))

    def import_pack(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Import pack', '', 'JSON (*.json)')
        if not path: return
        loaded = import_pack(path, self.log)
        if loaded:
            try:
                prev = load_manifest(self.manifest_path)
                save_manifest(self.manifest_path,
                              prev + [manifest_entry(p) for p in loaded])
            except Exception as e:
                self.log(f'manifest update failed: {e}')
        for i, p in enumerate(loaded, 1):
            p.index = i; self.puzzles.append(p)
            self.listw.addItem(p.label)
        self.listw.setCurrentRow(len(self.puzzles)-1)
        if loaded:
            self.log(f'imported {len(loaded)} puzzle(s) — manifest updated')
            self.btn_export.setEnabled(True)

    def set_manifest_path(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Manifest file',
                                              self.manifest_path, 'JSON (*.json)')
        if not path: return
        self.manifest_path = path
        self.settings.setValue('manifest_path', path)
        n = len(load_manifest(path))
        self.man_label.setText(f'manifest: {n} · {os.path.basename(path)}')
        self.log(f'manifest set: {path} ({n} recorded)')

    def clear_manifest(self):
        if QMessageBox.question(
                self, 'Clear manifest',
                'Forget all recorded puzzles?\nPast puzzles may be '
                'generated again.') != QMessageBox.StandardButton.Yes:
            return
        try:
            save_manifest(self.manifest_path, [])
            self.man_label.setText('manifest: 0 · '
                                   f'{os.path.basename(self.manifest_path)}')
            self.log('manifest cleared')
        except Exception as e:
            self.log(f'manifest clear failed: {e}')

    def export(self):
        if self.exp or not self.puzzles: return
        d = QFileDialog.getExistingDirectory(self, 'Output folder')
        if not d: return
        self.exp = ExportWorker(self.puzzles, d, self.vopts())
        self.exp.progress.connect(lambda a, b: self.progress.setValue(
            int(100*a/max(1, b))))
        self.exp.file_done.connect(lambda p: self.log(f'✔ done: {p}'))
        self.exp.log.connect(self.log)
        self.exp.finished_all.connect(
            lambda _: (self.btn_export.setEnabled(True),
                       setattr(self, 'exp', None)))
        self.btn_export.setEnabled(False)
        self.exp.start()

# ============================================================================
# 16. SELFTEST + MAIN
# ============================================================================
def perft(pos, d):
    if d == 0: return 1
    n = 0
    for m in pos.legal_moves(): n += perft(pos.apply(m), d-1)
    return n

def _legality_selftest():
    bd = [0]*81; bd[8] = KING; bd[72] = -KING
    bd[45] = PAWN; bd[54] = PAWN
    bd[0] = PAWN
    bd[40] = ROOK; bd[41] = ROOK; bd[42] = ROOK
    p = normalize_position(Position(bd, {SENTE: {}, GOTE: {}}, SENTE),
                           random.Random(1))
    assert p is not None and not position_violations(p), 'repair failed'
    print('repair: OK')
    q = fill_to_full(p, random.Random(2))
    g = global_type_counts(q)
    assert all(g[t] == INITIAL_TOTAL[t] for t in INITIAL_TOTAL), 'fill sum failed'
    assert not position_violations(q), 'fill introduced violations'
    print('strict fill: OK')
    bd = [0]*81; bd[40] = KING; bd[31] = -KING
    assert normalize_position(Position(bd, {SENTE: {}, GOTE: {}}, SENTE),
                              random.Random(1)) is None
    bd = [0]*81; bd[76] = KING; bd[4] = -KING; bd[40] = ROOK
    assert normalize_position(Position(bd, {SENTE: {}, GOTE: {}}, SENTE),
                              random.Random(1)) is None
    print('unfixable rejects: OK')

def _selftest():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    app = QGuiApplication([])
    resolve_fonts()
    sp = startpos()
    p1, p2, p3 = perft(sp, 1), perft(sp, 2), perft(sp, 3)
    print(f'perft: {p1} {p2} {p3}',
          'OK' if (p1, p2, p3) == (30, 900, 25470) else 'MISMATCH!')
    assert (p1, p2, p3) == (30, 900, 25470)
    _legality_selftest()
    assert property_tests()
    print('property tests: OK (mirror law, naive differential, attack symmetry)')
    pos = pos_from_sfen('4k4/9/9/9/9/9/9/9/4K4 b RG2rgp1 - 1')
    print('sfen roundtrip:',
          'OK' if pos_to_sfen(pos).startswith('4k4/9/9/9/9/9/9/9/4K4 b ')
          else 'FAIL')
    rng = random.Random(7)
    opts = {'unique': True, 'drops': True, 'pawn_mate': False, 'strict40': False}
    o = Opts(); o.handle = '@shogi.short'; o.fps = 12
    o.pacing = 'Auto'; o.target_s = 30.0; o.sound = True
    last = None
    for kind in ('mate1', 'mate2', 'sacrifice', 'fork'):
        t0 = time.time()
        p = generate_one(rng, kind, opts)
        assert p, f'failed to generate {kind}'
        assert not position_violations(p.pos), f'illegal position for {kind}'
        assert validate_puzzle(p), f'validation failed for {kind}'
        if p.depth:
            assert len(p.pv) == 2*p.depth-1, f'pv length wrong for {kind}'
            sA = MateSolver(False, time.time()+8, persistent=False, mode='std')
            sB = MateSolver(False, time.time()+8, persistent=False, mode=777)
            assert sA.mates_fast(p.pos, p.depth) == \
                   sB.mates_fast(p.pos, p.depth), 'consensus mismatch'
        print(f'{kind}: {p.label}  pv={" ".join(m.usi() for m in p.pv)} '
              f'({time.time()-t0:.1f}s)  plaus={p.plaus:.2f}')
        alt = alt_branch(p)
        if alt:
            print('  alt defense:', move_notation(alt[1]), '→',
                  ' '.join(move_notation(m) for m in alt[2]))
        last = p
    for kind in ('gote', 'parry', 'quiet', 'promoch', 'handpair', 'findall'):
        p = generate_one(rng, kind, opts)
        assert p, f'{kind}: not found'
        assert validate_puzzle(p), f'{kind}: invalid'
        assert not position_violations(p.pos), f'{kind}: illegal position'
        if kind == 'gote': assert p.pos.side == GOTE
        if kind == 'findall': assert p.keys and len(p.keys) >= 1
        print(f'{kind}: OK  pv={" ".join(m.usi() for m in p.pv)}'
              + (f' +{len(p.keys)} more keys' if p.keys else ''))
        last = p
    tl = timeline(last, o)
    total = tl[-1][3] + tl[-1][1]
    print(f'auto-timeline: {total:.1f}s (target {o.target_s:.0f}s), '
          f'{video_frames_count(last, o)} frames')
    export_video(last, '.', o, print)
    ent = manifest_entry(last)
    save_manifest('manifest_test.json', [ent])
    back = load_manifest('manifest_test.json')
    assert back and back[0]['csfen'] == ent['csfen']
    assert canonical_sfen(last.pos) == ent['csfen']
    os.remove('manifest_test.json')
    print('manifest roundtrip: OK')
    print('pack:', export_pack([last], '.'))
    b2 = import_pack('pack.json', print)
    print('reimport:', len(b2), 'puzzle(s)')
    print('SELFTEST OK')

if __name__ == '__main__':
    if '--selftest' in sys.argv:
        _selftest(); sys.exit(0)
    app = QApplication(sys.argv)
    resolve_fonts()
    win = MainWindow(); win.show()
    sys.exit(app.exec())
