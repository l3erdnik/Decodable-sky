#!/usr/bin/env python
"""
Decodable Sky -- score the celestial-coordinate recall test.

Reads the per-model answer tables written by `radec_recall.py`, parses the
right ascension / declination out of each model's answer, and writes a per-model
scored table:

    name, type, true_ra_deg, true_dec_deg, pred_ra_deg, pred_dec_deg,
    parsed, angular_error_deg

Angular error is the great-circle separation between the true and predicted sky
positions. Answers that cannot be parsed -- e.g. a reasoning model that never
committed a final answer -- are assigned 90 deg (a right angle: the mean
separation of a uniformly random guess on the sphere is 90 deg).

Pure Python + stdlib (no numpy needed); runs anywhere, including the laptop.

Usage
-----
    python radec_score.py --indir recall --outdir recall
"""
import argparse, csv, glob, math, os, re

DEG = math.pi / 180.0


def normalize(text):
    """Unicode cleanup so the regexes only see ASCII-ish separators."""
    repl = {
        "−": "-",   # minus sign
        "′": "'",   # prime
        "″": '"',   # double prime
        "º": "°",  # masculine ordinal -> degree
        "‐": "-", "‑": "-", "–": "-", "—": "-",
        "α": " ra ",   # alpha
        "δ": " dec ",  # delta
        "*": "",        # markdown bold/italic
    }
    for a, b in repl.items():
        text = text.replace(a, b)
    return text


def strip_thinking(text):
    """Drop chain-of-thought so we parse only the committed answer.

    Handles three families:
      - Qwen3 / GLM hybrid thinking:  <think> ... </think> <answer>
      - gpt-oss harmony (decoded):    analysis <reasoning> assistantfinal <answer>
      - gpt-oss with special tokens:  ... <|channel|>final<|message|> <answer>
    """
    # gpt-oss harmony: keep only the final channel. If the model never emitted
    # a final channel (e.g. it overthought past the token budget), there is no
    # committed answer -> return "" so it scores as unparsed (90 deg) rather
    # than accidentally parsing a tentative coordinate from the analysis.
    for marker in ("<|channel|>final<|message|>", "assistantfinal"):
        if marker in text:
            return text.rsplit(marker, 1)[1]
    if text.lstrip().startswith("analysis"):
        return ""
    # Qwen3 / GLM think block
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text


# ---- RA parsing (returns degrees in [0,360) or None) -----------------------
_RA_HMS = re.compile(r'(\d{1,2})\s*[hH]\s*(\d{1,2})\s*[m\']\s*(\d{1,2}(?:\.\d+)?)\s*[s"]')
_RA_HM = re.compile(r'(\d{1,2})\s*[hH]\s*(\d{1,2}(?:\.\d+)?)\s*[m\']')
_RA_HOURS = re.compile(r'(-?\d{1,2}(?:\.\d+)?)\s*[hH]')
_RA_DEG = re.compile(r'(-?\d{1,3}(?:\.\d+)?)\s*(?:°|deg|degree)', re.I)
_RA_BARE = re.compile(r'(-?\d{1,3}(?:\.\d+)?)')


def parse_ra(seg):
    m = _RA_HMS.search(seg)
    if m:
        h, mi, s = float(m[1]), float(m[2]), float(m[3])
        return ((h + mi / 60 + s / 3600) * 15) % 360
    m = _RA_HM.search(seg)
    if m:
        h, mi = float(m[1]), float(m[2])
        return ((h + mi / 60) * 15) % 360
    m = _RA_HOURS.search(seg)
    if m:
        return (float(m[1]) * 15) % 360
    m = _RA_DEG.search(seg)
    if m:
        return float(m[1]) % 360
    m = _RA_BARE.search(seg)
    if m:
        v = float(m[1])
        return v % 360 if 0 <= v <= 360 else None
    return None


# ---- Dec parsing (returns degrees in [-90,90] or None) ---------------------
_DEC_DMS = re.compile(r'([+-]?\d{1,2})\s*[°dD]\s*(\d{1,2})\s*[m\']\s*(\d{1,2}(?:\.\d+)?)\s*[s"]')
_DEC_DM = re.compile(r'([+-]?\d{1,2})\s*[°dD]\s*(\d{1,2}(?:\.\d+)?)\s*[m\']')
_DEC_DEG = re.compile(r'([+-]?\d{1,2}(?:\.\d+)?)\s*(?:°|deg|degree)', re.I)
_DEC_BARE = re.compile(r'([+-]?\d{1,2}(?:\.\d+)?)')


def _sign(seg, val):
    low = seg.lower()
    if re.search(r'\bs(?:outh)?\b', low):
        return -abs(val)
    if re.search(r'\bn(?:orth)?\b', low):
        return abs(val)
    return val


def parse_dec(seg):
    m = _DEC_DMS.search(seg)
    if m:
        sign = -1 if m[1].strip().startswith("-") else 1
        d, mi, s = abs(float(m[1])), float(m[2]), float(m[3])
        v = sign * (d + mi / 60 + s / 3600)
        return v if -90 <= v <= 90 else None
    m = _DEC_DM.search(seg)
    if m:
        sign = -1 if m[1].strip().startswith("-") else 1
        d, mi = abs(float(m[1])), float(m[2])
        v = sign * (d + mi / 60)
        return v if -90 <= v <= 90 else None
    m = _DEC_DEG.search(seg)
    if m:
        v = _sign(seg, float(m[1]))
        return v if -90 <= v <= 90 else None
    m = _DEC_BARE.search(seg)
    if m:
        v = _sign(seg, float(m[1]))
        return v if -90 <= v <= 90 else None
    return None


def split_ra_dec(text):
    """Return (ra_segment, dec_segment).

    Priority: (1) a declination keyword splits RA|Dec; (2) otherwise anchor on
    the RA sexagesimal token and treat everything after it as the declination
    (robust to leading newlines and comma/newline separators); (3) fall back to
    the first two non-empty separator-delimited tokens.
    """
    text = text.strip()
    low = text.lower()
    # Last occurrence of a declination keyword = the committed answer (models
    # often mention coordinates mid-reasoning before restating them at the end).
    di = max(low.rfind(kw) for kw in
             ("declination", " dec ", "dec:", "dec.", "\ndec", "dec="))
    if di != -1:
        ra_seg = text[:di]
        ri = ra_seg.lower().rfind("ascension")
        if ri == -1:
            ri = ra_seg.lower().rfind(" ra ")
        if ri != -1:
            ra_seg = ra_seg[ri:]
        return ra_seg, text[di:di + 120]

    # No keyword: anchor on the RA sexagesimal / hours token.
    m = _RA_HMS.search(text) or _RA_HM.search(text) or _RA_HOURS.search(text)
    if m:
        return text[:m.end()], text[m.end():]

    # Fallback: first two non-empty comma/newline/semicolon tokens.
    toks = [t for t in re.split(r'[\n;,]', text) if t.strip()]
    if len(toks) >= 2:
        return toks[0], toks[1]
    return text, text


# Full-string sexagesimal patterns (RA then Dec), tried before segment-splitting
# because ':'/space separators otherwise get mis-split.
_COLON = re.compile(
    r'(\d{1,2})\s*:\s*(\d{1,2})\s*:\s*(\d{1,2}(?:\.\d+)?)'       # RA h:m:s
    r'[\s,;]+'
    r'([+-]?\d{1,2})\s*:\s*(\d{1,2})\s*:\s*(\d{1,2}(?:\.\d+)?)')  # Dec d:m:s
_SPACESEX = re.compile(
    r'(?<![\d.])(\d{1,2})\s+(\d{1,2})\s+(\d{1,2}(?:\.\d+)?)'      # RA h m s
    r'[\s,;]+'
    r'([+-]?\d{1,2})\s+(\d{1,2})\s+(\d{1,2}(?:\.\d+)?)(?![\d.])')  # Dec d m s


def _sexages_ra(h, m, s):
    return ((h + m / 60 + s / 3600) * 15) % 360


def _sexages_dec(d_str, m, s):
    sign = -1 if d_str.strip().startswith("-") else 1
    v = sign * (abs(float(d_str)) + m / 60 + s / 3600)
    return v if -90 <= v <= 90 else None


def parse_coords(text):
    """Return (ra_deg, dec_deg) or (None, None). Tries full colon/space
    sexagesimal first, then keyword/letter/decimal segment parsing."""
    m = _COLON.search(text)
    if m:
        ra = _sexages_ra(float(m[1]), float(m[2]), float(m[3]))
        dec = _sexages_dec(m[4], float(m[5]), float(m[6]))
        if dec is not None:
            return ra, dec
    m = _SPACESEX.search(text)
    if m:
        ra = _sexages_ra(float(m[1]), float(m[2]), float(m[3]))
        dec = _sexages_dec(m[4], float(m[5]), float(m[6]))
        if dec is not None:
            return ra, dec
    ra_seg, dec_seg = split_ra_dec(text)
    return parse_ra(ra_seg), parse_dec(dec_seg)


def angular_error(ra1, dec1, ra2, dec2):
    a1, d1, a2, d2 = ra1 * DEG, dec1 * DEG, ra2 * DEG, dec2 * DEG
    c = (math.sin(d1) * math.sin(d2) +
         math.cos(d1) * math.cos(d2) * math.cos(a1 - a2))
    c = max(-1.0, min(1.0, c))
    return math.acos(c) / DEG


def score_file(path, outdir):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    out_rows = []
    n_parsed = 0
    errs = []
    for r in rows:
        true_ra = float(r["true_ra_deg"])
        true_dec = float(r["true_dec_deg"])
        text = strip_thinking(normalize(r["answer"]))
        pra, pdec = parse_coords(text)
        if pra is not None and pdec is not None:
            err = angular_error(true_ra, true_dec, pra, pdec)
            parsed = 1
            n_parsed += 1
        else:
            err = 90.0
            parsed = 0
        errs.append(err)
        out_rows.append([
            r["name"], r["type"], f"{true_ra:.4f}", f"{true_dec:.4f}",
            "" if pra is None else f"{pra:.4f}",
            "" if pdec is None else f"{pdec:.4f}",
            parsed, f"{err:.4f}",
        ])

    base = os.path.basename(path).replace("radec_answers_", "radec_error_")
    out_csv = os.path.join(outdir, base)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "type", "true_ra_deg", "true_dec_deg",
                    "pred_ra_deg", "pred_dec_deg", "parsed", "angular_error_deg"])
        w.writerows(out_rows)

    n = len(rows)
    mean_err = sum(errs) / n if n else float("nan")
    parsed_errs = [e for e, o in zip(errs, out_rows) if o[6] == 1]
    mean_parsed = sum(parsed_errs) / len(parsed_errs) if parsed_errs else float("nan")
    median = sorted(errs)[n // 2] if n else float("nan")
    print(f"{os.path.basename(path):34s}  n={n:3d}  parsed={n_parsed:3d}/{n}  "
          f"mean_err={mean_err:6.2f}  mean_err(parsed)={mean_parsed:6.2f}  "
          f"median={median:6.2f}  ->  {out_csv}")
    return out_csv


def main():
    ap = argparse.ArgumentParser(description="Parse RA/Dec answers and score angular error.")
    ap.add_argument("--indir", default="recall")
    ap.add_argument("--outdir", default="recall")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.indir, "radec_answers_*.csv")))
    if not files:
        print("no radec_answers_*.csv found in", args.indir)
        return
    for p in files:
        score_file(p, args.outdir)


if __name__ == "__main__":
    main()
