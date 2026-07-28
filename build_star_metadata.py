#!/usr/bin/env python
"""
Collect per-object metadata for the Decodable-Sky object list.

  distance_ly : for every STAR, from SIMBAD trigonometric parallax
                distance[ly] = 3.261563777 * 1000 / parallax[mas].
  zipf        : text-corpus frequency (wordfreq.zipf_frequency, English) of the
                object's plain name. The appended " constellation" tag is stripped
                first, so a constellation is scored by its real name
                (e.g. "Andromeda constellation" -> zipf("Andromeda")).
  Vmag        : apparent V magnitude for stars, read from the curated object list
                (notes="Vmag=.."); SIMBAD's V is used only as a fallback if missing.
  type        : star / constellation / other, carried through.

Input : astro_test_set_v2.csv  (name, type, notes)
Output: astro_metadata_full.csv (name, type, Vmag, distance_ly, zipf)

The SIMBAD query uses only the standard library (urllib) via the sim-script
interface, which resolves common names (Sirius, Toliman, Rigil Kentaurus, ...).
Distance is collected for stars; constellations/"other" have no stellar parallax
and are left blank (they are not used in any distance correlation).
"""
import csv
import os
import sys
import time
import urllib.parse
import urllib.request

from wordfreq import zipf_frequency

SIMBAD = "https://simbad.cds.unistra.fr/simbad/sim-script"
PC_TO_LY = 3.261563777
# object catalog with name, type and a "notes" column (Vmag=.. for stars); the
# canonical repo file lives in data/, but a flat copy in the cwd also works.
IN_FILE = next((p for p in ("data/astro_objects.csv", "astro_objects.csv",
                            "astro_test_set_v2.csv") if os.path.exists(p)),
               "data/astro_objects.csv")
OUT_FILE = "astro_metadata_full.csv"
HEADER = ('output console=off script=off\n'
          'format object f1 "%IDLIST(1)|%PLX(V)|%FLUXLIST(V;F)"\n')   # 2 header lines


def _split_data(txt):
    """sim-script POST returns bare data lines when all queries resolve, or a
    ::error:: + ::data:: structure when some fail. Return (error_text, data_text)."""
    if "::data:" in txt:
        err, data = txt.split("::data:", 1)
        return err, (data.split("\n", 1)[1] if "\n" in data else "")
    return "", txt                                     # no failures -> all lines are data


def _f(s):
    s = (s or "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _parse_line(ln):
    p = [x.strip() for x in ln.split("|")]
    return (_f(p[1]) if len(p) > 1 else None,   # parallax [mas]
            _f(p[2]) if len(p) > 2 else None,    # V mag
            p[0] if p else "")                   # resolved main id


def _query(script, timeout):
    """POST the script (avoids GET URL-length limits for big batches)."""
    data = urllib.parse.urlencode({"script": script}).encode()
    req = urllib.request.Request(SIMBAD, data=data)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _one(name, timeout):
    try:
        _err, body = _split_data(_query(HEADER + f"query id {name}\n", timeout))
        for ln in body.splitlines():
            if "|" in ln:
                return _parse_line(ln)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] {name}: {e}", file=sys.stderr)
    return (None, None, "")


def simbad_lookup(names, timeout=120):
    """{name: (parallax_mas, Vmag, main_id)} via a single sim-script batch call."""
    err, data = _split_data(_query(HEADER + "".join(f"query id {n}\n" for n in names), timeout))
    failed = set()
    for line in err.splitlines():
        line = line.strip()
        if line.startswith("[") and "]" in line:
            try:
                failed.add(int(line[1:line.index("]")]) - 3)   # query idx = script line - 2 headers - 1
            except ValueError:
                pass
    data_lines = [ln for ln in data.splitlines() if "|" in ln]
    ok = [i for i in range(len(names)) if i not in failed]
    out = {}
    if len(data_lines) == len(ok):                       # clean batch mapping
        for i, ln in zip(ok, data_lines):
            out[names[i]] = _parse_line(ln)
    else:                                                # ambiguous -> per-name fallback
        print("  [info] batch mapping ambiguous; querying names individually", flush=True)
        for n in names:
            out[n] = _one(n, timeout)
            time.sleep(0.05)
    for n in names:
        out.setdefault(n, (None, None, ""))
    return out


def parse_vmag(notes):
    n = (notes or "").strip()
    if n.startswith("Vmag="):
        try:
            return float(n[5:])
        except ValueError:
            return None
    return None


def zipf_of(name, typ):
    base = name
    if typ == "constellation" and name.endswith(" constellation"):
        base = name[:-len(" constellation")]
    return round(zipf_frequency(base.strip(), "en"), 2)


def main():
    objs = list(csv.DictReader(open(IN_FILE, encoding="utf-8")))
    stars = [r["name"] for r in objs if r["type"] == "star"]
    print(f"{len(objs)} objects, {len(stars)} stars -> querying SIMBAD ...", flush=True)
    sim = simbad_lookup(stars)

    rows, missing = [], []
    for r in objs:
        nm, typ = r["name"], r["type"]
        vmag = parse_vmag(r.get("notes"))
        dist = ""
        if typ == "star":
            plx, simv, _mid = sim.get(nm, (None, None, ""))
            if vmag is None and simv is not None:
                vmag = simv
            if plx and plx > 0:
                dist = round(PC_TO_LY * 1000.0 / plx, 3)
            else:
                missing.append(nm)
        rows.append([nm, typ, "" if vmag is None else vmag, dist, zipf_of(nm, typ)])

    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "type", "Vmag", "distance_ly", "zipf"])
        w.writerows(rows)

    nstar = sum(1 for r in rows if r[1] == "star")
    ndist = sum(1 for r in rows if r[1] == "star" and r[3] != "")
    print(f"saved {OUT_FILE}: {len(rows)} objects; stars with distance {ndist}/{nstar}")
    if missing:
        print("stars WITHOUT a positive parallax (blank distance):", missing)


if __name__ == "__main__":
    main()
