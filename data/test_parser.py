"""
data/test_parser.py
-------------------
Tests the HistData/Dukascopy parser against all supported format variants.
Run: python data/test_parser.py
"""

import sys, os, io, tempfile, textwrap
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.histdata_parser import parse_histdata, _detect_format, _read_file
import pandas as pd

PASS = "✓"
FAIL = "✗"

results = []

# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_tmp(content: str, suffix=".csv") -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix,
                                   delete=False, encoding="utf-8")
    f.write(content)
    f.flush()
    return f.name

def _check(name: str, df: pd.DataFrame, expected_rows: int = None):
    try:
        assert not df.empty,              "DataFrame is empty"
        assert df.index.tz is not None,   "Index is not timezone-aware"
        assert "close" in df.columns,     "Missing 'close' column"
        assert (df["high"] >= df["low"]).all(), "High < Low in some rows"
        if expected_rows:
            assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"
        results.append((PASS, name, f"{len(df)} rows, {df.index[0].date()} → {df.index[-1].date()}"))
    except AssertionError as e:
        results.append((FAIL, name, str(e)))

# ── Test 1: HistData M1 semicolon format ─────────────────────────────────────
csv1 = textwrap.dedent("""\
    20220103 000000;1.13218;1.13221;1.13200;1.13205;120
    20220103 000100;1.13205;1.13230;1.13200;1.13220;95
    20220103 000200;1.13220;1.13240;1.13215;1.13235;110
    20220103 000300;1.13235;1.13250;1.13220;1.13240;88
    20220103 000400;1.13240;1.13260;1.13230;1.13255;143
""")
path1 = _write_tmp(csv1)
try:
    df1 = parse_histdata(path1, symbol="EURUSD", target_tf="M1")
    _check("HistData M1 semicolon", df1, 5)
except Exception as e:
    results.append((FAIL, "HistData M1 semicolon", str(e)))

# ── Test 2: HistData D1 format ────────────────────────────────────────────────
csv2 = textwrap.dedent("""\
    20220103;1.13218;1.13300;1.13100;1.13205;12000
    20220104;1.13205;1.13450;1.13150;1.13380;15000
    20220105;1.13380;1.13500;1.13200;1.13280;13500
""")
path2 = _write_tmp(csv2)
try:
    df2 = parse_histdata(path2, symbol="EURUSD", target_tf="M1")
    _check("HistData D1 semicolon", df2, 3)
except Exception as e:
    results.append((FAIL, "HistData D1 semicolon", str(e)))

# ── Test 3: Dukascopy JForex with header ──────────────────────────────────────
csv3 = textwrap.dedent("""\
    Time (UTC),Open,High,Low,Close,Volume
    03.01.2022 00:00:00,1.13218,1.13221,1.13200,1.13205,120
    03.01.2022 00:01:00,1.13205,1.13230,1.13200,1.13220,95
    03.01.2022 00:02:00,1.13220,1.13240,1.13215,1.13235,110
""")
path3 = _write_tmp(csv3)
try:
    df3 = parse_histdata(path3, symbol="EURUSD", target_tf="M1")
    _check("Dukascopy Time(UTC) header", df3, 3)
except Exception as e:
    results.append((FAIL, "Dukascopy Time(UTC) header", str(e)))

# ── Test 4: Dukascopy GMT time header ─────────────────────────────────────────
csv4 = textwrap.dedent("""\
    Gmt time,Open,High,Low,Close,Volume
    03.01.2022 00:00:00.000,1.13218,1.13221,1.13200,1.13205,120.0
    03.01.2022 00:01:00.000,1.13205,1.13230,1.13200,1.13220,95.0
    03.01.2022 00:02:00.000,1.13220,1.13240,1.13215,1.13235,110.0
    03.01.2022 00:03:00.000,1.13235,1.13250,1.13225,1.13245,88.0
""")
path4 = _write_tmp(csv4)
try:
    df4 = parse_histdata(path4, symbol="GBPUSD", target_tf="M1")
    _check("Dukascopy Gmt time header", df4, 4)
except Exception as e:
    results.append((FAIL, "Dukascopy Gmt time header", str(e)))

# ── Test 5: MT5 tab-delimited export ──────────────────────────────────────────
csv5 = textwrap.dedent("""\
    <DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>
    2022.01.03\t00:00\t1.13218\t1.13221\t1.13200\t1.13205\t120\t0\t1
    2022.01.03\t00:05\t1.13205\t1.13230\t1.13200\t1.13220\t95\t0\t1
    2022.01.03\t00:10\t1.13220\t1.13240\t1.13215\t1.13235\t110\t0\t1
""")
path5 = _write_tmp(csv5)
try:
    df5 = parse_histdata(path5, symbol="EURUSD", target_tf="M1")
    _check("MT5 tab-delimited export", df5, 3)
except Exception as e:
    results.append((FAIL, "MT5 tab-delimited export", str(e)))

# ── Test 6: Resampling M1 → M5 ───────────────────────────────────────────────
# 10 M1 bars at 00:00-00:09 → 2 M5 bars (00:00 covers 00:00-00:04, 00:05 covers 00:05-00:09)
csv6_lines = []
for i in range(10):
    hh = i // 60
    mm = i % 60
    ts = f"20220103 {hh:02d}{mm:02d}00"
    p  = 1.13200 + i * 0.00001
    csv6_lines.append(f"{ts};{p:.5f};{p+0.00010:.5f};{p-0.00010:.5f};{p+0.00005:.5f};100")
csv6 = "\n".join(csv6_lines)
path6 = _write_tmp(csv6)
try:
    df6 = parse_histdata(path6, symbol="EURUSD", target_tf="M5")
    _check("M1→M5 resampling", df6, 2)
except Exception as e:
    results.append((FAIL, "M1→M5 resampling", str(e)))

# ── Test 7: Date range filter ──────────────────────────────────────────────────
csv7 = textwrap.dedent("""\
    20220101 000000;1.13000;1.13100;1.12900;1.13050;100
    20220102 000000;1.13050;1.13150;1.12950;1.13100;110
    20220103 000000;1.13100;1.13200;1.13000;1.13150;120
    20220104 000000;1.13150;1.13250;1.13050;1.13200;130
    20220105 000000;1.13200;1.13300;1.13100;1.13250;140
""")
path7 = _write_tmp(csv7)
try:
    df7 = parse_histdata(path7, symbol="EURUSD", target_tf="M1",
                          start="2022-01-02", end="2022-01-04")
    _check("Date range filter", df7, 3)
except Exception as e:
    results.append((FAIL, "Date range filter", str(e)))

# ── Test 8: Spike / corrupt row filtering ────────────────────────────────────
# Bar 2 has H=99.99 which is massively beyond normal range.
# With 6 rows the rolling(20,min_periods=1) median ≈ 0.00020; 50× = 0.01.
# The spike bar H-L = 98.86 >> 0.01, so it must be removed.
csv8_rows = [
    "20220103 000000;1.13218;1.13221;1.13200;1.13205;120",
    "20220103 000100;1.13205;99.99000;1.13200;1.13220;95",   # spike
    "20220103 000200;1.13220;1.13240;1.13215;1.13235;110",
    "20220103 000300;1.13235;1.13250;1.13225;1.13240;80",
    "20220103 000400;1.13240;1.13260;1.13230;1.13255;90",
    "20220103 000500;1.13255;1.13270;1.13245;1.13260;75",
]
csv8 = "\n".join(csv8_rows)
path8 = _write_tmp(csv8)
try:
    df8 = parse_histdata(path8, symbol="EURUSD", target_tf="M1")
    assert (df8["high"] < 2.0).all(), \
        f"Spike row (H=99.99) not removed. Max high={df8['high'].max()}"
    results.append((PASS, "Spike row filtering",
                    f"{len(df8)}/6 rows kept (spike at H=99.99 removed)"))
except Exception as e:
    results.append((FAIL, "Spike row filtering", str(e)))

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "═"*62)
print("  HistData Parser Tests")
print("═"*62)
for icon, name, detail in results:
    print(f"  {icon}  {name:<38} {detail}")
print("═"*62)
passed = sum(1 for r in results if r[0] == PASS)
total  = len(results)
print(f"  Result: {passed}/{total} passed")
print("═"*62 + "\n")

# Cleanup
import os
for p in [path1, path2, path3, path4, path5, path6, path7, path8]:
    try: os.unlink(p)
    except: pass

sys.exit(0 if passed == total else 1)
