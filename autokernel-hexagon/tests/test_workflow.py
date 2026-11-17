"""End-to-end state-machine tests on a scripted fake device (no hardware).

Verifies: adoption, rejection + restore, compile/device failures and missing
timings keep best and log consistent, noise handling, host-regression hold,
interrupted runs, identical candidates, cumulative gain vs initial.
"""
import csv
import json
import shutil
from pathlib import Path

import pytest

from fake_device import FakeDeviceRunner
from search import Workspace
from akh.launch import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def timing(l):
    h = l.hexagon_opts
    dsp = 10.0
    if l.consts.get("USE_RSQRT"):
        dsp = 8.0
    if l.consts.get("BLOCK_H", 4096) < l.consts.get("H", 4096):
        dsp *= 1.2
    if h.get("fakeTinyGain"):
        dsp *= 0.997
    host = 100.0
    if h.get("fakeHostSlow"):
        dsp, host = 7.0, 150.0
    return dsp * l.consts.get("M", 1) ** 0.5, host


@pytest.fixture
def ws(tmp_path):
    shutil.copy2(ROOT / "kernel.py", tmp_path / "kernel.py")
    w = Workspace(tmp_path, "hexagon", runner=FakeDeviceRunner(timing), samples=4, max_rounds=3,
                  env_info={"target": "fake", "soc": "", "triton_rev": "t", "hexmlir_rev": "h", "sdk": ""})
    return w


def rows(ws):
    with ws.tsv.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_full_loop(ws):
    exp0, st, v = ws.baseline()
    assert st == "ok" and v == "initial_baseline"
    best0 = ws.best()
    assert best0["timed"] and best0["metrics"]["1x4096"]["dsp_us_median"] == pytest.approx(10, rel=0.02)

    # 1) improvement is adopted
    e1, st, v = ws.compare(ws.best_kernel(), {"use_rsqrt": True}, "grid", "rsqrt")
    assert v == "improved" and ws.best()["exp_id"] == e1

    # 2) regression is rejected, best unchanged
    e2, st, v = ws.compare(ws.best_kernel(), {"block_h": 1024}, "grid", "chunked")
    assert v == "keep_previous" and ws.best()["exp_id"] == e1

    # 3) compile failure: rejected, best unchanged, failure class logged
    e3, st, v = ws.compare(ws.best_kernel(), {"hexagon": {"fakeFail": "compile"}}, "grid", "boom")
    assert (st, v) == ("compile_fail", "rejected") and ws.best()["exp_id"] == e1

    # 4) device failure
    e4, st, v = ws.compare(ws.best_kernel(), {"hexagon": {"fakeFail": "device"}}, "grid", "adb")
    assert (st, v) == ("device_fail", "rejected")

    # 5) timing missing -> no numbers, excluded from ranking
    e5, st, v = ws.compare(ws.best_kernel(), {"hexagon": {"fakeFail": "no_timing"}}, "grid", "noperf")
    assert st == "timing_parse_fail" and ws.best()["exp_id"] == e1

    # 6) gain smaller than noise: extra rounds, then keep previous
    calls = ws.runner.calls
    e6, st, v = ws.compare(ws.best_kernel(), {"hexagon": {"fakeTinyGain": True}}, "grid", "tiny")
    assert v == "keep_previous"
    assert ws.runner.calls - calls > 2 * 4 * 2  # more than one round was measured

    # 7) DSP better but host worse -> hold, not adopted; manual adopt after investigation
    e7, st, v = ws.compare(ws.best_kernel(), {"hexagon": {"fakeHostSlow": True}}, "grid", "hostslow")
    assert v == "hold_host_regression" and ws.best()["exp_id"] == e1
    ws.adopt(e7, "host cost is adb push of a larger .so; DSP-side win is real")
    assert ws.best()["exp_id"] == e7

    # 8) identical source -> no measurement
    calls = ws.runner.calls
    e8, st, v = ws.compare(ws.best_kernel(), None, "agent", "same")
    assert v == "keep_previous" and ws.runner.calls == calls  # nothing re-run

    # log consistency
    r = rows(ws)
    by = {}
    for x in r:
        by.setdefault(x["exp_id"], []).append(x)
    assert len(by[e3]) == 2 and all(x["dsp_us_median"] == "" for x in by[e3] + by[e4] + by[e5])
    assert all(x["adopted"] == "yes" for x in by[e1])
    assert by[e1][0]["dsp_gain_vs_initial"] and float(by[e1][0]["dsp_gain_vs_initial"]) == pytest.approx(0.2, abs=0.03)
    assert by[e3][0]["status"] == "compile_fail" and by[e4][0]["status"] == "device_fail"
    adopted = [x["exp_id"] for x in r if x["adopted"] == "yes"]
    assert adopted[-1] == ws.best()["exp_id"]
    # best kernel file equals the adopted experiment snapshot
    assert file_sha256(ws.best_kernel()) == file_sha256(ws.runs / e7 / "kernel.py")


def test_agent_try_accuracy_failure_restores_kernel(ws):
    ws.baseline()
    good = ws.kernel.read_text()
    # agent "optimisation" that drops the fp32 upcast -> x*x overflows in fp16 on the large case
    bad = good.replace("x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)\n            ms",
                       "x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)\n            ms", 1)
    assert bad != good
    ws.kernel.write_text(bad)
    e, st, v = ws.compare(ws.kernel, None, "agent", "skip fp32 upcast")
    assert (st, v) == ("accuracy_fail", "rejected")
    res = json.loads((ws.runs / e / "result.json").read_text())
    failed = [a["case"] for a in res["accuracy"] if a.get("ok") is False]
    assert "val_large_1x4096" in failed
    assert (ws.runs / e / "diff.patch").read_text().count("-") > 0
    ws.restore(backup=False)
    assert ws.kernel.read_text() == good


def test_interrupted_experiment_is_recorded_and_best_kept(ws):
    ws.baseline()
    best = ws.best()["exp_id"]
    (ws.state / "inflight.json").write_text(json.dumps({"exp_id": "hex9999_grid", "kind": "grid", "hypothesis": "x"}))
    ws.compare(ws.best_kernel(), {"use_rsqrt": True}, "grid", "rsqrt")
    r = rows(ws)
    assert any(x["exp_id"] == "hex9999_grid" and x["status"] == "interrupted" for x in r)
    assert ws.best()["exp_id"] != best  # the new valid experiment still works


def test_user_edits_are_backed_up(ws):
    ws.baseline()
    ws.kernel.write_text(ws.kernel.read_text() + "\n# user edit\n")
    backup = ws.restore()
    assert backup and "# user edit" in Path(backup).read_text()


def test_final_reports_cumulative(ws, capsys):
    ws.baseline()
    ws.compare(ws.best_kernel(), {"use_rsqrt": True}, "grid", "rsqrt")
    ws.final()
    out = capsys.readouterr().out
    assert "[1x4096] initial" in out
    fin = [x for x in rows(ws) if x["kind"] == "final"]
    assert len(fin) == 2 and float(fin[0]["dsp_gain_vs_initial"]) == pytest.approx(0.2, abs=0.03)
