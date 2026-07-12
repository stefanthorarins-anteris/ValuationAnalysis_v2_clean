"""
Structured run logging for the deep-fetch / delisted-ingestion run
(investment-filter restructure, 2026-07-12; CEO explicit requirement).

WHY THIS EXISTS
---------------
The delisted-ingestion run acquires multi-GB of data over many hours.  The house
inspects the run WITHOUT pulling the data from Drive -- it reads a compact
run-summary MANIFEST that travels back via GitHub.  This module gives the fetch /
calc paths a small, dedicated logger that emits:

  1. a structured JSONL EVENT LOG   (run_logs/run_events_<id>.jsonl) -- one
     timestamped, leveled JSON object per predetermined event, in three buckets:
       * data   -- universe build, registry pagination, per-endpoint counts, dead
                   coverage, emptyfail set, filing-date fallbacks, bulk-price span +
                   dead-symbol presence, data-quality drops, rate/retry events;
       * calc   -- DCF finite/NaN + reason, beta computed/NaN, entity recycled/X_2
                   + death/acquisition band counts, scoring outputs, shrink
                   engagements, all-NaN metrics;
       * verify -- explicit PASS/FAIL + values: the two ride-along checks and the
                   as_of=None bit-for-bit check.
  2. a compact MANIFEST              (run_logs/run_manifest_<id>.json) -- counts,
     coverage figures, verification PASS/FAILs, key distributions.  Small enough
     to commit to GitHub separately from the data.

HARD SECURITY RULE
------------------
The FMP API key must NEVER appear in any log line.  Every value written is passed
through `_scrub`, which replaces any registered secret substring with '***'.  Call
sites must ALSO avoid logging full URLs with the key (log endpoint paths, not keyed
URLs) -- the scrubber is the last line of defence, not the first.
"""
import json
import os
from datetime import datetime, timezone


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class RunLogger:
    """A tiny append-only JSONL logger + manifest accumulator.

    Parameters
    ----------
    run_id : str          -- stamps the event log + manifest filenames.
    out_dir : str         -- directory for run_logs (created if absent).  These
                             files travel via GitHub; keep them out of the Drive
                             data dir.
    secrets : iterable    -- strings that must never appear in a log line (the API
                             key above all).  Each occurrence is replaced by '***'.
    echo : bool           -- also print each event to stdout (operator visibility).
    """

    def __init__(self, run_id, out_dir="run_logs", secrets=(), echo=True):
        self.run_id = run_id
        self.out_dir = out_dir
        self.echo = echo
        # keep only non-empty secrets, longest first so a longer secret is scrubbed
        # before a substring of it
        self._secrets = sorted({s for s in secrets if s}, key=len, reverse=True)
        os.makedirs(out_dir, exist_ok=True)
        self.events_path = os.path.join(out_dir, f"run_events_{run_id}.jsonl")
        self.manifest_path = os.path.join(out_dir, f"run_manifest_{run_id}.json")
        self._fh = open(self.events_path, "a", encoding="utf-8")
        # manifest accumulator
        self._summary = {
            "run_id": run_id,
            "started": _now_iso(),
            "finished": None,
            "counts": {},
            "coverage": {},
            "distributions": {},
            "verifications": [],
            "warnings": [],
            "errors": [],
        }

    # ---- secret scrubbing ------------------------------------------------- #
    def _scrub(self, obj):
        """Recursively replace any registered secret substring with '***'."""
        if isinstance(obj, str):
            s = obj
            for sec in self._secrets:
                if sec in s:
                    s = s.replace(sec, "***")
            return s
        if isinstance(obj, dict):
            return {self._scrub(k): self._scrub(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._scrub(v) for v in obj]
        return obj

    # ---- core event emit -------------------------------------------------- #
    def log(self, bucket, event, level="INFO", **fields):
        rec = {
            "ts": _now_iso(),
            "run_id": self.run_id,
            "level": level,
            "bucket": bucket,
            "event": event,
        }
        rec.update(fields)
        rec = self._scrub(rec)
        line = json.dumps(rec, default=str)
        self._fh.write(line + "\n")
        self._fh.flush()
        if level in ("ERROR", "WARN", "WARNING"):
            tgt = self._summary["errors"] if level == "ERROR" else self._summary["warnings"]
            tgt.append({"event": event, **{k: rec.get(k) for k in fields}})
        if self.echo:
            print(f"[{rec['ts']}][{level}][{bucket}] {event} "
                  + " ".join(f"{k}={rec.get(k)}" for k in fields), flush=True)
        return rec

    # bucket helpers
    def data(self, event, level="INFO", **f):
        return self.log("data", event, level=level, **f)

    def calc(self, event, level="INFO", **f):
        return self.log("calc", event, level=level, **f)

    def verify(self, event, passed, values=None, **f):
        """A verification/invariant event: explicit PASS/FAIL + the values behind it."""
        self._summary["verifications"].append(
            self._scrub({"event": event, "pass": bool(passed), "values": values or {}}))
        return self.log("verify", event, level=("INFO" if passed else "ERROR"),
                        result=("PASS" if passed else "FAIL"), values=values, **f)

    # ---- manifest accumulators ------------------------------------------- #
    def set_count(self, key, value):
        self._summary["counts"][key] = value

    def incr(self, key, n=1):
        self._summary["counts"][key] = self._summary["counts"].get(key, 0) + n

    def set_coverage(self, key, value):
        self._summary["coverage"][key] = self._scrub(value)

    def set_distribution(self, key, value):
        self._summary["distributions"][key] = self._scrub(value)

    def write_manifest(self):
        self._summary["finished"] = _now_iso()
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._scrub(self._summary), f, indent=2, default=str)
        if self.echo:
            print(f"[manifest] wrote {self.manifest_path}", flush=True)
        return self.manifest_path

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass
