"""The polling account pool: token reuse, health checks, rotation, capped registration.

Accounts live in the private repository as ``state/accounts/<name>.json``. Each job
uses the account whose ``role`` matches, because a login invalidates the account's
other sessions. See docs/community-stats/03-secrets.md in the servoom repository.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import rawrepo
from .api import Api, ApiDown

EMAIL_DOMAIN = os.environ.get("ACCOUNT_EMAIL_DOMAIN", "servoom.invalid")
REGISTER_EVERY = 7 * 86400
ISSUES: List[str] = []          # markdown bodies; the workflow turns them into GitHub issues


class Halted(Exception):
    pass


def _mask(*values) -> None:
    if os.environ.get("GITHUB_ACTIONS"):
        for v in values:
            if v:
                print(f"::add-mask::{v}")


def note_issue(title: str, body: str) -> None:
    ISSUES.append(f"{title}\n\n{body}")
    print(f"[issue] {title}")


def flush_issues(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(ISSUES):
        (out_dir / f"{int(time.time())}-{i}.md").write_text(text, encoding="utf-8")


class Pool:
    def __init__(self, main: Path):
        self.dir = main / "state" / "accounts"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pool_file = main / "state" / "pool.json"
        self.health_file = main / "state" / "health.json"

    # -- files --------------------------------------------------------------
    def accounts(self) -> List[Dict]:
        out = []
        for p in sorted(self.dir.glob("*.json")):
            a = json.loads(p.read_text())
            a["_name"] = p.stem
            out.append(a)
        return out

    def save(self, a: Dict) -> None:
        d = {k: v for k, v in a.items() if not k.startswith("_")}
        (self.dir / f"{a['_name']}.json").write_text(json.dumps(d, indent=1))

    def health(self) -> Dict:
        return json.loads(self.health_file.read_text()) if self.health_file.exists() else {"halted": False}

    def halt(self, reason: str) -> None:
        self.health_file.write_text(json.dumps({"halted": True, "reason": reason, "since": int(time.time())}))
        note_issue("Polling halted", f"The pipeline stopped itself: {reason}.\n\n"
                   "Jobs exit early until `state/health.json` in the raw repository is reset to "
                   "`{\"halted\": false}`.")

    # -- checks -------------------------------------------------------------
    @staticmethod
    def _login(api: Api, a: Dict) -> bool:
        r = api.post("UserLogin", {"Email": a["email"], "Password": a["md5"]}, auth=False)
        if r.get("ReturnCode") == 0 and r.get("Token"):
            a["token"], a["user_id"], a["token_at"] = r["Token"], r["UserId"], int(time.time())
            _mask(a["token"])
            return True
        return False

    @staticmethod
    def _canary(api: Api) -> bool:
        """A page past the anonymous cap must be non-empty, and a like list must answer."""
        r = api.post("GetCategoryFileListV2", {"Classify": 1, "FileSize": 1, "FileType": 5, "FileSort": 0,
                                               "Version": 19, "StartNum": 3001, "EndNum": 3030})
        if r.get("ReturnCode") != 0 or not r.get("FileList"):
            return False
        gid = r["FileList"][0]["GalleryId"]
        return api.post("Cloud/GetLikeUserList", {"GalleryId": gid, "StartNum": 1, "EndNum": 1}).get("ReturnCode") == 0

    def _try(self, api: Api, a: Dict) -> bool:
        _mask(a.get("password"), a.get("md5"), a.get("email"), a.get("token"))
        if a.get("token"):
            api.auth = {"Token": a["token"], "UserId": a["user_id"]}
            if self._canary(api):
                return True
        if self._login(api, a):
            api.auth = {"Token": a["token"], "UserId": a["user_id"]}
            if self._canary(api):
                return True
        api.auth = {}
        return False

    def _register(self, api: Api, role: str) -> Optional[Dict]:
        pool = json.loads(self.pool_file.read_text()) if self.pool_file.exists() else {}
        if time.time() - pool.get("last_registration", 0) < REGISTER_EVERY:
            return None
        pw = secrets.token_urlsafe(18)
        a = {"email": f"stats-{secrets.token_hex(5)}@{EMAIL_DOMAIN}", "password": pw,
             "md5": hashlib.md5(pw.encode()).hexdigest(), "role": role, "status": "ok", "strikes": 0,
             "created": time.strftime("%Y-%m-%d"), "_name": f"auto-{time.strftime('%Y%m%d%H%M')}"}
        _mask(a["password"], a["md5"], a["email"])
        r = api.post("UserRegister", {"Email": a["email"], "Password": a["md5"], "CountryISOCode": "US",
                                      "Language": "en", "TimeZone": "America/New_York"}, auth=False)
        pool["last_registration"] = int(time.time())
        self.pool_file.write_text(json.dumps(pool))
        if r.get("ReturnCode") != 0:
            return None
        self.save(a)
        note_issue("A polling account was registered automatically",
                   f"The pool had no working account for the `{role}` job, so one new account was "
                   "registered. At most one is registered per 7 days.")
        return a

    # -- entry point --------------------------------------------------------
    def acquire(self, api: Api, role: str) -> Dict:
        """Authenticate ``api`` for a job. Raises Halted when no account can be used."""
        if self.health().get("halted"):
            raise Halted(self.health().get("reason", "halted"))
        accts = self.accounts()
        mine = [a for a in accts if a.get("role") == role and a.get("status") == "ok"]
        spares = [a for a in accts if a.get("role") == "spare" and a.get("status") == "ok"]
        for a in mine:
            if self._try(api, a):                      # ApiDown propagates: not the account's fault
                a["strikes"], a["last_ok"] = 0, int(time.time())
                self.save(a)
                return a
            a["strikes"] = a.get("strikes", 0) + 1
            if a["strikes"] >= 3:
                a["status"] = "bad"
                note_issue(f"Polling account for `{role}` was retired",
                           "It failed its health check on three runs in a row. A spare takes over.")
            self.save(a)
            if a["status"] == "ok":
                try:                   # keep the strike: three in a row retire the account
                    rawrepo.push_main(f"{role}: account strike")
                except Exception as exc:
                    print(f"[accounts] could not push the strike: {exc!r}")
                raise ApiDown("account failed its health check; will retry next run")
        for a in spares:
            if self._try(api, a):
                a["role"], a["strikes"], a["last_ok"] = role, 0, int(time.time())
                self.save(a)
                return a
            a["status"] = "bad"
            self.save(a)
        a = self._register(api, role)
        if a is not None and self._try(api, a):
            a["last_ok"] = int(time.time())
            self.save(a)
            return a
        self.halt(f"no working account for the {role} job, and "
                  + ("the newly registered account failed too" if a is not None
                     else "registration is capped at one per 7 days or was refused"))
        raise Halted("no account")
