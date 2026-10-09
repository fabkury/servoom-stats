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
REGISTER_EVERY = 3 * 86400       # at most one new account per 3 days
SPARES = 3                       # healthy spares the pool keeps ready
ISSUES: List[str] = []          # markdown bodies; the workflow turns them into GitHub issues


class Halted(Exception):
    pass


class RegistrationFailed(Exception):
    """Divoom refused /UserRegister."""


def _mask(*values) -> None:
    if os.environ.get("GITHUB_ACTIONS"):
        for v in values:
            if v:
                print(f"::add-mask::{v}")


def note_issue(title: str, body: str, error: bool = False) -> None:
    """Queue a GitHub issue; with ``error`` the run is also annotated as failed-looking."""
    ISSUES.append(f"{title}\n\n{body}")
    print(f"[issue] {title}")
    if error:
        print(f"::error::{title}")


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
    def _canary(api: Api) -> str:
        """'ok', 'capped' (a page past the anonymous cap is empty although the server
        answers ReturnCode 0) or 'fail' (a listing or a like list is refused)."""
        r = api.post("GetCategoryFileListV2", {"Classify": 1, "FileSize": 1, "FileType": 5, "FileSort": 0,
                                               "Version": 19, "StartNum": 3001, "EndNum": 3030})
        if r.get("ReturnCode") != 0:
            return "fail"
        if not r.get("FileList"):
            return "capped"
        gid = r["FileList"][0]["GalleryId"]
        ok = api.post("Cloud/GetLikeUserList", {"GalleryId": gid, "StartNum": 1, "EndNum": 1}).get("ReturnCode") == 0
        return "ok" if ok else "fail"

    def _try(self, api: Api, a: Dict) -> str:
        """Authenticate ``api`` as ``a``: 'ok', 'capped' or 'fail'. A stale token also looks
        capped, so the cap is only reported after a fresh login shows the same."""
        _mask(a.get("password"), a.get("md5"), a.get("email"), a.get("token"))
        if a.get("token"):
            api.auth = {"Token": a["token"], "UserId": a["user_id"]}
            if self._canary(api) == "ok":
                return "ok"
        if self._login(api, a):
            api.auth = {"Token": a["token"], "UserId": a["user_id"]}
            c = self._canary(api)
            if c in ("ok", "capped"):
                return c
        api.auth = {}
        return "fail"

    def _register(self, api: Api, role: str) -> Optional[Dict]:
        """A new account with ``role``, or None when the registration cap is in force.
        Raises RegistrationFailed when Divoom refuses; the refusal still counts against
        the cap, so a refusing server is asked again only after REGISTER_EVERY."""
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
            raise RegistrationFailed(f"UserRegister answered {r.get('ReturnCode')} {r.get('ReturnMessage', '')}".strip())
        self.save(a)
        note_issue("A polling account was registered automatically",
                   f"One new account was registered with role `{role}`"
                   + (" because the pool had no working account for that job." if role != "spare"
                      else f" to keep {SPARES} spares ready.")
                   + f" At most one is registered per {REGISTER_EVERY // 86400} days.")
        return a

    def _retire(self, a: Dict, role: str, why: str) -> None:
        a["status"] = "bad"
        self.save(a)
        note_issue(f"Polling account for `{role}` was retired", why + " A spare takes over.")

    def top_up(self, api: Api) -> None:
        """Keep SPARES healthy spares, registering at most one per REGISTER_EVERY. A
        registration that is refused, or a new account that fails its health check,
        opens an issue marked as an error. ``api.auth`` is left as it was."""
        healthy = [a for a in self.accounts() if a.get("role") == "spare" and a.get("status") == "ok"]
        if len(healthy) >= SPARES:
            return
        auth = dict(api.auth)
        try:
            a = self._register(api, "spare")
            if a is None:
                return
            if self._try(api, a) != "ok":
                a["status"] = "bad"
                self.save(a)
                note_issue("A newly registered account failed its health check",
                           f"The pool has {len(healthy)} of {SPARES} spares. The account registered to fill "
                           "it logged in but did not pass the canary (deep listing page and like list), so "
                           "it was marked bad. Divoom may be restricting new accounts.", error=True)
        except RegistrationFailed as exc:
            note_issue("Account registration was refused",
                       f"The pool has {len(healthy)} of {SPARES} spares and `/UserRegister` was refused: "
                       f"{exc}. The next attempt is in {REGISTER_EVERY // 86400} days.", error=True)
        finally:
            api.auth = auth

    # -- entry point --------------------------------------------------------
    def acquire(self, api: Api, role: str) -> Dict:
        """Authenticate ``api`` for a job. Raises Halted when no account can be used."""
        if self.health().get("halted"):
            raise Halted(self.health().get("reason", "halted"))
        accts = self.accounts()
        mine = [a for a in accts if a.get("role") == role and a.get("status") == "ok"]
        spares = [a for a in accts if a.get("role") == "spare" and a.get("status") == "ok"]
        for a in mine:
            c = self._try(api, a)                      # ApiDown propagates: not the account's fault
            if c == "ok":
                a["strikes"], a["last_ok"] = 0, int(time.time())
                self.save(a)
                self.top_up(api)
                return a
            if c == "capped":                          # certain after a fresh login: retire at once
                self._retire(a, role, "Divoom answers its listing pages past the anonymous depth as "
                                      "empty although the login succeeds, so it can no longer poll.")
                continue
            a["strikes"] = a.get("strikes", 0) + 1
            if a["strikes"] >= 3:
                self._retire(a, role, "It failed its health check on three runs in a row.")
                continue
            self.save(a)
            try:                   # keep the strike: three in a row retire the account
                rawrepo.push_main(f"{role}: account strike")
            except Exception as exc:
                print(f"[accounts] could not push the strike: {exc!r}")
            raise ApiDown("account failed its health check; will retry next run")
        for a in spares:
            if self._try(api, a) == "ok":
                a["role"], a["strikes"], a["last_ok"] = role, 0, int(time.time())
                self.save(a)
                self.top_up(api)
                return a
            a["status"] = "bad"
            self.save(a)
        try:
            a = self._register(api, role)
        except RegistrationFailed as exc:
            a = None
            note_issue("Account registration was refused", f"`/UserRegister` answered: {exc}", error=True)
        if a is not None and self._try(api, a) == "ok":
            a["last_ok"] = int(time.time())
            self.save(a)
            return a
        self.halt(f"no working account for the {role} job, and "
                  + ("the newly registered account failed too" if a is not None
                     else f"registration is capped at one per {REGISTER_EVERY // 86400} days or was refused"))
        raise Halted("no account")
