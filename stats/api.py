"""Throttled, retrying access to the Divoom cloud API. Read-only commands only."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Iterable, List, Optional

import requests

HOST = "https://appin.divoom-gz.com"
FILE_HOST = "https://f.divoom-gz.com"

# A failed request is retried with backoff capped at RETRY_CAP seconds. The job gives up
# only when the whole client has had no good answer for OUTAGE seconds, or one request
# has failed for that long on its own. Before 2026-10-09 five attempts and a shared
# streak of 40 failures were the limit, and a server hiccup of about a minute ended a
# three-hour crawl.
OUTAGE = float(os.environ.get("API_OUTAGE_SECONDS", str(15 * 60)))
RETRY_CAP = 60.0
# Since 2026-10-23 about 01:30 UTC the server serves one source at most BUDGET requests
# per sliding window of a minute, whatever the command, token or connection, and answers
# the rest with HTTP 429 carrying the body of a ReturnCode 12 "Request data is null".
# Refused requests count against the window too, so a client that retries quickly
# starves itself. Measured from home on 2026-10-26 at 2 and 1 requests per second:
# 30 served, then refusals until a minute after the first of them. From 2026-10-12 to
# 2026-10-23 the budget was 120 per clock minute; see CLOUD_API.md.
BUDGET = int(os.environ.get("API_BUDGET", "30"))
WINDOW = float(os.environ.get("API_WINDOW_SECONDS", "62"))      # a little over the server's 60
LIMITED_WAIT = WINDOW          # after a refusal the client stops for a whole window
# Adaptive pacing (2026-10-27). Jobs on different runners can share one egress address,
# or the server may count by account as well as by address, so a client that holds the
# full budget keeps colliding and loses half of its requests. After a refusal the client
# halves its rate (up to SLOWDOWN_MAX times slower); every clean window afterwards
# restores half of the lost speed.
SLOWDOWN_MAX = float(os.environ.get("API_SLOWDOWN_MAX", "8"))

# Files come from a different host that has not shown a limit; they are paced apart.
FILE_INTERVAL = float(os.environ.get("API_FILE_INTERVAL", "0.25"))

# Commands this pipeline is allowed to send. Anything else is a programming error:
# the collector must never call a command that changes server state.
READ_ONLY = {
    "UserLogin", "UserRegister",
    "GetCategoryFileListV2", "Cloud/GalleryInfo", "Cloud/GetLikeUserList",
    "Comment/GetCommentListV3", "GetSomeoneInfoV2", "LookScore", "GetExpertListV4",
    "Cloud/GetMatchInfo", "Discover/GetTheme", "Discover/GetTopNew",
    "Discover/GetAlbumListV3", "Forum/GetList", "Mall/GetListV2", "GetStoreV2",
    "GetGalleryAdvert", "Lottery/Announce",
}


class ApiDown(Exception):
    """The server or the network is failing; not the account's fault."""


class OutOfTime(ApiDown):
    """The job's reading budget is spent; whatever was read so far is kept."""


class Api:
    """One paced client. ``rps`` can only lower the rate below the server's budget."""

    def __init__(self, rps: Optional[float] = None, workers: int = 4):
        self.s = requests.Session()
        self.interval = WINDOW / BUDGET
        if rps:
            self.interval = max(self.interval, 1.0 / rps)
        self.workers = workers
        self._lock = threading.Lock()
        self._next = 0.0
        self._next_file = 0.0
        self.auth: Dict = {}
        self.n_requests = 0            # attempts sent, refused ones included
        self.n_served = 0              # answers with a body
        self.n_limited = 0             # refusals (HTTP 429 or ReturnCode 12)
        self.n_files = 0
        self._last_ok = time.monotonic()
        self.deadline: Optional[float] = None
        self.slowdown = 1.0            # current factor on the interval between requests
        self.slowdown_peak = 1.0       # the worst factor reached during this run
        self._resume = 0.0             # when the hold after the last refusal ends
        self._last_adjust = time.monotonic()

    def set_budget(self, seconds: float) -> None:
        """Requests sent after ``seconds`` from now raise OutOfTime instead."""
        self.deadline = time.monotonic() + seconds

    @property
    def seconds_left(self) -> Optional[float]:
        return None if self.deadline is None else self.deadline - time.monotonic()

    def _wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            t = max(now, self._next)
            if self.deadline is not None and t > self.deadline:
                raise OutOfTime("reading budget spent")
            self._next = t + self.interval * self.slowdown
        if t > now:
            time.sleep(t - now)

    def _slow_down(self) -> None:
        """A refusal: halve the rate. Refusals that other workers meet during the same
        hold count as the same burst and do not halve it again. Recovery is measured
        from the end of the hold, so the slower rate holds for at least a window."""
        with self._lock:
            now = time.monotonic()
            if now >= self._resume:
                self.slowdown = min(self.slowdown * 2, SLOWDOWN_MAX)
                self.slowdown_peak = max(self.slowdown_peak, self.slowdown)
            self._resume = now + LIMITED_WAIT
            self._last_adjust = self._resume

    def _recover(self) -> None:
        """A served answer: after a whole clean window, restore half of the lost speed."""
        if self.slowdown == 1.0:
            return
        with self._lock:
            now = time.monotonic()
            if now - self._last_adjust >= WINDOW:
                self.slowdown = max(1.0, self.slowdown / 2)
                self._last_adjust = now

    def _hold(self, seconds: float) -> None:
        """Stop every worker for ``seconds``: the next request of any of them waits."""
        with self._lock:
            self._next = max(self._next, time.monotonic() + seconds)

    def post(self, cmd: str, body: Optional[Dict] = None, auth: bool = True) -> Dict:
        if cmd not in READ_ONLY:
            raise ValueError(f"command not on the read-only list: {cmd}")
        payload = {**(self.auth if auth else {}), **(body or {})}
        t0, attempt = time.monotonic(), 0
        while True:
            self._wait()
            try:
                r = self.s.post(f"{HOST}/{cmd}", json=payload, timeout=30)
                self.n_requests += 1
                if r.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                limited = r.status_code == 429
                data = None
                if not limited:
                    data = r.json()
                    limited = data.get("ReturnCode") == 12
                if limited:                      # over the window's budget: let it empty
                    self.n_limited += 1
                    if time.monotonic() - t0 > OUTAGE:
                        raise ApiDown(f"{cmd}: rate-limited for {OUTAGE:.0f}s")
                    self._hold(LIMITED_WAIT)
                    self._slow_down()
                    continue
                self._last_ok = time.monotonic()
                self.n_served += 1
                self._recover()
                return data
            except (requests.RequestException, ValueError) as exc:
                t = time.monotonic()
                if t - self._last_ok > OUTAGE or t - t0 > OUTAGE:
                    raise ApiDown(f"{cmd}: {exc}") from exc
                attempt += 1
                time.sleep(min(RETRY_CAP, 2.0 ** attempt))

    def _wait_file(self) -> None:
        with self._lock:
            now = time.monotonic()
            t = max(now, self._next_file)
            if self.deadline is not None and t > self.deadline:
                raise OutOfTime("reading budget spent")
            self._next_file = t + FILE_INTERVAL
        if t > now:
            time.sleep(t - now)

    def get_file(self, file_id: str) -> Optional[bytes]:
        for attempt in range(3):
            self._wait_file()
            try:
                r = self.s.get(f"{FILE_HOST}/{file_id}", timeout=60)
                self.n_files += 1
                if r.status_code == 200:
                    return r.content
                if r.status_code == 404:
                    return None
                if r.status_code == 429:
                    self._next_file = time.monotonic() + LIMITED_WAIT
            except requests.RequestException:
                pass
            time.sleep(2 ** attempt)
        return None

    def summary(self) -> str:
        """``N answers, M refusals`` plus the peak slow-down when there was one."""
        out = f"{self.n_served} answers, {self.n_limited} refusals"
        if self.slowdown_peak > 1:
            out += f", paced down to 1/{self.slowdown_peak:g} at worst"
        return out

    def pmap(self, fn: Callable, items: Iterable) -> List:
        items = list(items)
        if not items:
            return []
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(fn, items))

    # -- helpers ------------------------------------------------------------
    def list_page(self, classify: int, size: int, start: int, sort: int = 0, n: int = 30) -> List[Dict]:
        body = {"Classify": classify, "FileSize": size, "FileType": 5, "FileSort": sort,
                "Version": 19, "RefreshIndex": 0, "StartNum": start, "EndNum": start + n - 1}
        r = self.post("GetCategoryFileListV2", body)
        if r.get("ReturnCode") not in (0, 1):
            raise ApiDown(f"listing answered {r.get('ReturnCode')} {r.get('ReturnMessage')}")
        return r.get("FileList") or []

    def crawl_list(self, classify: int, size: int, stop_before: Optional[int] = None,
                   max_pages: Optional[int] = None, transform: Optional[Callable] = None,
                   old_streak: int = 2) -> List:
        """Page one (category, size) list in small waves.

        Lists are newest-first, but not strictly: some carry a block of about 120 much
        older artworks right after the first 60. So with ``stop_before`` the crawl ends
        only after ``old_streak`` consecutive pages hold nothing dated at or after it
        (5 pages clear such a block). Without it, it ends at the first empty page.
        ``self.last_gap`` tells whether an old page was followed by a newer one.

        Waves no longer double: the client is paced well under one request a second,
        so reading ahead buys no time and every page past the end is a wasted request.
        """
        out: List = []
        page, wave, streak, seen_old = 0, max(2, old_streak), 0, False
        self.last_gap = False
        while True:
            if max_pages is not None:
                wave = min(wave, max_pages - page)
                if wave <= 0:
                    break
            pages = self.pmap(lambda k: self.list_page(classify, size, 1 + 30 * k),
                              range(page, page + wave))
            done = False
            for fl in pages:
                if done:
                    break
                if not fl:
                    done = True
                    continue
                out.extend(fl if transform is None else [transform(x) for x in fl])
                if stop_before is not None:
                    if max(x["Date"] for x in fl) < stop_before:
                        streak += 1
                        seen_old = True
                        done = streak >= old_streak
                    else:
                        self.last_gap = self.last_gap or seen_old
                        streak = 0
            if done:
                break
            page += wave
        return out

    def like_list(self, gid: int, n: int, cap_pages: int = 5) -> Optional[List[Dict]]:
        """The newest ``n`` likers of an artwork (newest first), or None if refused."""
        out: List[Dict] = []
        start = 1
        for _ in range(cap_pages):
            r = self.post("Cloud/GetLikeUserList", {"GalleryId": gid, "StartNum": start, "EndNum": start + 99})
            if r.get("ReturnCode") == 11:
                return None
            ul = r.get("UserList") or []
            out.extend(ul)
            start += len(ul)
            if len(ul) < 100 or len(out) >= n:
                break
        return out[:n]

    def comments(self, gid: int) -> List[Dict]:
        r = self.post("Comment/GetCommentListV3", {"GalleryId": gid, "MessageId": 0, "Language": "en",
                                                   "StartNum": 1, "EndNum": 100}, auth=False)
        out: List[Dict] = []

        def walk(cl):
            for c in cl or []:
                out.append(c)
                walk(c.get("CommentChildList"))
        walk(r.get("CommentList"))
        return out

    def gallery_info(self, gid: int) -> Dict:
        return self.post("Cloud/GalleryInfo", {"GalleryId": gid})

    def user_info(self, uid: int) -> Dict:
        return self.post("GetSomeoneInfoV2", {"SomeOneUserId": uid, "Language": "en"}, auth=False)
