"""Throttled, retrying access to the Divoom cloud API. Read-only commands only."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Iterable, List, Optional

import requests

HOST = "https://appin.divoom-gz.com"
FILE_HOST = "https://f.divoom-gz.com"

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


class Api:
    def __init__(self, rps: float = 4.0, workers: int = 4):
        self.s = requests.Session()
        self.interval = 1.0 / rps
        self.workers = workers
        self._lock = threading.Lock()
        self._next = 0.0
        self.auth: Dict = {}
        self.n_requests = 0
        self._fail_streak = 0

    def _wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            t = max(now, self._next)
            self._next = t + self.interval
        if t > now:
            time.sleep(t - now)

    def post(self, cmd: str, body: Optional[Dict] = None, auth: bool = True) -> Dict:
        if cmd not in READ_ONLY:
            raise ValueError(f"command not on the read-only list: {cmd}")
        payload = {**(self.auth if auth else {}), **(body or {})}
        last: Optional[Exception] = None
        for attempt in range(5):
            self._wait()
            try:
                r = self.s.post(f"{HOST}/{cmd}", json=payload, timeout=30)
                self.n_requests += 1
                if r.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                data = r.json()
                self._fail_streak = 0
                return data
            except (requests.RequestException, ValueError) as exc:
                last = exc
                self._fail_streak += 1
                if self._fail_streak > 40:
                    raise ApiDown(f"{cmd}: {exc}") from exc
                time.sleep(2 ** attempt)
        raise ApiDown(f"{cmd}: {last}")

    def get_file(self, file_id: str) -> Optional[bytes]:
        for attempt in range(3):
            self._wait()
            try:
                r = self.s.get(f"{FILE_HOST}/{file_id}", timeout=60)
                self.n_requests += 1
                if r.status_code == 200:
                    return r.content
                if r.status_code == 404:
                    return None
            except requests.RequestException:
                pass
            time.sleep(2 ** attempt)
        return None

    def pmap(self, fn: Callable, items: Iterable) -> List:
        items = list(items)
        if not items:
            return []
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(fn, items))

    # -- helpers ------------------------------------------------------------
    def list_page(self, classify: int, size: int, start: int, sort: int = 0, n: int = 30) -> List[Dict]:
        r = self.post("GetCategoryFileListV2", {
            "Classify": classify, "FileSize": size, "FileType": 5, "FileSort": sort,
            "Version": 19, "RefreshIndex": 0, "StartNum": start, "EndNum": start + n - 1})
        if r.get("ReturnCode") not in (0, 1):
            raise ApiDown(f"listing answered {r.get('ReturnCode')} {r.get('ReturnMessage')}")
        return r.get("FileList") or []

    def crawl_list(self, classify: int, size: int, stop_before: Optional[int] = None,
                   max_pages: Optional[int] = None, transform: Optional[Callable] = None,
                   old_streak: int = 2) -> List:
        """Page one (category, size) list in parallel waves.

        Lists are newest-first, but not strictly: some carry a block of about 120 much
        older artworks right after the first 60. So with ``stop_before`` the crawl ends
        only after ``old_streak`` consecutive pages hold nothing dated at or after it
        (5 pages clear such a block). Without it, it ends at the first empty page.
        ``self.last_gap`` tells whether an old page was followed by a newer one.
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
            wave = min(wave * 2, 32)
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
