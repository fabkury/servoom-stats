# servoom-stats

The data pipeline behind the statistics pages of [servoom.pages.dev](https://servoom.pages.dev/stats/):
an independent, read-only measurement of the public Divoom gallery. Unofficial and not
affiliated with Divoom.

This repository holds the collector, the aggregation, the workflows that run them, and
their output in [`data/`](data/). The pages that draw the data live in the
[servoom](https://github.com/fabkury/servoom) repository (`docs/public/stats/`), whose
build downloads `data/` from here. The design is documented in
[servoom/docs/community-stats](https://github.com/fabkury/servoom/tree/main/docs/community-stats).

## What runs

| Job | When | What it does |
|-----|------|--------------|
| `pulse` | hourly | Reads every upload of the past 30 days, new artwork ids, the Popular ordering, and who gave each new like. About 700 requests; once a day about 4,500, when every list is read with a long look-ahead. |
| refresh | every 4 hours, inside the pulse | Adds new-account sampling, new comments and category totals, then rebuilds `data/pulse/`. |
| `snapshot` | daily | Reads the whole public catalog (about 1.5 million artworks), new files, vanished artworks and featured-artist profiles, then rebuilds `data/daily/`. About 66,000 requests, roughly 3.5 hours. The crawl is checkpointed to the raw repository every 20 minutes, and a run started within 6 hours of a failed one resumes from the checkpoint. |

Each job commits `data/` when it changed and asks Cloudflare to rebuild the site.

## What starts the jobs

GitHub's own schedule for Actions proved unreliable (one scheduled run in the first eight
hours), so a small Cloudflare Worker in [`worker/`](worker/) starts the workflows through
the `workflow_dispatch` API: the pulse at :17 every hour and the snapshot at 05:43 UTC.
The workflows keep cron lines at other minutes as a fallback. A pulse exits at once when
the previous one is under 40 minutes old, and a snapshot when the previous one is under
12 hours old, so extra triggers cost a few seconds.

The Worker holds one secret, `GITHUB_TOKEN`: a fine-grained personal access token limited
to this repository with the permission "Actions: read and write". Deploy with
`npx wrangler deploy` from `worker/`; set the secret with `npx wrangler secret put GITHUB_TOKEN`.

The collector only sends read commands (`stats/api.py` keeps the list) and never likes,
views or uploads anything. A failed request is retried with backoff for up to 15 minutes
(`API_OUTAGE_SECONDS`) before a job gives up, so a short server outage does not end a run.

## Layout

```
stats/api.py        throttled access to the Divoom API
stats/accounts.py   polling account pool: token reuse, health checks, rotation, capped registration
stats/rawrepo.py    git plumbing for the private raw-data repository
stats/pulse.py      hourly job
stats/refresh.py    4-hour polling and data/pulse/*.json
stats/snapshot.py   daily job
stats/daily.py      data/daily/*.json
stats/files.py      artwork file features and hashes, avatars
config/             automated account ranges, artists who asked not to be listed
data/               published aggregates (JSON) and featured artists' avatars (WebP)
```

## What is public and what is not

`data/` holds aggregates only. Individual accounts are named only when Divoom itself
features them (Recommend picks, the expert list, an ambassador badge), and any figure
counted in accounts needs at least five of them. No artworks are stored here; the only
images are featured artists' avatars.

Raw observations (per-artwork counters, like events, account ids) and the polling
accounts live in a separate private repository. This repository reaches it with a
deploy key held as the Actions secret `RAW_REPO_DEPLOY_KEY`.

## Asking to be removed

If you are listed among the featured artists and would rather not be, open an issue.
Your account id is added to `config/excluded_artists.json` and your row, page and
avatar disappear at the next daily run.

## Running locally

```
pip install -r requirements.txt -r requirements-snapshot.txt
set RAW_REPO_URL=https://github.com/<you>/<your-raw-repo>.git
python -m stats.pulse        # PULSE_DAYS=1 for a quick run
python -m stats.snapshot     # LIMIT_LISTS=16_1,5_4 to read only a few lists
```

`FORCE_REFRESH=1` makes a pulse rebuild `data/pulse/`; `STRICT=1` turns skipped parts
into errors.
