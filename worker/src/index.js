// Starts a GitHub Actions workflow of servoom-stats when a Cloudflare cron trigger fires.
// The hourly trigger starts the pulse, the daily one the snapshot. Nothing else happens
// here: the workflows themselves decide whether there is work to do.

const WORKFLOW = {
  '17 * * * *': 'pulse.yml',
  '43 5 * * *': 'snapshot.yml',
};

async function dispatch(env, workflow) {
  const res = await fetch(`https://api.github.com/repos/${env.REPO}/actions/workflows/${workflow}/dispatches`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${env.GITHUB_TOKEN}`,
      Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
      'User-Agent': 'servoom-stats-trigger',
    },
    body: JSON.stringify({ ref: 'main' }),
  });
  if (res.status !== 204) {
    // Shows up in the Worker's logs; the body never contains the token.
    throw new Error(`GitHub answered ${res.status} for ${workflow}: ${(await res.text()).slice(0, 200)}`);
  }
}

export default {
  async scheduled(event, env) {
    const workflow = WORKFLOW[event.cron];
    if (workflow) await dispatch(env, workflow);
  },
  // No public endpoint: the Worker has no route, and a stray request gets nothing.
  async fetch() {
    return new Response('Not found', { status: 404 });
  },
};
