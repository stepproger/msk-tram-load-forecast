import http from 'k6/http';
import { check } from 'k6';
import encoding from 'k6/encoding';

// One iteration makes six concurrent requests, so 50 iterations/s produces
// 300 HTTP RPS. This mirrors the UI's Promise.all refresh and includes a 10m soak.
const base = __ENV.BASE_URL || 'http://localhost:3000';
const targetIterationsPerSecond = Number(__ENV.TARGET_ITERATIONS_PER_SECOND || 50);
const soakMinutes = Number(__ENV.SOAK_MINUTES || 10);
// The API is behind HTTP Basic; every request authenticates like the dispatcher UI does.
const credentials = encoding.b64encode(`${__ENV.APP_USER || 'dispatcher'}:${__ENV.APP_PASSWORD || 'tram2025'}`);
const params = { headers: { Accept: 'application/json', Authorization: `Basic ${credentials}` }, timeout: '10s' };

export const options = {
  scenarios: {
    dispatcher_mix: {
      executor: 'ramping-arrival-rate',
      startRate: Math.max(1, Math.round(targetIterationsPerSecond * 0.2)),
      timeUnit: '1s',
      preAllocatedVUs: 50,
      maxVUs: 200,
      stages: [
        { target: Math.max(1, Math.round(targetIterationsPerSecond * 0.4)), duration: '30s' },
        { target: Math.max(1, Math.round(targetIterationsPerSecond * 0.7)), duration: '30s' },
        { target: targetIterationsPerSecond, duration: '30s' },
        { target: targetIterationsPerSecond, duration: `${soakMinutes}m` },
        { target: 0, duration: '30s' },
      ],
      gracefulStop: '30s',
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.01'],
    checks: ['rate>0.99'],
    dropped_iterations: ['count==0'],
    'http_req_duration{endpoint:forecast_day}': ['p(95)<300'],
    'http_req_duration{endpoint:forecast_month}': ['p(95)<300'],
    'http_req_duration{endpoint:map_snapshot}': ['p(95)<300'],
    'http_req_duration{endpoint:xlsx_export}': ['p(95)<300'],
    'http_req_duration{endpoint:components}': ['p(95)<300'],
    'http_req_duration{endpoint:risk}': ['p(95)<300'],
  },
};

// RANDOMIZE=1 draws route, date and hour per iteration, so the data-service response cache
// sees mostly new URLs; without it the profile repeats one dispatcher screen.
const randomize = __ENV.RANDOMIZE === '1';
const ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50];
const pick = (items) => items[Math.floor(Math.random() * items.length)];
const pad = (value) => String(value).padStart(2, '0');

export default function () {
  const route = randomize ? pick(ROUTES) : 25;
  const month = randomize ? pick([11, 12]) : 11;
  const day = randomize ? pad(1 + Math.floor(Math.random() * (month === 11 ? 30 : 31))) : '03';
  const date = `2025-${month}-${day}`;
  const hour = randomize ? Math.floor(Math.random() * 24) : 8;
  const requests = [
    ['forecast_day', `/forecast?route=${route}&from=${date}&to=${date}&horizon=day&granularity=hour`],
    ['forecast_month', `/forecast?from=2025-${month}-01&to=2025-${month}-${month === 11 ? 30 : 31}&horizon=month&granularity=day`],
    ['map_snapshot', `/map/snapshot?datetime=${date}T${pad(hour)}%3A00%3A00%2B03%3A00`],
    ['xlsx_export', `/export?route=${route}&from=2025-${month}-01&to=2025-${month}-${month === 11 ? 30 : 31}&horizon=month&granularity=day&format=xlsx`],
    ['components', `/components?route=${route}&date=${date}&hour=${hour}`],
    ['risk', `/risk?date=${date}&hour=${hour}`],
  ].map(([endpoint, path]) => ({
    method: 'GET',
    url: `${base}/api/v1${path}`,
    params: { ...params, tags: { endpoint } },
  }));
  const responses = http.batch(requests);
  responses.forEach((response, index) => {
    const endpoint = requests[index].params.tags.endpoint;
    check(response, { [`${endpoint}: HTTP 200`]: (r) => r.status === 200 });
  });
}
