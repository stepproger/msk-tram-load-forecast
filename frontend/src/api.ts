export const API = import.meta.env.VITE_API_BASE_URL || '/api/v1';

const AUTH_KEY = 'tram-auth';
const listeners = new Set<() => void>();
let tabAuth: string | null = null;

// Credentials live only for the browser tab: sessionStorage is cleared when the tab closes.
export function getAuth(): string | null {
  try { return window.sessionStorage.getItem(AUTH_KEY) ?? tabAuth; } catch { return tabAuth; }
}
export function authUser(): string {
  const token = getAuth();
  if (!token) return '';
  try { return new TextDecoder().decode(Uint8Array.from(atob(token), (c) => c.charCodeAt(0))).split(':')[0]; } catch { return ''; }
}
function encodeBasic(user: string, password: string) {
  return btoa(String.fromCharCode(...new TextEncoder().encode(`${user}:${password}`)));
}
export function logout() {
  tabAuth = null;
  try { window.sessionStorage.removeItem(AUTH_KEY); } catch { /* Storage disabled: nothing to clear. */ }
  listeners.forEach((listener) => listener());
}
export function onAuthChange(listener: () => void) {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

export class ApiError extends Error {
  constructor(message: string, readonly status: number) { super(message); }
}

async function errorMessage(res: Response) {
  const body = await res.json().catch(() => null);
  return body?.message || body?.detail || `Ошибка запроса (${res.status})`;
}

export async function apiFetch(url: string, init: RequestInit = {}, token = getAuth()): Promise<Response> {
  const headers = new Headers(init.headers);
  if (token) headers.set('Authorization', `Basic ${token}`);
  const res = await fetch(`${API}${url}`, { ...init, headers });
  if (res.status === 401) {
    if (token === getAuth()) logout();
    throw new ApiError(await errorMessage(res), 401);
  }
  if (!res.ok) throw new ApiError(await errorMessage(res), res.status);
  return res;
}

export async function get<T>(url: string): Promise<T> {
  return (await apiFetch(url)).json();
}

/** Checks the credentials against a protected endpoint and stores them only on success. */
export async function login(user: string, password: string) {
  const token = encodeBasic(user, password);
  await apiFetch('/routes', {}, token);
  tabAuth = token;
  try { window.sessionStorage.setItem(AUTH_KEY, token); } catch { /* Keep working without persistence. */ }
  listeners.forEach((listener) => listener());
}

/** Downloads an authenticated file; a plain link cannot carry the Authorization header. */
export async function download(url: string, fallbackName: string) {
  const res = await apiFetch(url);
  const disposition = res.headers.get('Content-Disposition') ?? '';
  const name = /filename="?([^";]+)"?/.exec(disposition)?.[1] ?? fallbackName;
  const href = URL.createObjectURL(await res.blob());
  const link = Object.assign(document.createElement('a'), { href, download: name });
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(href);
}

/**
 * Server-sent events over fetch: EventSource cannot send the Authorization header.
 * Returns a function that aborts the stream.
 */
export function streamEvents(
  url: string,
  onEvent: (event: string, data: string) => void,
  onClose: (error?: Error) => void,
): () => void {
  const controller = new AbortController();
  (async () => {
    const res = await apiFetch(url, { signal: controller.signal, headers: { Accept: 'text/event-stream' } });
    const reader = res.body!.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;
      let boundary;
      while ((boundary = buffer.search(/\r?\n\r?\n/)) >= 0) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary).replace(/^\r?\n\r?\n/, '');
        let event = 'message';
        const data: string[] = [];
        block.split(/\r?\n/).forEach((line) => {
          if (line.startsWith('event:')) event = line.slice(6).trim();
          else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
        });
        onEvent(event, data.join('\n'));
      }
    }
    onClose();
  })().catch((error: Error) => { if (!controller.signal.aborted) onClose(error); });
  return () => controller.abort();
}
