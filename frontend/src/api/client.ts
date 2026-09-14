/**
 * The single axios instance every request goes through, plus the one fetch()
 * call that streaming needs.
 *
 * Session model (mirrors backend/app/routers/auth.py):
 *
 *   - The access token lives in a module variable -- memory only. It used to
 *     sit in localStorage, which any script running on the page can read;
 *     an XSS bug there would have handed out sessions. In memory, a stolen
 *     token is worth the minutes until it expires, and a page reload simply
 *     asks for a new one.
 *   - The refresh token is an httpOnly cookie the browser sends to /api/auth
 *     by itself. JavaScript never sees it.
 *
 * Two interceptors do the boring work once:
 *
 *   request  -> attach the Bearer token if we have one
 *   response -> on 401, try ONE silent refresh and retry the request; if the
 *               refresh fails too, clear state and go to /login
 */

import axios from 'axios'
import type {
  ChatMessage,
  ChatResponse,
  ClaimResponse,
  CompareResponse,
  JobStatus,
  MemoryInfo,
  Paper,
  PaperReport,
  RelatedOut,
  Section,
  StreamEvent,
  Token,
  Usage,
  User,
} from './types'

let accessToken: string | null = null

export function getStoredToken(): string | null {
  return accessToken
}

export function storeToken(token: string): void {
  accessToken = token
}

export function clearToken(): void {
  accessToken = null
}

// One refresh in flight at a time. Several requests failing with 401 at once
// (a page load fires four) must share a single refresh, not race four.
let refreshing: Promise<string | null> | null = null

/**
 * Exchange the refresh cookie for a new access token. Resolves to null when
 * there is no valid session -- the normal state on a first visit.
 */
export function refreshAccessToken(): Promise<string | null> {
  if (!refreshing) {
    refreshing = axios
      .post<Token>(`${API_BASE_URL}/auth/refresh`, null, { withCredentials: true })
      .then((r) => {
        storeToken(r.data.access_token)
        return r.data.access_token
      })
      .catch(() => {
        clearToken()
        return null
      })
      .finally(() => {
        refreshing = null
      })
  }
  return refreshing
}

// Where the API lives.
//
// Default '/api' is a *relative* path, which works whenever the app and the API
// share an origin: Vite proxies it in development (see vite.config.ts), and in
// the deployed image FastAPI serves this bundle itself, so there is genuinely
// only one origin. Same-origin means CORS never applies and nothing needs
// configuring -- which is why this default is never overridden in practice.
//
// `||`, deliberately not `??`. Vite inlines this variable as an empty string
// when it is defined-but-blank -- which is exactly how the Dockerfile passes
// "no override" -- and an empty string is not nullish, so `??` would keep it
// and every request would drop the /api prefix.
export const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'

export const api = axios.create({ baseURL: API_BASE_URL })

api.interceptors.request.use((config) => {
  const token = getStoredToken()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const original = error.config
    const isAuthRoute = typeof original?.url === 'string' && original.url.startsWith('/auth/')
    if (error.response?.status === 401 && original && !original._retried && !isAuthRoute) {
      // Expired access token, most likely. Refresh once and replay.
      original._retried = true
      const token = await refreshAccessToken()
      if (token) {
        original.headers.Authorization = `Bearer ${token}`
        return api(original)
      }
      clearToken()
      // A hard redirect rather than a router navigate: interceptors live outside
      // React, so they have no access to the router's navigate function.
      if (window.location.pathname !== '/login') {
        window.location.href = '/login'
      }
    }
    return Promise.reject(error)
  },
)

/**
 * Pull a readable message out of an axios error.
 *
 * FastAPI puts human-readable text in `detail`, but validation errors (422) make
 * it an array of objects instead of a string -- rendering that directly would
 * put "[object Object]" in front of the user.
 */
export function errorMessage(error: unknown, fallback = 'Something went wrong.'): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail) && detail[0]?.msg) return String(detail[0].msg)
    if (!error.response) return 'Cannot reach the server. Is the backend running?'
  }
  return fallback
}

export function isNotFound(error: unknown): boolean {
  return axios.isAxiosError(error) && error.response?.status === 404
}

// ---------- Endpoint wrappers ----------
// Thin named functions rather than raw api.get(...) calls scattered through the
// components: the URL for each endpoint then exists in exactly one place.

export const authApi = {
  register: (email: string, password: string, fullName?: string) =>
    api
      .post<Token>('/auth/register', {
        email,
        password,
        full_name: fullName || null,
      })
      .then((r) => r.data),

  login: (email: string, password: string) =>
    api.post<Token>('/auth/login', { email, password }).then((r) => r.data),

  me: () => api.get<User>('/auth/me').then((r) => r.data),

  /** Clears the refresh cookie; `everywhere` also revokes other devices. */
  logout: (everywhere = false) =>
    api.post('/auth/logout', null, { params: everywhere ? { everywhere } : {} }).then(() => undefined),
}

export const papersApi = {
  list: () => api.get<Paper[]>('/papers').then((r) => r.data),

  get: (id: string) => api.get<Paper>(`/papers/${id}`).then((r) => r.data),

  sections: (id: string) =>
    api.get<Section[]>(`/papers/${id}/sections`).then((r) => r.data),

  job: (id: string) => api.get<JobStatus>(`/papers/${id}/job`).then((r) => r.data),

  /** Ingest from an arXiv or DOI link; the server fetches the PDF itself. */
  fromUrl: (url: string) => api.post<Paper>('/papers/from-url', { url }).then((r) => r.data),

  /** The original PDF, for the in-browser viewer. Needs the bearer header. */
  fileUrl: (id: string) => `${API_BASE_URL}/papers/${id}/file`,

  usage: (id: string) => api.get<Usage>(`/papers/${id}/usage`).then((r) => r.data),

  claim: (id: string, claim: string) =>
    api.post<ClaimResponse>(`/papers/${id}/claim`, { claim }).then((r) => r.data),

  compare: (paperIds: string[], question: string) =>
    api
      .post<CompareResponse>('/papers/compare', { paper_ids: paperIds, question })
      .then((r) => r.data),

  related: (id: string, refresh = false) =>
    api
      .get<RelatedOut>(`/papers/${id}/related`, { params: refresh ? { refresh } : {} })
      .then((r) => r.data),

  upload: (file: File) => {
    // multipart/form-data -- the browser sets the boundary header itself, so we
    // deliberately do not set Content-Type here.
    const form = new FormData()
    form.append('file', file)
    return api.post<Paper>('/papers', form).then((r) => r.data)
  },

  remove: (id: string) => api.delete(`/papers/${id}`).then(() => undefined),

  chat: (id: string, question: string, sectionId?: string | null) =>
    api
      .post<ChatResponse>(`/papers/${id}/chat`, { question, section_id: sectionId ?? null })
      .then((r) => r.data),

  messages: (id: string) =>
    api.get<ChatMessage[]>(`/papers/${id}/messages`).then((r) => r.data),

  clearMessages: (id: string) => api.delete(`/papers/${id}/messages`).then(() => undefined),

  followups: (id: string) =>
    api
      .get<{ questions: string[] }>(`/papers/${id}/followups`)
      .then((r) => r.data.questions),

  report: (id: string) =>
    api.get<PaperReport>(`/papers/${id}/report`).then((r) => r.data),

  startAnalysis: (id: string, force = false) =>
    api
      .post<{ status: string; job_id: string }>(`/papers/${id}/report`, null, {
        params: { force },
      })
      .then((r) => r.data),

  exportUrl: (id: string, format: 'md' | 'pdf') =>
    `${API_BASE_URL}/papers/${id}/export.${format}`,
}

/**
 * Download an export.
 *
 * A plain <a href> would be simpler but cannot carry the Authorization header,
 * and these endpoints are authenticated -- the link would 401. So: fetch the
 * bytes with the header, wrap them in an object URL, and click a synthetic
 * link. The object URL is revoked afterwards, otherwise the blob stays in
 * memory until the tab closes.
 */
export async function downloadExport(
  paperId: string,
  format: 'md' | 'pdf',
  filename: string,
): Promise<void> {
  const response = await api.get(papersApi.exportUrl(paperId, format).replace(API_BASE_URL, ''), {
    responseType: 'blob',
  })

  const url = URL.createObjectURL(response.data as Blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}

/**
 * Stream a chat answer over Server-Sent Events.
 *
 * fetch() rather than the browser's EventSource: EventSource can only issue GET
 * requests and cannot set headers, so it can neither send the question in a body
 * nor carry the bearer token. Reading the response body as a stream and parsing
 * the `data:` frames ourselves is a dozen lines and works with both.
 *
 * The `signal` lets a component abort the stream on unmount. Without it, a user
 * who navigates away mid-answer leaves a reader running that then calls
 * setState on an unmounted component.
 */
export async function streamChat(
  paperId: string,
  question: string,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
  sectionId?: string | null,
): Promise<void> {
  const send = (token: string | null) =>
    fetch(`${API_BASE_URL}/papers/${paperId}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({ question, section_id: sectionId ?? null }),
      signal,
    })

  let response = await send(getStoredToken())
  if (response.status === 401) {
    // fetch() bypasses the axios interceptor, so the one-shot refresh is
    // repeated here by hand.
    const token = await refreshAccessToken()
    if (token) response = await send(token)
  }

  if (!response.ok) {
    // The failure happened before streaming began, so there is a normal JSON
    // body to read. Once the stream has started the status is already 200 and
    // errors arrive as `error` events instead.
    let detail = `Request failed (${response.status}).`
    try {
      const body = await response.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      // Non-JSON error body; the generic message above is the best we have.
    }
    onEvent({ type: 'error', detail, retryable: response.status === 429 })
    return
  }

  if (!response.body) {
    onEvent({ type: 'error', detail: 'Streaming is not supported here.', retryable: false })
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    // `stream: true` matters: a multi-byte UTF-8 character can be split across
    // two network chunks, and decoding each chunk independently would corrupt it.
    buffer += decoder.decode(value, { stream: true })

    const parsed = parseSseBuffer(buffer)
    buffer = parsed.rest
    parsed.events.forEach(onEvent)
  }
}

/**
 * Split an SSE text buffer into complete events and the trailing partial frame.
 *
 * Frames are separated by a blank line; anything after the last blank line is
 * a partial frame and must be kept for the next read. Exported so it can be
 * unit-tested without a network -- the chunk boundaries are exactly where the
 * bugs live.
 */
export function parseSseBuffer(buffer: string): { events: StreamEvent[]; rest: string } {
  const frames = buffer.split('\n\n')
  const rest = frames.pop() ?? ''
  const events: StreamEvent[] = []
  for (const frame of frames) {
    const line = frame.split('\n').find((l) => l.startsWith('data: '))
    if (!line) continue
    try {
      events.push(JSON.parse(line.slice(6)) as StreamEvent)
    } catch {
      // A malformed frame is not worth tearing the stream down over.
    }
  }
  return { events, rest }
}

export const opsApi = {
  health: () => api.get<{ status: string }>('/health').then((r) => r.data),
  memory: () => api.get<MemoryInfo>('/debug/memory').then((r) => r.data),
}
