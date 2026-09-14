import { Suspense, lazy, useCallback, useEffect, useRef, useState, type FormEvent } from 'react'
import { Link, useParams } from 'react-router'

import { errorMessage, isNotFound, papersApi, streamChat } from '../api/client'
import type {
  ChatMessage,
  Citation,
  ClaimResponse,
  JobStatus,
  Paper,
  PaperReport,
  Section,
  StreamEvent,
} from '../api/types'
import { ClaimCard } from '../components/ClaimCard'
import { JobProgress } from '../components/JobProgress'
import { RelatedWorkPanel } from '../components/RelatedWorkPanel'
import { MessageBubble } from '../components/MessageBubble'
import type { Highlight } from '../components/PdfViewer'
import { ReportPanel } from '../components/ReportPanel'
import { Spinner } from '../components/Spinner'

/**
 * The reader view: analysis on the left, chat on the right.
 *
 * Two independent async lifecycles live here and it is worth being explicit
 * about both:
 *
 *  - **Job polling.** While the paper is ingesting or analysing, we poll
 *    /job every couple of seconds and render real progress. Polling stops the
 *    moment the job reports done or failed, so an idle open tab is not making
 *    requests forever against a free instance.
 *  - **Answer streaming.** A question opens an SSE stream and tokens append to
 *    a placeholder message. On `done` the placeholder is replaced wholesale,
 *    because the backend renumbers citation markers against the set the model
 *    actually used and that renumbering is only knowable at the end.
 */

const POLL_INTERVAL_MS = 2000

// PDF.js is ~400 kB minified. Loaded only when someone opens the viewer, so
// the dashboard and the chat never pay for it.
const PdfViewer = lazy(() =>
  import('../components/PdfViewer').then((m) => ({ default: m.PdfViewer })),
)

export function PaperPage() {
  // The :paperId segment declared in the route. Typed as possibly undefined
  // because the router cannot prove the URL matched.
  const { paperId } = useParams<{ paperId: string }>()

  const [paper, setPaper] = useState<Paper | null>(null)
  const [sections, setSections] = useState<Section[]>([])
  const [report, setReport] = useState<PaperReport | null>(null)
  const [job, setJob] = useState<JobStatus | null>(null)

  // The transcript interleaves chat messages and claim verdicts. Claim
  // results are not persisted (they are lookups, not conversation), so they
  // live only in this component's state for the session.
  type Entry = { kind: 'message'; message: ChatMessage } | { kind: 'claim'; result: ClaimResponse }
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [claims, setClaims] = useState<{ afterMessageId: string | null; result: ClaimResponse }[]>([])
  const [mode, setMode] = useState<'ask' | 'claim'>('ask')
  const [streamingId, setStreamingId] = useState<string | null>(null)
  const [followups, setFollowups] = useState<string[]>([])

  const [question, setQuestion] = useState('')
  // Retrieval scope: null means the whole paper.
  const [scope, setScope] = useState<string | null>(null)
  // The PDF viewer replaces the report in the reader column while open.
  const [viewerOpen, setViewerOpen] = useState(false)
  const [highlight, setHighlight] = useState<Highlight | null>(null)
  const [loading, setLoading] = useState(true)
  const [asking, setAsking] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const bottomRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Keyboard shortcuts: "/" focuses the question box from anywhere on the
  // page (unless already typing), Escape aborts an answer in progress.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes((event.target as HTMLElement)?.tagName)
      if (event.key === '/' && !typing) {
        event.preventDefault()
        inputRef.current?.focus()
      } else if (event.key === 'Escape' && abortRef.current) {
        abortRef.current.abort()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  async function clearConversation() {
    if (!paperId || !confirm('Clear this conversation? The paper and its analysis stay.')) return
    try {
      await papersApi.clearMessages(paperId)
      setMessages([])
      setClaims([])
    } catch (err) {
      setError(errorMessage(err, 'Could not clear the conversation.'))
    }
  }

  // --- Initial load ------------------------------------------------------
  useEffect(() => {
    if (!paperId) return
    setLoading(true)

    // In parallel: none of these depend on each other, so waiting for one
    // before starting the next is wasted time.
    Promise.all([
      papersApi.get(paperId),
      papersApi.messages(paperId),
      papersApi.sections(paperId).catch(() => [] as Section[]),
    ])
      .then(([loadedPaper, history, loadedSections]) => {
        setPaper(loadedPaper)
        setMessages(history)
        setSections(loadedSections)
      })
      .catch((err) => setError(errorMessage(err, 'Could not open that paper.')))
      .finally(() => setLoading(false))
  }, [paperId])

  // --- Report, once available -------------------------------------------
  const loadReport = useCallback(async () => {
    if (!paperId) return
    try {
      setReport(await papersApi.report(paperId))
    } catch (err) {
      // A 404 here is the normal state before the analysis finishes, not a
      // failure worth showing the user -- the job panel already explains it.
      if (!isNotFound(err)) {
        setError(errorMessage(err, 'Could not load the analysis.'))
      }
    }
  }, [paperId])

  // --- Job polling -------------------------------------------------------
  useEffect(() => {
    if (!paperId || !paper) return
    let cancelled = false

    async function poll(): Promise<void> {
      while (!cancelled) {
        let status: JobStatus
        try {
          status = await papersApi.job(paperId!)
        } catch {
          return // the paper was deleted, or we are offline; stop quietly
        }
        if (cancelled) return
        setJob(status)

        if (status.status === 'done' || status.status === 'failed') {
          // Refresh the paper and its sections either way: title, page count
          // and chunk count are filled in during ingestion, and a job that
          // failed at the *analysis* stage still leaves a fully indexed,
          // chattable paper behind. Showing the stale "0 passages" from
          // page load would make it look unusable.
          void papersApi.get(paperId!).then(setPaper).catch(() => undefined)
          void papersApi.sections(paperId!).then(setSections).catch(() => undefined)
          if (status.status === 'done') {
            void loadReport()
            void papersApi.followups(paperId!).then(setFollowups).catch(() => undefined)
          }
          return
        }

        await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS))
      }
    }

    void poll()
    return () => {
      cancelled = true
    }
    // `paper?.id` rather than the whole `paper` object, deliberately. The poll
    // itself calls setPaper on completion, so depending on the object would
    // change its identity, restart the effect, and poll forever. The id is what
    // actually determines which paper we are watching.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paperId, paper?.id, loadReport])

  // Keep the newest message in view. Runs after every render that changes the
  // list, which is exactly when the scroll position needs correcting.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, claims, asking])

  // Abort any in-flight stream when the component unmounts, so a user who
  // navigates away mid-answer does not leave a reader calling setState on an
  // unmounted component.
  useEffect(() => () => abortRef.current?.abort(), [])

  // --- Asking ------------------------------------------------------------
  async function ask(text: string) {
    if (!text || !paperId || asking) return

    setError(null)
    setQuestion('')
    setFollowups([])

    const pendingId = `pending-${Date.now()}`
    const answerId = `answer-${Date.now()}`

    // Show the question immediately instead of waiting for the server.
    setMessages((current) => [
      ...current,
      {
        id: pendingId,
        role: 'user',
        content: text,
        created_at: new Date().toISOString(),
        citations: [],
      },
      {
        id: answerId,
        role: 'assistant',
        content: '',
        created_at: new Date().toISOString(),
        citations: [],
      },
    ])
    setAsking(true)
    setStreamingId(answerId)

    const controller = new AbortController()
    abortRef.current = controller

    function patchAnswer(update: (message: ChatMessage) => ChatMessage) {
      setMessages((current) =>
        current.map((m) => (m.id === answerId ? update(m) : m)),
      )
    }

    let failed = false

    function handle(event: StreamEvent) {
      if (event.type === 'token') {
        patchAnswer((m) => ({ ...m, content: m.content + event.text }))
      } else if (event.type === 'done') {
        // Replace rather than append: markers were renumbered server-side
        // against the citations actually used, so the streamed text and the
        // final text can differ.
        patchAnswer((m) => ({
          ...m,
          content: event.answer,
          citations: event.citations as Citation[],
        }))
        setFollowups(event.followups)
      } else if (event.type === 'error') {
        failed = true
        setError(event.detail)
      }
    }

    try {
      await streamChat(paperId, text, handle, controller.signal, scope)
    } catch (err) {
      if (!controller.signal.aborted) {
        failed = true
        setError(errorMessage(err, 'Could not get an answer.'))
      }
    } finally {
      setAsking(false)
      setStreamingId(null)
      abortRef.current = null
    }

    if (failed) {
      // Roll the exchange back so a failed question does not sit there looking
      // as though it was asked and ignored, and give them their typing back.
      setMessages((current) =>
        current.filter((m) => m.id !== pendingId && m.id !== answerId),
      )
      setQuestion(text)
    }
  }

  function showCitation(citation: Citation) {
    // Jump to the first page of the chunk; highlight only a verified quote --
    // a 300-character snippet would match nothing on the page anyway, and an
    // unverified quote is exactly what we refuse to point at.
    setHighlight({
      page: citation.page_start,
      phrase: citation.verified ? citation.quote : null,
      nonce: Date.now(),
    })
    setViewerOpen(true)
  }

  async function checkClaim(text: string) {
    if (!text || !paperId || asking) return
    setError(null)
    setQuestion('')
    setAsking(true)
    try {
      const result = await papersApi.claim(paperId, text)
      const last = messages[messages.length - 1]?.id ?? null
      setClaims((current) => [...current, { afterMessageId: last, result }])
    } catch (err) {
      setError(errorMessage(err, 'Could not check that claim.'))
      setQuestion(text)
    } finally {
      setAsking(false)
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault()
    const text = question.trim()
    if (mode === 'claim') await checkClaim(text)
    else await ask(text)
  }

  // Merge messages and claim cards into one ordered transcript.
  const transcript: Entry[] = []
  claims
    .filter((c) => c.afterMessageId === null)
    .forEach((c) => transcript.push({ kind: 'claim', result: c.result }))
  messages.forEach((message) => {
    transcript.push({ kind: 'message', message })
    claims
      .filter((c) => c.afterMessageId === message.id)
      .forEach((c) => transcript.push({ kind: 'claim', result: c.result }))
  })

  async function handleRegenerate() {
    if (!paperId) return
    setReport(null)
    try {
      await papersApi.startAnalysis(paperId, true)
      setJob(await papersApi.job(paperId))
      // Nudge the polling effect by refreshing the paper object.
      setPaper((p) => (p ? { ...p } : p))
    } catch (err) {
      setError(errorMessage(err, 'Could not restart the analysis.'))
    }
  }

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center text-slate-300">
        <Spinner className="h-8 w-8" />
      </div>
    )
  }

  if (!paper) {
    return (
      <div className="mx-auto max-w-5xl px-4 py-16 text-center">
        <p className="text-slate-600">{error ?? 'Paper not found.'}</p>
        <Link to="/" className="mt-4 inline-block text-sm text-slate-900 hover:underline">
          Back to your papers
        </Link>
      </div>
    )
  }

  const meta = [paper.authors, paper.venue, paper.year].filter(Boolean).join(' · ')
  const busy = job !== null && job.status !== 'done' && job.status !== 'failed'

  return (
    <div className="mx-auto h-full max-w-7xl px-4 py-6">
      <div className="mb-4">
        <Link to="/" className="text-sm text-slate-500 hover:text-slate-900">
          &larr; All papers
        </Link>
        <div className="mt-1 flex flex-wrap items-baseline justify-between gap-2">
          <h1 className="text-xl font-semibold text-slate-900">{paper.title}</h1>
          <button
            onClick={() => {
              setHighlight(null)
              setViewerOpen((open) => !open)
            }}
            className="rounded-md border border-slate-200 px-2.5 py-1 text-xs text-slate-600 transition hover:border-slate-900 hover:text-slate-900"
          >
            {viewerOpen ? 'Show analysis' : 'View PDF'}
          </button>
        </div>
        {meta && <p className="text-sm text-slate-500">{meta}</p>}
        <p className="text-xs text-slate-400">
          {paper.num_pages} pages · {paper.num_chunks} indexed passages
          {sections.length > 0 && ` · ${sections.length} sections`}
          {paper.ocr && (
            <span
              className="ml-2 rounded bg-amber-50 px-1.5 py-0.5 text-[10px] font-medium text-amber-800"
              title="No text layer in the PDF; text was transcribed by the model and may contain errors."
            >
              OCR
            </span>
          )}
        </p>
      </div>

      <div className="grid h-[calc(100%-6rem)] grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        {/* --- Reader column --- */}
        <div className={`min-h-0 pr-1 ${viewerOpen ? 'flex flex-col' : 'space-y-4 overflow-y-auto'}`}>
          {viewerOpen && paperId ? (
            <Suspense
              fallback={
                <div className="flex h-full items-center justify-center text-slate-300">
                  <Spinner className="h-6 w-6" />
                </div>
              }
            >
              <PdfViewer
                paperId={paperId}
                highlight={highlight}
                onClose={() => setViewerOpen(false)}
              />
            </Suspense>
          ) : (
            <>
              {busy && job && <JobProgress job={job} />}
              {!busy && job?.status === 'failed' && <JobProgress job={job} />}

          {report ? (
            <ReportPanel paper={paper} report={report} onRegenerate={handleRegenerate} />
          ) : (
            !busy && (
              <div className="rounded-lg border border-slate-200 bg-white p-6 text-center">
                <p className="text-sm text-slate-600">
                  No analysis for this paper yet.
                </p>
                <button
                  onClick={handleRegenerate}
                  className="mt-3 rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white transition hover:bg-slate-800"
                >
                  Run the analysis
                </button>
              </div>
            )
          )}

          {paper.status === 'ready' && paperId && <RelatedWorkPanel paperId={paperId} />}

          {sections.length > 0 && (
            <div className="rounded-lg border border-slate-200 bg-white p-4">
              <h2 className="text-xs font-medium tracking-wide text-slate-400 uppercase">
                Detected structure
              </h2>
              <ul className="mt-2 space-y-1">
                {sections.map((section) => (
                  <li
                    key={section.id}
                    className="flex items-baseline justify-between gap-3 text-sm"
                  >
                    <span className="truncate text-slate-700">{section.name}</span>
                    <span className="shrink-0 text-xs tabular-nums text-slate-400">
                      {section.page_start === section.page_end
                        ? `p${section.page_start}`
                        : `pp${section.page_start}–${section.page_end}`}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
            </>
          )}
        </div>

        {/* --- Chat column --- */}
        <div className="flex min-h-0 flex-col">
          <div className="min-h-0 flex-1 space-y-4 overflow-y-auto pr-1">
            {messages.length === 0 && claims.length === 0 && (
              <div className="rounded-lg border border-slate-200 bg-white p-6 text-center">
                <p className="text-sm text-slate-600">
                  Ask anything about this paper. Every answer cites the passages it
                  came from.
                </p>
                <p className="mt-2 text-xs text-slate-400">
                  Press <kbd className="rounded border border-slate-300 px-1">/</kbd> to start typing,{' '}
                  <kbd className="rounded border border-slate-300 px-1">Esc</kbd> to stop an answer.
                </p>
              </div>
            )}
            {(messages.length > 0 || claims.length > 0) && !asking && (
              <div className="flex justify-end">
                <button
                  onClick={() => void clearConversation()}
                  className="text-xs text-slate-400 hover:text-slate-900"
                >
                  Clear conversation
                </button>
              </div>
            )}

            {transcript.map((entry, index) =>
              entry.kind === 'message' ? (
                <MessageBubble
                  key={entry.message.id}
                  message={entry.message}
                  streaming={streamingId === entry.message.id}
                  onCite={showCitation}
                />
              ) : (
                <ClaimCard key={`claim-${index}`} result={entry.result} onCite={showCitation} />
              ),
            )}

            {asking && streamingId === null && (
              <div className="flex items-center gap-2 text-sm text-slate-400">
                <Spinner />
                Reading the paper…
              </div>
            )}

            {/* Empty div as a scroll anchor -- scrolling to it scrolls to the bottom. */}
            <div ref={bottomRef} />
          </div>

          {followups.length > 0 && !asking && (
            <div className="mt-3 shrink-0">
              <p className="text-xs text-slate-400">Suggested questions</p>
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {followups.map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => void ask(suggestion)}
                    className="rounded-full border border-slate-200 bg-white px-3 py-1 text-left text-xs text-slate-600 transition hover:border-slate-900 hover:text-slate-900"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          {error && (
            <p className="mt-3 shrink-0 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">
              {error}
            </p>
          )}

          <form onSubmit={handleSubmit} className="mt-4 flex shrink-0 gap-2">
            {/* Ask a question, or state a claim and get a verdict on it. */}
            <div className="flex shrink-0 overflow-hidden rounded-md border border-slate-300 text-xs">
              {(['ask', 'claim'] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setMode(m)}
                  className={`px-2.5 py-2 transition ${
                    mode === m ? 'bg-slate-900 text-white' : 'bg-white text-slate-600 hover:bg-slate-50'
                  }`}
                  title={m === 'ask' ? 'Ask a question' : 'Check whether the paper supports a claim'}
                >
                  {m === 'ask' ? 'Ask' : 'Check claim'}
                </button>
              ))}
            </div>
            {mode === 'ask' && sections.length > 1 && (
              // Narrow retrieval to one section. The backend ignores an id
              // that does not belong to this paper, so this is a convenience,
              // not a security boundary.
              <select
                value={scope ?? ''}
                onChange={(e) => setScope(e.target.value || null)}
                disabled={paper.status !== 'ready'}
                title="Search only within one section"
                className="max-w-[10rem] rounded-md border border-slate-300 bg-white px-2 py-2 text-xs text-slate-700 outline-none focus:border-slate-900 disabled:bg-slate-50"
              >
                <option value="">Whole paper</option>
                {sections.map((section) => (
                  <option key={section.id} value={section.id}>
                    {section.name.length > 28 ? section.name.slice(0, 27) + '…' : section.name}
                  </option>
                ))}
              </select>
            )}
            <input
              ref={inputRef}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={
                paper.status !== 'ready'
                  ? 'Waiting for the paper to finish processing…'
                  : mode === 'claim'
                    ? 'State a claim, e.g. "The model beats the baseline on every task"'
                    : 'Ask a question about this paper…'
              }
              disabled={paper.status !== 'ready'}
              className="flex-1 rounded-md border border-slate-300 bg-white px-3 py-2 outline-none focus:border-slate-900 disabled:bg-slate-50"
            />
            <button
              type="submit"
              disabled={asking || question.trim().length < 3 || paper.status !== 'ready'}
              className="rounded-md bg-slate-900 px-4 py-2 font-medium text-white transition hover:bg-slate-800 disabled:opacity-40"
            >
              {mode === 'claim' ? 'Check' : 'Ask'}
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}
