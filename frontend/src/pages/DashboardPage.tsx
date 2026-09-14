import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router'

import { errorMessage, papersApi } from '../api/client'
import type { Paper } from '../api/types'
import { Spinner } from '../components/Spinner'
import { StatusBadge } from '../components/StatusBadge'
import { UploadCard } from '../components/UploadCard'

const POLL_INTERVAL_MS = 2000

export function DashboardPage() {
  const [papers, setPapers] = useState<Paper[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // useCallback keeps this function identity stable across renders, so the
  // effect below does not tear down and restart its timer on every render.
  const load = useCallback(async () => {
    try {
      setPapers(await papersApi.list())
      setError(null)
    } catch (err) {
      setError(errorMessage(err, 'Could not load your papers.'))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  // Poll only while something is actually being ingested.
  //
  // Polling is the simple, defensible choice here: ingestion takes 10-30s and
  // the alternative (WebSockets or server-sent events) means holding a
  // connection open and a lot more moving parts for one status field. The
  // important detail is that the interval is torn down the moment nothing is
  // processing, so an idle dashboard makes no requests at all.
  const isBusy = papers.some((p) => p.status === 'pending' || p.status === 'processing')

  useEffect(() => {
    if (!isBusy) return
    const id = setInterval(() => void load(), POLL_INTERVAL_MS)
    // The cleanup function is what stops the timer -- when isBusy flips to false,
    // or when this component unmounts. Forgetting it is the classic React memory
    // leak: a timer that keeps firing against a component that no longer exists.
    return () => clearInterval(id)
  }, [isBusy, load])

  async function handleDelete(paper: Paper) {
    if (!confirm(`Delete "${paper.title}"? This cannot be undone.`)) return
    try {
      await papersApi.remove(paper.id)
      // Drop it locally rather than refetching -- the UI updates instantly and
      // the server has already confirmed the delete.
      setPapers((current) => current.filter((p) => p.id !== paper.id))
    } catch (err) {
      setError(errorMessage(err, 'Could not delete that paper.'))
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h1 className="text-2xl font-semibold text-slate-900">Your papers</h1>
        {papers.filter((p) => p.status === 'ready').length >= 2 && (
          <Link
            to="/compare"
            className="rounded-md border border-slate-200 px-3 py-1.5 text-sm text-slate-700 transition hover:border-slate-900"
          >
            Compare papers
          </Link>
        )}
      </div>
      <p className="mt-1 text-sm text-slate-500">
        Upload a paper to get a section-by-section analysis, then ask questions
        and get answers that cite the passages they came from.
      </p>

      <div className="mt-6">
        <UploadCard onUploaded={load} />
      </div>

      {error && (
        <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>
      )}

      <div className="mt-8">
        {loading ? (
          <div className="flex justify-center py-12 text-slate-300">
            <Spinner className="h-8 w-8" />
          </div>
        ) : papers.length === 0 ? (
          <p className="rounded-lg border border-slate-200 bg-white py-12 text-center text-sm text-slate-500">
            No papers yet. Upload your first one above.
          </p>
        ) : (
          <ul className="space-y-3">
            {papers.map((paper) => (
              // `key` lets React match elements to data across re-renders. Using
              // the array index instead would reorder state incorrectly when the
              // list changes -- the id is stable, so it is the right key.
              <li
                key={paper.id}
                className="flex items-center justify-between gap-4 rounded-lg border border-slate-200 bg-white p-4"
              >
                <div className="min-w-0">
                  <div className="flex items-center gap-3">
                    <h2 className="truncate font-medium text-slate-900">{paper.title}</h2>
                    <StatusBadge status={paper.status} />
                    {paper.ocr && (
                      <span
                        className="rounded bg-amber-50 px-1.5 py-0.5 text-[10px] font-medium text-amber-800"
                        title="This PDF had no text layer; the text was transcribed by the model and may contain errors."
                      >
                        OCR
                      </span>
                    )}
                  </div>

                  <p className="mt-1 truncate text-xs text-slate-500">
                    {paper.filename}
                    {paper.status === 'ready' &&
                      ` · ${paper.num_pages} pages · ${paper.num_chunks} chunks`}
                  </p>

                  {paper.status === 'failed' && paper.error_message && (
                    <p className="mt-1 text-xs text-red-600">{paper.error_message}</p>
                  )}
                </div>

                <div className="flex shrink-0 items-center gap-2">
                  {/* Open is available while processing too, not only when
                      ready: the paper page shows live job progress, which is
                      more useful to watch than this row's status badge. */}
                  {paper.status !== 'failed' && (
                    <Link
                      to={`/papers/${paper.id}`}
                      className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white transition hover:bg-slate-800"
                    >
                      {paper.status === 'ready' ? 'Open' : 'View progress'}
                    </Link>
                  )}
                  <button
                    onClick={() => void handleDelete(paper)}
                    className="rounded-md px-3 py-1.5 text-sm text-slate-500 transition hover:bg-red-50 hover:text-red-700"
                  >
                    Delete
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
