import { useEffect, useState } from 'react'

import { papersApi } from '../api/client'
import type { RelatedEntry } from '../api/types'

/**
 * "Papers this builds on": the reference list, enriched from OpenAlex.
 *
 * The first request queues the lookups on the server's worker thread and
 * answers `pending`; this polls until the cache is filled. Entries OpenAlex
 * could not match are still listed, from the parsed reference, so the panel
 * is the whole bibliography with extra detail where it was available -- not
 * a curated subset.
 */

const POLL_MS = 3000

export function RelatedWorkPanel({ paperId }: { paperId: string }) {
  const [open, setOpen] = useState(false)
  const [entries, setEntries] = useState<RelatedEntry[] | null>(null)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!open || entries) return
    let cancelled = false

    async function poll() {
      while (!cancelled) {
        try {
          const result = await papersApi.related(paperId)
          if (cancelled) return
          if (result.status === 'ready') {
            setEntries(result.entries)
            setPending(false)
            return
          }
          setPending(true)
        } catch {
          setError('Could not load related work.')
          return
        }
        await new Promise((resolve) => setTimeout(resolve, POLL_MS))
      }
    }
    void poll()
    return () => {
      cancelled = true
    }
  }, [open, entries, paperId])

  const matched = entries?.filter((e) => e.openalex_id) ?? []
  const unmatched = entries?.filter((e) => !e.openalex_id) ?? []

  return (
    <div className="rounded-lg border border-slate-200 bg-white">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left"
      >
        <span className="flex items-baseline gap-2">
          <span className="text-sm font-medium text-slate-900">Papers this builds on</span>
          {entries && (
            <span className="text-xs tabular-nums text-slate-400">
              {matched.length} found · {entries.length} references
            </span>
          )}
        </span>
        <span className={`shrink-0 text-slate-400 transition-transform ${open ? 'rotate-90' : ''}`} aria-hidden>
          ›
        </span>
      </button>

      {open && (
        <div className="border-t border-slate-100 px-4 pb-4">
          <p className="mt-2 text-xs text-slate-400">
            Parsed from the reference list and looked up on OpenAlex. No model
            involved — this is metadata, not generation.
          </p>

          {error && <p className="mt-2 text-sm text-red-700">{error}</p>}
          {pending && !entries && (
            <p className="mt-3 text-sm text-slate-500">Looking up references…</p>
          )}

          {entries && entries.length === 0 && (
            <p className="mt-3 text-sm text-slate-500">
              No reference list was detected in this paper.
            </p>
          )}

          <ul className="mt-3 space-y-3">
            {matched.map((entry, index) => (
              <li key={index} className="text-sm">
                <div className="flex items-baseline justify-between gap-2">
                  <a
                    href={entry.url ?? undefined}
                    target="_blank"
                    rel="noreferrer"
                    className="font-medium text-slate-800 hover:underline"
                  >
                    {entry.matched_title ?? entry.title}
                  </a>
                  {entry.cited_by !== null && (
                    <span className="shrink-0 text-xs tabular-nums text-slate-400">
                      {entry.cited_by.toLocaleString()} citations
                    </span>
                  )}
                </div>
                <p className="text-xs text-slate-500">
                  {[entry.first_author, entry.year, entry.venue].filter(Boolean).join(' · ')}
                </p>
                {entry.abstract && (
                  <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-slate-600">
                    {entry.abstract}
                  </p>
                )}
              </li>
            ))}
          </ul>

          {unmatched.length > 0 && (
            <details className="mt-4">
              <summary className="cursor-pointer text-xs text-slate-500">
                {unmatched.length} reference{unmatched.length === 1 ? '' : 's'} not matched on OpenAlex
              </summary>
              <ul className="mt-2 space-y-1">
                {unmatched.map((entry, index) => (
                  <li key={index} className="text-xs text-slate-500">
                    {entry.title ?? entry.raw.slice(0, 120)}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}
    </div>
  )
}
