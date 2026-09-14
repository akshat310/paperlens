import { useEffect, useState } from 'react'

import { downloadExport, errorMessage, papersApi } from '../api/client'
import type { Paper, PaperReport, Usage } from '../api/types'

/**
 * The deep-analysis report, as collapsible sections.
 *
 * Collapsible because the report is long by design -- that was the whole point
 * of replacing the one-shot summary -- and a wall of twelve headings with no
 * way to scan it is worse than the shallow version it replaced. The two
 * orienting sections open by default; everything else is one click away.
 *
 * The section list is data rather than markup so the order lives in one place,
 * next to the same list in the backend's exporter.
 */

type Field = keyof PaperReport

interface Block {
  field: Field
  title: string
  hint?: string
  defaultOpen?: boolean
}

const BLOCKS: Block[] = [
  {
    field: 'plain_language',
    title: 'In plain language',
    hint: 'For a smart reader outside the field',
    defaultOpen: true,
  },
  { field: 'problem', title: 'The problem', defaultOpen: true },
  { field: 'contributions', title: 'Contributions' },
  {
    field: 'method_walkthrough',
    title: 'How the method works',
    hint: 'The actual mechanism, step by step',
  },
  { field: 'experimental_setup', title: 'Experimental setup' },
  { field: 'key_results', title: 'Key results' },
  { field: 'results_interpretation', title: 'What the results show' },
  { field: 'limitations_stated', title: 'Limitations the authors state' },
  {
    field: 'limitations_observed',
    title: 'Limitations a careful reader would notice',
    hint: 'Not raised by the paper itself',
  },
  { field: 'assumptions', title: 'Assumptions' },
  { field: 'prior_work', title: 'Relation to prior work' },
  { field: 'reproducibility', title: 'Reproducibility' },
  { field: 'open_questions', title: 'Open questions' },
]

/**
 * Highlights inline "(Section, page N)" references the model was told to emit.
 *
 * Purely presentational -- unlike a chat citation, these are not clickable,
 * because they are the model's own prose references rather than markers we
 * resolved against chunk ids. Styling them differently is the honest signal:
 * they point you at a place to look, they are not a verified link.
 */
const REFERENCE_SOURCE = String.raw`(\([^()]*\bp(?:age|p)?\.?\s*\d+[^()]*\))`
// Two regexes from one source. A /g regex carries `lastIndex` between calls, so
// reusing the split pattern for .test() would return alternating true/false on
// identical input -- the bug shows up as every other reference losing its
// styling. The test regex is deliberately not global.
const REFERENCE_SPLIT = new RegExp(REFERENCE_SOURCE, 'gi')
const REFERENCE_TEST = new RegExp(`^${REFERENCE_SOURCE}$`, 'i')

function Prose({ text }: { text: string }) {
  return (
    <>
      {text.split('\n').map((paragraph, index) =>
        paragraph.trim() ? (
          <p key={index} className="mt-2 text-sm leading-relaxed text-slate-700 first:mt-0">
            {paragraph.split(REFERENCE_SPLIT).map((part, i) =>
              REFERENCE_TEST.test(part) ? (
                <span key={i} className="text-xs text-slate-400">
                  {part}
                </span>
              ) : (
                <span key={i}>{part}</span>
              ),
            )}
          </p>
        ) : null,
      )}
    </>
  )
}

function Collapsible({ block, report }: { block: Block; report: PaperReport }) {
  const [open, setOpen] = useState(Boolean(block.defaultOpen))
  const value = report[block.field]

  // An empty field means the paper genuinely did not cover it -- the prompt
  // says to return nothing rather than pad. Rendering an empty heading would
  // look like a bug, so the block is skipped entirely.
  const isEmpty =
    !value || (Array.isArray(value) && value.length === 0) || value === ''
  if (isEmpty) return null

  const count = Array.isArray(value) ? value.length : null

  return (
    <div className="border-b border-slate-100 last:border-b-0">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-3 py-3 text-left"
      >
        <span className="flex items-baseline gap-2">
          <span className="text-sm font-medium text-slate-900">{block.title}</span>
          {count !== null && (
            <span className="text-xs tabular-nums text-slate-400">{count}</span>
          )}
        </span>
        <span
          className={`shrink-0 text-slate-400 transition-transform ${open ? 'rotate-90' : ''}`}
          aria-hidden
        >
          ›
        </span>
      </button>

      {open && (
        <div className="pb-4">
          {block.hint && (
            <p className="mb-2 text-xs text-slate-400">{block.hint}</p>
          )}
          {Array.isArray(value) ? (
            <ul className="space-y-2">
              {(value as string[]).map((item, index) => (
                <li key={index} className="flex gap-2 text-sm leading-relaxed text-slate-700">
                  <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-slate-300" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          ) : (
            <Prose text={String(value)} />
          )}
        </div>
      )}
    </div>
  )
}

function Glossary({ report }: { report: PaperReport }) {
  const [open, setOpen] = useState(false)
  if (report.glossary.length === 0) return null

  return (
    <div className="border-t border-slate-100">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-3 py-3 text-left"
      >
        <span className="flex items-baseline gap-2">
          <span className="text-sm font-medium text-slate-900">Glossary</span>
          <span className="text-xs tabular-nums text-slate-400">
            {report.glossary.length}
          </span>
        </span>
        <span
          className={`shrink-0 text-slate-400 transition-transform ${open ? 'rotate-90' : ''}`}
          aria-hidden
        >
          ›
        </span>
      </button>

      {open && (
        <dl className="space-y-2 pb-4">
          {report.glossary.map((entry) => (
            <div key={entry.term}>
              <dt className="text-sm font-medium text-slate-800">{entry.term}</dt>
              <dd className="text-sm leading-relaxed text-slate-600">{entry.definition}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  )
}

export function ReportPanel({
  paper,
  report,
  onRegenerate,
}: {
  paper: Paper
  report: PaperReport
  onRegenerate: () => void
}) {
  const [busy, setBusy] = useState<'md' | 'pdf' | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [usage, setUsage] = useState<Usage | null>(null)

  // What this paper has cost so far, from the call ledger. Fetched once per
  // report render; failures just hide the line.
  useEffect(() => {
    papersApi.usage(paper.id).then(setUsage).catch(() => setUsage(null))
  }, [paper.id, report])

  async function handleExport(format: 'md' | 'pdf') {
    setBusy(format)
    setError(null)
    try {
      const stem = paper.title.replace(/[^\w\s-]/g, '').trim().replace(/\s+/g, '-')
      await downloadExport(paper.id, format, `${stem.slice(0, 60) || 'report'}.${format}`)
    } catch (err) {
      setError(errorMessage(err, 'Could not download the export.'))
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="rounded-lg border border-slate-200 bg-white">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 px-4 py-3">
        <div className="flex items-baseline gap-2">
          <h2 className="font-medium text-slate-900">Analysis</h2>
          {usage && usage.calls > 0 && (
            <span
              className="text-xs tabular-nums text-slate-400"
              title={Object.entries(usage.by_purpose)
                .map(([k, v]) => `${k}: ${v}`)
                .join(' · ')}
            >
              {usage.calls} calls · {((usage.prompt_tokens + usage.output_tokens) / 1000).toFixed(0)}k tokens ·{' '}
              {(usage.total_latency_ms / 1000).toFixed(0)}s
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => void handleExport('md')}
            disabled={busy !== null}
            className="rounded px-2 py-1 text-xs text-slate-500 transition hover:bg-slate-100 hover:text-slate-900 disabled:opacity-40"
          >
            {busy === 'md' ? 'Preparing…' : 'Markdown'}
          </button>
          <button
            onClick={() => void handleExport('pdf')}
            disabled={busy !== null}
            className="rounded px-2 py-1 text-xs text-slate-500 transition hover:bg-slate-100 hover:text-slate-900 disabled:opacity-40"
          >
            {busy === 'pdf' ? 'Preparing…' : 'PDF'}
          </button>
          <button
            onClick={onRegenerate}
            className="rounded px-2 py-1 text-xs text-slate-500 transition hover:bg-slate-100 hover:text-slate-900"
            title="Re-run the analysis from scratch"
          >
            Regenerate
          </button>
        </div>
      </div>

      {error && (
        <p className="mx-4 mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">
          {error}
        </p>
      )}

      <div className="px-4">
        {BLOCKS.map((block) => (
          <Collapsible key={block.field} block={block} report={report} />
        ))}
        <Glossary report={report} />
      </div>
    </div>
  )
}
