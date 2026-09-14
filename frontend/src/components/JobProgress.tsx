import type { JobStatus } from '../api/types'

/**
 * A real progress display for the background ingest + analysis run.
 *
 * The point is that a spinner is a lie: it says "something is happening" for
 * ninety seconds without ever saying what, so a slow run and a hung run look
 * identical. This shows the named stage, the position within it ("section 4 of
 * 11"), and a bar that only ever moves forward.
 *
 * Progress is computed from two levels -- which stage, and how far through it --
 * because the stages take very different amounts of time. Parsing is a second;
 * analysing eleven sections is a minute. Weighting by stage alone would sit at
 * 60% for most of the wait.
 */

const STAGE_LABELS: Record<JobStatus['stage'], string> = {
  queued: 'Queued',
  parsing: 'Reading the PDF',
  embedding: 'Building the search index',
  analyzing: 'Analysing sections',
  synthesizing: 'Writing the report',
  done: 'Done',
}

function percentComplete(job: JobStatus): number {
  if (job.status === 'done') return 100
  // Each stage owns an equal slice of the bar; within a stage, `current/total`
  // fills that slice. A stage with no subdivision sits at its slice's start.
  const slice = 100 / Math.max(1, job.stage_count - 1)
  const within = job.total > 0 ? Math.min(1, job.current / job.total) : 0
  return Math.min(99, Math.round(job.stage_index * slice + within * slice))
}

export function JobProgress({ job }: { job: JobStatus }) {
  if (job.status === 'failed') {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-4">
        <p className="text-sm font-medium text-red-800">Analysis failed</p>
        <p className="mt-1 text-sm text-red-700">
          {job.error_message ?? job.message ?? 'Something went wrong.'}
        </p>
      </div>
    )
  }

  const percent = percentComplete(job)

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex items-baseline justify-between gap-3">
        <p className="text-sm font-medium text-slate-900">{STAGE_LABELS[job.stage]}</p>
        <p className="text-xs tabular-nums text-slate-500">{percent}%</p>
      </div>

      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-100">
        <div
          className="h-full rounded-full bg-slate-900 transition-[width] duration-500 ease-out"
          style={{ width: `${percent}%` }}
        />
      </div>

      {job.message && (
        <p className="mt-2 text-xs text-slate-500">{job.message}</p>
      )}

      {job.total > 0 && (
        <p className="mt-1 text-xs tabular-nums text-slate-400">
          {job.current} of {job.total}
        </p>
      )}
    </div>
  )
}
