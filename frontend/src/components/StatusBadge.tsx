import type { PaperStatus } from '../api/types'
import { Spinner } from './Spinner'

// A lookup object rather than a chain of ifs. Tailwind classes must appear as
// complete literal strings somewhere in the source -- building them dynamically
// (`bg-${color}-100`) means the compiler never sees them and they get dropped.
const STYLES: Record<PaperStatus, { label: string; className: string }> = {
  pending: { label: 'Queued', className: 'bg-slate-100 text-slate-600' },
  processing: { label: 'Processing', className: 'bg-amber-100 text-amber-700' },
  ready: { label: 'Ready', className: 'bg-emerald-100 text-emerald-700' },
  failed: { label: 'Failed', className: 'bg-red-100 text-red-700' },
}

export function StatusBadge({ status }: { status: PaperStatus }) {
  const { label, className } = STYLES[status]
  const busy = status === 'pending' || status === 'processing'

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ${className}`}
    >
      {busy && <Spinner className="h-3 w-3" />}
      {label}
    </span>
  )
}
