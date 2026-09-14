import type { ClaimResponse, ClaimVerdict, Citation } from '../api/types'

/**
 * A verdict on one claim. The verdict is a fixed vocabulary from the backend,
 * rendered as a badge; the reasoning carries [n] markers resolved against
 * `citations`, and each evidence quote says whether it was found verbatim in
 * the passage it points at.
 */

const VERDICTS: Record<ClaimVerdict, { label: string; className: string }> = {
  supports: { label: 'Supported', className: 'bg-emerald-50 text-emerald-800 ring-emerald-200' },
  contradicts: { label: 'Contradicted', className: 'bg-red-50 text-red-800 ring-red-200' },
  partial: { label: 'Partly supported', className: 'bg-amber-50 text-amber-800 ring-amber-200' },
  not_addressed: { label: 'Not addressed', className: 'bg-slate-100 text-slate-700 ring-slate-200' },
}

export function ClaimCard({
  result,
  onCite,
}: {
  result: ClaimResponse
  onCite?: (citation: Citation) => void
}) {
  const verdict = VERDICTS[result.verdict] ?? VERDICTS.not_addressed

  return (
    <div className="flex justify-start">
      <div className="max-w-[90%] rounded-2xl rounded-bl-sm border border-slate-200 bg-white px-4 py-3 text-slate-800">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-medium tracking-wide text-slate-400 uppercase">
            Claim check
          </span>
          <span className={`rounded-full px-2 py-0.5 text-xs font-medium ring-1 ${verdict.className}`}>
            {verdict.label}
          </span>
        </div>

        <p className="mt-2 text-sm text-slate-500">“{result.claim}”</p>

        <p className="mt-2 text-sm leading-relaxed">
          {result.reasoning.split(/\[(\d+)\]/).map((part, i) => {
            if (i % 2 === 0) return <span key={i}>{part}</span>
            const n = Number(part)
            const citation = result.citations[n - 1]
            if (!citation) return <span key={i}>[{n}]</span>
            return (
              <button
                key={i}
                onClick={() => onCite?.(citation)}
                title={citation.label}
                className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded bg-slate-200 px-1 align-baseline text-xs font-medium text-slate-700 hover:bg-slate-900 hover:text-white"
              >
                {n}
              </button>
            )
          })}
        </p>

        {result.evidence.length > 0 && (
          <ul className="mt-3 space-y-1.5 border-t border-slate-100 pt-3">
            {result.evidence.map((item, index) => (
              <li key={index} className="text-xs">
                <button
                  onClick={() => onCite?.(item.citation)}
                  className="w-full rounded-md px-2 py-1.5 text-left hover:bg-slate-50"
                >
                  <span className="font-medium text-slate-700">{item.citation.label}</span>
                  {item.verified ? (
                    <span className="ml-1.5 rounded bg-emerald-50 px-1.5 py-0.5 text-[10px] font-medium text-emerald-700">
                      verified quote
                    </span>
                  ) : (
                    <span
                      className="ml-1.5 rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-500"
                      title="The model's phrasing; not found verbatim in the passage."
                    >
                      paraphrase
                    </span>
                  )}
                  <span className="mt-0.5 block text-slate-500">“{item.quote}”</span>
                </button>
              </li>
            ))}
          </ul>
        )}

        {result.caveats && (
          <p className="mt-3 border-t border-slate-100 pt-3 text-xs text-slate-500">
            <span className="font-medium text-slate-600">Caveats: </span>
            {result.caveats}
          </p>
        )}
      </div>
    </div>
  )
}
