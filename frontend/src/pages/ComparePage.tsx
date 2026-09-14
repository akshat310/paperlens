import { useEffect, useState, type FormEvent } from 'react'
import { Link, useSearchParams } from 'react-router'

import { errorMessage, papersApi } from '../api/client'
import type { CompareResponse, Paper } from '../api/types'
import { MessageBubble } from '../components/MessageBubble'
import { Spinner } from '../components/Spinner'

/**
 * Ask one question across two or three papers.
 *
 * The selection arrives in the URL (?ids=a,b,c) so a comparison can be
 * linked to and reloaded. Answers are rendered with the same MessageBubble as
 * chat: the backend labels each citation "Paper B, pages 4-5", so no new
 * rendering logic is needed -- only the paper key above the answer.
 *
 * Comparisons are not saved as chat history (they belong to no single paper),
 * so this page keeps its exchanges in component state only.
 */

interface Exchange {
  question: string
  result: CompareResponse
}

export function ComparePage() {
  const [params, setParams] = useSearchParams()
  const [papers, setPapers] = useState<Paper[]>([])
  const [selected, setSelected] = useState<string[]>(
    (params.get('ids') ?? '').split(',').filter(Boolean),
  )
  const [question, setQuestion] = useState('')
  const [exchanges, setExchanges] = useState<Exchange[]>([])
  const [asking, setAsking] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    papersApi
      .list()
      .then((all) => setPapers(all.filter((p) => p.status === 'ready')))
      .catch((err) => setError(errorMessage(err, 'Could not load your papers.')))
  }, [])

  function toggle(id: string) {
    setSelected((current) => {
      const next = current.includes(id)
        ? current.filter((x) => x !== id)
        : current.length >= 3
          ? current
          : [...current, id]
      setParams(next.length ? { ids: next.join(',') } : {}, { replace: true })
      return next
    })
  }

  async function ask(event: FormEvent) {
    event.preventDefault()
    const text = question.trim()
    if (text.length < 3 || selected.length < 2 || asking) return
    setError(null)
    setAsking(true)
    try {
      const result = await papersApi.compare(selected, text)
      setExchanges((current) => [...current, { question: text, result }])
      setQuestion('')
    } catch (err) {
      setError(errorMessage(err, 'Could not compare those papers.'))
    } finally {
      setAsking(false)
    }
  }

  const chosen = selected
    .map((id) => papers.find((p) => p.id === id))
    .filter((p): p is Paper => Boolean(p))

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <Link to="/" className="text-sm text-slate-500 hover:text-slate-900">
        &larr; All papers
      </Link>
      <h1 className="mt-1 text-2xl font-semibold text-slate-900">Compare papers</h1>
      <p className="mt-1 text-sm text-slate-500">
        Pick two or three papers, then ask a question across them. Each citation
        says which paper it came from.
      </p>

      <div className="mt-6 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {papers.map((paper) => {
          const index = selected.indexOf(paper.id)
          const label = index >= 0 ? 'ABC'[index] : null
          const disabled = index < 0 && selected.length >= 3
          return (
            <button
              key={paper.id}
              onClick={() => toggle(paper.id)}
              disabled={disabled}
              className={`flex items-start gap-3 rounded-lg border p-3 text-left text-sm transition disabled:opacity-40 ${
                label
                  ? 'border-slate-900 bg-slate-50'
                  : 'border-slate-200 bg-white hover:border-slate-400'
              }`}
            >
              <span
                className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded text-xs font-semibold ${
                  label ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-400'
                }`}
              >
                {label ?? '+'}
              </span>
              <span className="min-w-0">
                <span className="line-clamp-2 font-medium text-slate-900">{paper.title}</span>
                <span className="block text-xs text-slate-500">
                  {[paper.year, `${paper.num_pages} pages`].filter(Boolean).join(' · ')}
                </span>
              </span>
            </button>
          )
        })}
        {papers.length === 0 && (
          <p className="text-sm text-slate-500">No analysed papers yet.</p>
        )}
      </div>

      <form onSubmit={ask} className="mt-6 flex gap-2">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder={
            selected.length < 2
              ? 'Select at least two papers first…'
              : 'e.g. How do their training setups differ?'
          }
          disabled={selected.length < 2 || asking}
          className="flex-1 rounded-md border border-slate-300 bg-white px-3 py-2 outline-none focus:border-slate-900 disabled:bg-slate-50"
        />
        <button
          type="submit"
          disabled={selected.length < 2 || asking || question.trim().length < 3}
          className="rounded-md bg-slate-900 px-4 py-2 font-medium text-white transition hover:bg-slate-800 disabled:opacity-40"
        >
          {asking ? <Spinner /> : 'Compare'}
        </button>
      </form>

      {error && (
        <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>
      )}

      <div className="mt-6 space-y-6">
        {exchanges.map((exchange, index) => (
          <div key={index} className="space-y-3">
            <div className="flex justify-end">
              <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-slate-900 px-4 py-2.5 text-white">
                {exchange.question}
              </div>
            </div>
            <div className="flex flex-wrap gap-2 text-xs text-slate-500">
              {exchange.result.papers.map((p) => (
                <span key={p.id} className="rounded bg-slate-100 px-2 py-0.5">
                  <span className="font-semibold text-slate-700">{p.label}</span> · {p.title}
                </span>
              ))}
            </div>
            <MessageBubble
              message={{
                id: `compare-${index}`,
                role: 'assistant',
                content: exchange.result.answer,
                created_at: new Date().toISOString(),
                citations: exchange.result.citations,
              }}
            />
          </div>
        ))}
      </div>

      {chosen.length >= 2 && exchanges.length === 0 && (
        <p className="mt-6 text-sm text-slate-400">
          Comparing {chosen.map((p, i) => `${'ABC'[i]}: ${p.title}`).join('  ·  ')}
        </p>
      )}
    </div>
  )
}
