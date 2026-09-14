import { useEffect, useRef, useState } from 'react'
import Markdown from 'react-markdown'

import type { ChatMessage, Citation } from '../api/types'

/**
 * Splits the answer on citation markers while KEEPING them. A capturing group in
 * a split pattern makes JS include the captured text in the output array, so
 * "AdamW [1] on CIFAR [2]" becomes ["AdamW ", "1", " on CIFAR ", "2", ""].
 * Odd indices are therefore marker numbers, even indices are plain text.
 *
 * The backend has already stripped quoted phrases out of the prose (they live
 * on the citation), so only the bare `[n]` form reaches this component --
 * except mid-stream, where the raw model output is shown until `done`.
 */
const MARKER_PATTERN = /\[(\d+)(?::\s*"([^"]*)")?\]/g

/**
 * Renders an answer with its [n] markers turned into clickable chips.
 *
 * This works because the backend renumbers markers so that [n] is exactly
 * citations[n - 1] -- see _extract_citations in chat_service.py. Without that
 * guarantee this component would need the original retrieval list to resolve a
 * marker.
 *
 * Markdown: the model is not forbidden from writing **bold** or lists, and a
 * substantive answer usually does. Rendering it is a client-side concern with
 * no server cost, and `react-markdown` produces React elements (never
 * `dangerouslySetInnerHTML`), so paper text cannot inject markup.
 */
function AnswerText({
  content,
  citationCount,
  selected,
  onSelect,
}: {
  content: string
  citationCount: number
  selected: number | null
  onSelect: (index: number) => void
}) {
  // Markers are replaced with a placeholder token that survives Markdown
  // parsing untouched, then swapped for chips when rendering text nodes. A
  // quoted phrase (only seen mid-stream; the backend inlines it on `done`)
  // stays as prose so the sentence never has a hole in it.
  const withTokens = content.replace(
    MARKER_PATTERN,
    (_, n: string, phrase?: string) => `${phrase ? phrase.trim() + ' ' : ''}⁣${n}⁣`,
  )

  function renderText(text: string) {
    const parts = text.split(/⁣(\d+)⁣/)
    return parts.map((part, i) => {
      if (i % 2 === 0) return part
      const number = Number(part)
      // A marker with no matching citation should never happen, but a dead
      // chip would be worse than rendering the raw text.
      if (number < 1 || number > citationCount) return `[${number}]`
      const isSelected = selected === number - 1
      return (
        <button
          key={i}
          onClick={() => onSelect(number - 1)}
          title={`Jump to source ${number}`}
          className={`mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded px-1 align-baseline text-xs font-medium transition ${
            isSelected
              ? 'bg-slate-900 text-white'
              : 'bg-slate-200 text-slate-700 hover:bg-slate-900 hover:text-white'
          }`}
        >
          {number}
        </button>
      )
    })
  }

  // Walk children of every element and render string children through
  // renderText, so chips can appear inside paragraphs, list items and bold.
  function withChips(children: React.ReactNode): React.ReactNode {
    if (typeof children === 'string') return renderText(children)
    if (Array.isArray(children)) {
      return children.map((child, i) =>
        typeof child === 'string' ? <span key={i}>{renderText(child)}</span> : child,
      )
    }
    return children
  }

  return (
    <div className="space-y-3 leading-relaxed [&_code]:rounded [&_code]:bg-slate-100 [&_code]:px-1 [&_code]:text-[0.9em] [&_h1]:text-base [&_h1]:font-semibold [&_h2]:text-base [&_h2]:font-semibold [&_h3]:font-semibold [&_li]:ml-4 [&_ol]:list-decimal [&_ul]:list-disc">
      <Markdown
        components={{
          p: ({ children }) => <p>{withChips(children)}</p>,
          li: ({ children }) => <li>{withChips(children)}</li>,
          strong: ({ children }) => <strong>{withChips(children)}</strong>,
          em: ({ children }) => <em>{withChips(children)}</em>,
          td: ({ children }) => <td className="border px-2 py-1">{withChips(children)}</td>,
          h1: ({ children }) => <h1>{withChips(children)}</h1>,
          h2: ({ children }) => <h2>{withChips(children)}</h2>,
          h3: ({ children }) => <h3>{withChips(children)}</h3>,
        }}
      >
        {withTokens}
      </Markdown>
    </div>
  )
}

export function MessageBubble({
  message,
  streaming = false,
  onCite,
}: {
  message: ChatMessage
  streaming?: boolean
  /** Called when the reader wants to see a citation in the PDF. */
  onCite?: (citation: Citation) => void
}) {
  const [selected, setSelected] = useState<number | null>(null)
  const sourceRefs = useRef<(HTMLLIElement | null)[]>([])
  const isUser = message.role === 'user'

  // Scroll the selected source into view and let the highlight draw attention.
  // `block: 'nearest'` rather than 'center' so clicking a chip does not yank the
  // whole conversation around when the source is already visible.
  useEffect(() => {
    if (selected === null) return
    sourceRefs.current[selected]?.scrollIntoView({
      behavior: 'smooth',
      block: 'nearest',
    })
  }, [selected])

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-slate-900 px-4 py-2.5 text-white">
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    )
  }

  function toggle(index: number) {
    setSelected((current) => (current === index ? null : index))
  }

  async function copy() {
    // Markdown with the sources as footnotes, so the answer keeps its evidence
    // when pasted somewhere else.
    const sources = message.citations
      .map((c, i) => `[${i + 1}] ${c.label}${c.quote ? ` — "${c.quote}"` : ''}`)
      .join('\n')
    try {
      await navigator.clipboard.writeText(
        sources ? `${message.content}\n\nSources:\n${sources}` : message.content,
      )
    } catch {
      // Clipboard access can be denied; nothing useful to do about it here.
    }
  }

  return (
    <div className="flex justify-start">
      <div className="group max-w-[90%] rounded-2xl rounded-bl-sm border border-slate-200 bg-white px-4 py-3 text-slate-800">
        <AnswerText
          content={message.content}
          citationCount={message.citations.length}
          selected={selected}
          onSelect={toggle}
        />

        {streaming && (
          // A caret while tokens are still arriving, so a pause mid-answer reads
          // as "still writing" rather than "finished, oddly".
          <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-slate-400 align-text-bottom" />
        )}

        {message.citations.length > 0 && (
          <div className="mt-3 border-t border-slate-100 pt-3">
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium tracking-wide text-slate-400 uppercase">
                Sources
              </p>
              <button
                onClick={() => void copy()}
                className="text-xs text-slate-400 opacity-0 transition group-hover:opacity-100 hover:text-slate-900"
                title="Copy the answer with its sources"
              >
                Copy
              </button>
            </div>

            <ul className="mt-2 space-y-1.5">
              {message.citations.map((citation, index) => (
                <li
                  key={citation.chunk_id + index}
                  ref={(element) => {
                    sourceRefs.current[index] = element
                  }}
                >
                  <div
                    className={`w-full rounded-md px-2 py-1.5 text-left text-xs transition ${
                      selected === index
                        ? 'bg-amber-50 ring-1 ring-amber-300'
                        : 'hover:bg-slate-50'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <button onClick={() => toggle(index)} className="min-w-0 flex-1 text-left">
                        <span className="font-medium text-slate-700">
                          [{index + 1}] {citation.label}
                        </span>
                        {citation.verified && (
                          <span
                            className="ml-1.5 rounded bg-emerald-50 px-1.5 py-0.5 text-[10px] font-medium text-emerald-700"
                            title="This exact phrase occurs in the cited passage. Checked by string match, not by the model."
                          >
                            verified quote
                          </span>
                        )}
                        {/* The snippet is real chunk text from the database (or
                            the verified quote), never anything the model wrote --
                            that is what makes it checkable. */}
                        <span
                          className={`mt-0.5 block text-slate-500 ${
                            selected === index ? '' : 'line-clamp-1'
                          }`}
                        >
                          {citation.verified ? `“${citation.snippet}”` : citation.snippet}
                        </span>
                      </button>
                      {onCite && (
                        <button
                          onClick={() => onCite(citation)}
                          className="shrink-0 rounded border border-slate-200 px-1.5 py-0.5 text-[10px] text-slate-600 transition hover:border-slate-900 hover:text-slate-900"
                          title="Open this page of the PDF"
                        >
                          open page
                        </button>
                      )}
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}
