import { useEffect, useRef, useState } from 'react'
import * as pdfjs from 'pdfjs-dist'
import type { PDFDocumentLoadingTask, PDFDocumentProxy } from 'pdfjs-dist'

import { getStoredToken, papersApi, refreshAccessToken } from '../api/client'

/**
 * Renders one page of the paper's PDF in the browser, and highlights a phrase
 * on it when asked.
 *
 * Why the browser and not the server: the unconstrained design would
 * rasterise the page server-side with PyMuPDF and draw the highlight on it.
 * PyMuPDF is the library that was removed to fit the 512 MB budget -- a
 * rendering engine resident for the life of the process. The browser already
 * has one. So the backend streams the bytes (`GET /papers/{id}/file`) and
 * PDF.js does everything else here, at zero server memory.
 *
 * How the highlight works: PDF.js exposes a page's text as positioned items
 * with measured widths (the "text layer"). We join the items' text, look for
 * the quoted phrase in the joined string, and draw a yellow rectangle over
 * every item that overlaps the match, using PDF.js's own geometry for the
 * box. It is a string match against the extracted text, not anything the
 * model produced. When the match fails (a badly extracted page, a phrase
 * that crosses a column break) the viewer still jumps to the page and says
 * the phrase could not be located, rather than highlighting the wrong line.
 */

// PDF.js does its parsing on a web worker so the UI stays responsive. Vite
// resolves this URL at build time and serves the worker as its own chunk.
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString()

export interface Highlight {
  page: number
  /** Verified quote, or null to jump to the page without highlighting. */
  phrase: string | null
  /** Changes each time the same citation is clicked again, so it re-scrolls. */
  nonce: number
}

type TextItem = { str: string; transform: number[]; width: number; height: number }

function normalise(text: string): string {
  return text
    .replace(/[‘’]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/[–—−]/g, '-')
    .replace(/\s+/g, ' ')
    .trim()
}

export function PdfViewer({
  paperId,
  highlight,
  onClose,
}: {
  paperId: string
  highlight: Highlight | null
  onClose: () => void
}) {
  const [doc, setDoc] = useState<PDFDocumentProxy | null>(null)
  const [page, setPage] = useState(1)
  const [error, setError] = useState<string | null>(null)
  const [found, setFound] = useState<boolean | null>(null)
  const [scale, setScale] = useState(1.2)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const textLayerRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // --- Load the document once ---------------------------------------------
  useEffect(() => {
    let cancelled = false
    let task: PDFDocumentLoadingTask | null = null

    // PDF.js can fetch a URL itself, but it cannot send our Authorization
    // header. So fetch the bytes ourselves and hand it the ArrayBuffer.
    const load = (token: string | null) =>
      fetch(papersApi.fileUrl(paperId), {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      })
    load(getStoredToken())
      .then(async (response) => {
        if (response.status === 401) {
          // fetch() bypasses the axios interceptor; repeat its one-shot refresh.
          const token = await refreshAccessToken()
          if (token) response = await load(token)
        }
        if (!response.ok) {
          const body = await response.json().catch(() => ({}))
          throw new Error(body.detail ?? `Could not load the PDF (${response.status}).`)
        }
        return response.arrayBuffer()
      })
      .then((data) => {
        task = pdfjs.getDocument({ data })
        return task.promise
      })
      .then((pdf) => {
        if (!cancelled) setDoc(pdf)
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message)
      })

    return () => {
      cancelled = true
      // Frees the worker and every parsed page. Without it, navigating
      // between papers leaks a whole PDF.js instance per visit.
      void task?.destroy()
    }
  }, [paperId])

  // --- Follow the highlight -----------------------------------------------
  useEffect(() => {
    if (highlight) setPage(highlight.page)
  }, [highlight])

  // --- Render the current page --------------------------------------------
  useEffect(() => {
    if (!doc) return
    const canvas = canvasRef.current
    const textLayer = textLayerRef.current
    if (!canvas || !textLayer) return

    let cancelled = false
    const pageNumber = Math.min(Math.max(1, page), doc.numPages)

    doc.getPage(pageNumber).then(async (pdfPage) => {
      if (cancelled) return
      const viewport = pdfPage.getViewport({ scale })

      // Draw the page.
      canvas.width = viewport.width
      canvas.height = viewport.height
      const context = canvas.getContext('2d')
      if (!context) return
      await pdfPage.render({ canvas, canvasContext: context, viewport }).promise
      if (cancelled) return

      // Lay the text over it. Each item is positioned with the same transform
      // PDF.js used to draw it, so the invisible text sits exactly on the
      // glyphs -- that is what makes the highlight land on the right words.
      const content = await pdfPage.getTextContent()
      if (cancelled) return
      textLayer.innerHTML = ''
      textLayer.style.width = `${viewport.width}px`
      textLayer.style.height = `${viewport.height}px`

      const items = content.items as TextItem[]
      const needle = highlight?.phrase && highlight.page === pageNumber
        ? normalise(highlight.phrase)
        : null

      // Find which items make up the phrase. Text items are fragments (a word,
      // a line, part of a line), so we concatenate them and look for the
      // phrase in the joined string, then mark every item that overlaps the
      // matched range. Whitespace is normalised on both sides.
      const marks = new Set<number>()
      if (needle) {
        let joined = ''
        const spans: Array<[number, number]> = []
        items.forEach((item) => {
          const text = normalise(item.str)
          const start = joined.length
          joined += (joined && text ? ' ' : '') + text
          spans.push([start, joined.length])
        })
        const at = joined.indexOf(needle)
        if (at !== -1) {
          const end = at + needle.length
          spans.forEach(([s, e], index) => {
            if (e > at && s < end && items[index].str.trim()) marks.add(index)
          })
        }
        setFound(marks.size > 0)
      } else {
        setFound(null)
      }

      let firstMark: HTMLDivElement | null = null
      items.forEach((item, index) => {
        if (!item.str.trim()) return
        // transform = [a, b, c, d, e, f]: (e, f) is the baseline origin in
        // canvas pixels, and the font size is the scale of the a/b axis.
        const [a, b, , , e, f] = pdfjs.Util.transform(viewport.transform, item.transform)
        const fontSize = Math.hypot(a, b)
        const width = item.width * viewport.scale
        const height = (item.height || fontSize / viewport.scale) * viewport.scale

        // Invisible, selectable text over the glyphs, so the page can be
        // copied from like any PDF viewer.
        const span = document.createElement('span')
        span.textContent = item.str
        span.style.cssText =
          `position:absolute;left:${e}px;top:${f - height}px;font-size:${fontSize}px;` +
          `font-family:sans-serif;white-space:pre;color:transparent;line-height:1;` +
          `width:${width}px;overflow:hidden;`
        textLayer.appendChild(span)

        if (marks.has(index)) {
          const box = document.createElement('div')
          box.style.cssText =
            `position:absolute;left:${e - 1}px;top:${f - height - 1}px;` +
            `width:${width + 2}px;height:${height + 2}px;` +
            `background:rgba(250,204,21,0.45);border-radius:2px;pointer-events:none;`
          textLayer.appendChild(box)
          firstMark ??= box
        }
      })

      // Scroll the highlighted phrase into view inside the viewer, not the
      // whole page -- the chat column must not jump.
      const target = firstMark ?? containerRef.current
      if (target && containerRef.current) {
        const box = target.getBoundingClientRect()
        const outer = containerRef.current.getBoundingClientRect()
        containerRef.current.scrollTop += box.top - outer.top - 80
      }
    }).catch((err: Error) => {
      if (!cancelled) setError(err.message)
    })

    return () => {
      cancelled = true
    }
  }, [doc, page, scale, highlight])

  if (error) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <div className="flex items-center justify-between">
          <p className="text-sm font-medium text-slate-900">PDF</p>
          <button onClick={onClose} className="text-xs text-slate-500 hover:text-slate-900">
            Close
          </button>
        </div>
        <p className="mt-2 text-sm text-red-700">{error}</p>
      </div>
    )
  }

  const total = doc?.numPages ?? 0

  return (
    <div className="flex h-full min-h-0 flex-col rounded-lg border border-slate-200 bg-white">
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-slate-100 px-3 py-2 text-xs">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
            className="rounded px-2 py-1 text-slate-600 hover:bg-slate-100 disabled:opacity-30"
          >
            ‹
          </button>
          <span className="tabular-nums text-slate-700">
            page {page}{total ? ` / ${total}` : ''}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(total || p + 1, p + 1))}
            disabled={total > 0 && page >= total}
            className="rounded px-2 py-1 text-slate-600 hover:bg-slate-100 disabled:opacity-30"
          >
            ›
          </button>
          <button
            onClick={() => setScale((s) => Math.max(0.6, +(s - 0.2).toFixed(1)))}
            className="rounded px-2 py-1 text-slate-600 hover:bg-slate-100"
            title="Zoom out"
          >
            −
          </button>
          <button
            onClick={() => setScale((s) => Math.min(2.4, +(s + 0.2).toFixed(1)))}
            className="rounded px-2 py-1 text-slate-600 hover:bg-slate-100"
            title="Zoom in"
          >
            +
          </button>
        </div>

        <div className="flex items-center gap-3">
          {found === true && (
            <span className="rounded bg-amber-50 px-2 py-0.5 text-amber-800">
              quote highlighted
            </span>
          )}
          {found === false && (
            <span
              className="rounded bg-slate-100 px-2 py-0.5 text-slate-500"
              title="The quoted phrase is in this page's extracted text, but the viewer could not line it up with the rendered glyphs -- usually a two-column layout."
            >
              on this page · could not pinpoint
            </span>
          )}
          <button onClick={onClose} className="text-slate-500 hover:text-slate-900">
            Close
          </button>
        </div>
      </div>

      <div ref={containerRef} className="min-h-0 flex-1 overflow-auto bg-slate-100 p-3">
        {!doc && (
          <p className="py-12 text-center text-sm text-slate-400">Loading the PDF…</p>
        )}
        <div className="relative mx-auto w-fit shadow-sm">
          <canvas ref={canvasRef} className="block" />
          <div ref={textLayerRef} className="absolute top-0 left-0 select-text" />
        </div>
      </div>
    </div>
  )
}
