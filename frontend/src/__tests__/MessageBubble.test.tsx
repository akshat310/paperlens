import '@testing-library/jest-dom/vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

// Vitest does not auto-clean the DOM between tests unless globals are on.
afterEach(cleanup)

import { MessageBubble } from '../components/MessageBubble'
import type { ChatMessage, Citation } from '../api/types'

const cite = (n: number, verified = false): Citation => ({
  chunk_id: `c${n}`,
  page_start: n,
  page_end: n,
  section: 'Method',
  label: `page ${n}, section: Method`,
  snippet: `snippet ${n}`,
  quote: verified ? `quote ${n}` : null,
  verified,
})

const message = (content: string, citations: Citation[]): ChatMessage => ({
  id: 'm1',
  role: 'assistant',
  content,
  created_at: new Date().toISOString(),
  citations,
})

describe('MessageBubble', () => {
  it('turns [n] markers into chips that resolve to citations[n-1]', () => {
    render(<MessageBubble message={message('Trained with AdamW [1] on CIFAR [2].', [cite(1), cite(2)])} />)
    expect(screen.getByTitle('Jump to source 1')).toBeInTheDocument()
    expect(screen.getByTitle('Jump to source 2')).toBeInTheDocument()
    expect(screen.getByText('[1] page 1, section: Method')).toBeInTheDocument()
  })

  it('leaves an out-of-range marker as plain text rather than a dead chip', () => {
    render(<MessageBubble message={message('See [7].', [cite(1)])} />)
    expect(screen.queryByTitle('Jump to source 7')).not.toBeInTheDocument()
    expect(screen.getByText(/\[7\]/)).toBeInTheDocument()
  })

  it('keeps a mid-stream quoted phrase in the prose', () => {
    render(<MessageBubble message={message('Used [1: "AdamW for 40 epochs"] here.', [cite(1)])} streaming />)
    expect(screen.getByText(/AdamW for 40 epochs/)).toBeInTheDocument()
    expect(screen.getByTitle('Jump to source 1')).toBeInTheDocument()
  })

  it('shows the verified badge only for verified quotes and calls onCite', () => {
    const onCite = vi.fn()
    render(<MessageBubble message={message('A [1] B [2].', [cite(1, true), cite(2)])} onCite={onCite} />)
    expect(screen.getAllByText('verified quote')).toHaveLength(1)
    fireEvent.click(screen.getAllByTitle('Open this page of the PDF')[1])
    expect(onCite).toHaveBeenCalledWith(expect.objectContaining({ chunk_id: 'c2' }))
  })

  it('renders markdown without letting paper text inject markup', () => {
    render(<MessageBubble message={message('**Bold** <img src=x onerror=alert(1)> [1]', [cite(1)])} />)
    expect(screen.getByText('Bold').tagName).toBe('STRONG')
    expect(document.querySelector('img')).toBeNull()
  })
})
