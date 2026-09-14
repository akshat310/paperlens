import { describe, expect, it } from 'vitest'

import { parseSseBuffer } from '../api/client'

describe('parseSseBuffer', () => {
  it('returns complete frames and keeps the partial tail', () => {
    const { events, rest } = parseSseBuffer(
      'data: {"type":"token","text":"Hel"}\n\ndata: {"type":"token","text":"lo"}\n\ndata: {"type":"do',
    )
    expect(events).toEqual([
      { type: 'token', text: 'Hel' },
      { type: 'token', text: 'lo' },
    ])
    expect(rest).toBe('data: {"type":"do')
  })

  it('completes a frame once the rest of it arrives', () => {
    const first = parseSseBuffer('data: {"type":"status","mess')
    expect(first.events).toEqual([])
    const second = parseSseBuffer(first.rest + 'age":"Reading"}\n\n')
    expect(second.events).toEqual([{ type: 'status', message: 'Reading' }])
    expect(second.rest).toBe('')
  })

  it('ignores malformed frames and comment lines without dropping the stream', () => {
    const { events } = parseSseBuffer(': keep-alive\n\ndata: not json\n\ndata: {"type":"token","text":"ok"}\n\n')
    expect(events).toEqual([{ type: 'token', text: 'ok' }])
  })
})
