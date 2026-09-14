import { useEffect, useState } from 'react'

import { opsApi } from '../api/client'

/**
 * Handles the Render free-tier cold start explicitly.
 *
 * A free instance is suspended after 15 minutes of inactivity and takes 30-60
 * seconds to come back. During that window every request just hangs. Without
 * this the app looks broken on exactly the visit that matters most -- the first
 * one, by someone who has never seen it before.
 *
 * So: ping /health on mount. If it answers quickly, this renders nothing and
 * costs one cheap request. If it does not answer within a couple of seconds,
 * we show a "waking up" panel with an elapsed counter and keep retrying until
 * it comes back.
 *
 * The counter is deliberate. "Please wait" with no feedback feels broken after
 * ten seconds; a number climbing towards a stated expectation reads as a system
 * working, and it tells the truth about why.
 */

const PROBE_INTERVAL_MS = 3000
const SHOW_AFTER_MS = 2000
const EXPECTED_SECONDS = 50

export function WakeUpGate({ children }: { children: React.ReactNode }) {
  const [awake, setAwake] = useState(false)
  const [visible, setVisible] = useState(false)
  const [seconds, setSeconds] = useState(0)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    let cancelled = false

    // Only show the panel if the wait is actually long enough to notice. A warm
    // instance answers in ~50ms and this timer never fires.
    const showTimer = setTimeout(() => {
      if (!cancelled) setVisible(true)
    }, SHOW_AFTER_MS)

    const tick = setInterval(() => {
      if (!cancelled) setSeconds((s) => s + 1)
    }, 1000)

    async function probe(): Promise<void> {
      while (!cancelled) {
        try {
          await opsApi.health()
          if (!cancelled) setAwake(true)
          return
        } catch {
          // Expected while the instance boots. Keep waiting.
          if (!cancelled) {
            setFailed(true)
            await new Promise((resolve) => setTimeout(resolve, PROBE_INTERVAL_MS))
          }
        }
      }
    }

    void probe()

    return () => {
      cancelled = true
      clearTimeout(showTimer)
      clearInterval(tick)
    }
  }, [])

  if (awake || !visible) {
    // Render children immediately once awake -- and also while we are still
    // within the grace period, so a warm instance never flashes a panel.
    return <>{children}</>
  }

  const percent = Math.min(95, Math.round((seconds / EXPECTED_SECONDS) * 100))

  return (
    <div className="flex min-h-[60vh] items-center justify-center px-4">
      <div className="w-full max-w-md rounded-xl border border-slate-200 bg-white p-6 text-center">
        <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-full bg-slate-100">
          <span className="h-4 w-4 animate-pulse rounded-full bg-slate-900" />
        </div>

        <h2 className="mt-4 font-medium text-slate-900">Waking the server up</h2>
        <p className="mt-2 text-sm leading-relaxed text-slate-600">
          This runs on a free instance, which sleeps after 15 minutes of quiet
          and takes about a minute to start again. Nothing is broken — the
          first visit is just slow.
        </p>

        <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-slate-100">
          <div
            className="h-full rounded-full bg-slate-900 transition-[width] duration-1000 ease-linear"
            style={{ width: `${percent}%` }}
          />
        </div>

        <p className="mt-2 text-xs tabular-nums text-slate-400">
          {seconds}s elapsed
          {failed && seconds > EXPECTED_SECONDS + 30 && ' — taking longer than usual'}
        </p>
      </div>
    </div>
  )
}
