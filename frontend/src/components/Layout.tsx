/**
 * The page shell: a header with the logged-in user and a logout button, and a
 * content area. Pages render inside it, so the header is written once.
 */

import { Link } from 'react-router'
import type { ReactNode } from 'react'

import { useAuth } from '../auth/AuthContext'

export function Layout({ children }: { children: ReactNode }) {
  const { user, logout } = useAuth()

  return (
    <div className="flex h-full flex-col">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3">
          <Link to="/" className="flex items-center gap-2 font-semibold text-slate-900">
            <span className="text-xl">🔍</span>
            PaperLens
          </Link>

          {user && (
            <div className="flex items-center gap-4 text-sm">
              <span className="hidden text-slate-500 sm:inline">{user.email}</span>
              <button
                onClick={logout}
                className="rounded-md px-3 py-1.5 text-slate-600 transition hover:bg-slate-100 hover:text-slate-900"
              >
                Log out
              </button>
            </div>
          )}
        </div>
      </header>

      {/* min-h-0 lets the chat page scroll internally instead of stretching the
          page. A flex child defaults to min-height:auto, which refuses to shrink
          below its content -- a classic flexbox scrolling gotcha. */}
      <main className="min-h-0 flex-1">{children}</main>
    </div>
  )
}
