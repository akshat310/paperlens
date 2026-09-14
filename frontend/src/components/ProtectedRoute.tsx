/**
 * Route guard: renders its children only for a logged-in user.
 *
 * This is convenience, not security. Anyone can edit client-side state; the real
 * enforcement is the `get_current_user` dependency on every protected endpoint
 * in the backend. This just stops a logged-out visitor from seeing an empty
 * dashboard that 401s on every request.
 */

import { Navigate, useLocation } from 'react-router'
import type { ReactNode } from 'react'

import { useAuth } from '../auth/AuthContext'
import { Spinner } from './Spinner'

export function ProtectedRoute({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth()
  const location = useLocation()

  // While the stored token is being verified we know nothing yet. Redirecting
  // now would kick a legitimately logged-in user back to /login on every refresh.
  if (loading) {
    return (
      <div className="flex h-full items-center justify-center text-slate-400">
        <Spinner className="h-8 w-8" />
      </div>
    )
  }

  if (!user) {
    // `state` remembers where they were heading so login can send them back.
    // `replace` keeps the guarded URL out of history, so the back button does
    // not bounce them straight into this redirect again.
    return <Navigate to="/login" replace state={{ from: location.pathname }} />
  }

  return <>{children}</>
}
