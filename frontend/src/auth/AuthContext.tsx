/**
 * Who is logged in, shared across the whole app.
 *
 * Why Context here and nowhere else: the token and current user are needed by
 * the navbar, the route guard, and the login/logout actions -- components at
 * completely different depths of the tree. Passing them down as props would mean
 * threading them through every component in between ("prop drilling"). Context
 * is React's built-in answer to exactly that, and it is one concept rather than
 * a whole library. Everything else in this app stays plain useState, which is
 * why there is no Redux or React Query here.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'

import { authApi, clearToken, refreshAccessToken, storeToken } from '../api/client'
import type { User } from '../api/types'

interface AuthState {
  user: User | null
  /** true until the silent refresh on first load has settled */
  loading: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string, fullName?: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  // On first load there is no access token in memory -- by design. The
  // refresh cookie, if the browser has one, is exchanged for a fresh access
  // token, then we ask who it belongs to. That round trip is why `loading`
  // exists: without it the app would flash the login page for a moment before
  // recognising the user.
  useEffect(() => {
    refreshAccessToken()
      .then((token) => (token ? authApi.me() : null))
      .then((me) => setUser(me))
      .catch(() => clearToken())
      .finally(() => setLoading(false))
  }, [])

  const login = useCallback(async (email: string, password: string) => {
    const data = await authApi.login(email, password)
    storeToken(data.access_token)
    setUser(data.user)
  }, [])

  const register = useCallback(
    async (email: string, password: string, fullName?: string) => {
      const data = await authApi.register(email, password, fullName)
      storeToken(data.access_token)
      setUser(data.user)
    },
    [],
  )

  const logout = useCallback(() => {
    // Clear local state first so the UI responds instantly; the server call
    // that clears the cookie is best-effort.
    clearToken()
    setUser(null)
    void authApi.logout().catch(() => undefined)
  }, [])

  // useMemo so the context value is a stable object. Without it a new object
  // would be created on every render, re-rendering every consumer needlessly.
  const value = useMemo(
    () => ({ user, loading, login, register, logout }),
    [user, loading, login, register, logout],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

/** Read the auth state. Throws if used outside the provider -- that is a bug, not a state. */
// eslint-disable-next-line react-refresh/only-export-components
export function useAuth(): AuthState {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used inside <AuthProvider>')
  }
  return context
}
