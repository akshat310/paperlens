/**
 * Routing table for the whole app.
 *
 * Client-side routing: React Router swaps components based on the URL without a
 * page reload, so navigation keeps React state and feels instant. The cost is
 * that the server must serve index.html for every path -- handled by the Vite
 * dev server, and by the SPA catch-all in backend/app/main.py.
 */

import { BrowserRouter, Navigate, Route, Routes } from 'react-router'

import { AuthProvider } from './auth/AuthContext'
import { Layout } from './components/Layout'
import { ProtectedRoute } from './components/ProtectedRoute'
import { WakeUpGate } from './components/WakeUpGate'
import { ComparePage } from './pages/ComparePage'
import { DashboardPage } from './pages/DashboardPage'
import { LoginPage } from './pages/LoginPage'
import { PaperPage } from './pages/PaperPage'
import { RegisterPage } from './pages/RegisterPage'

export default function App() {
  return (
    // AuthProvider sits above the router so every route can read the user, and
    // BrowserRouter sits above Layout so the header can render <Link>s.
    <AuthProvider>
      <BrowserRouter>
        <Layout>
          {/* Inside Layout so the header stays visible while the free instance
              wakes up -- a fully blank page would read as a hard failure. */}
          <WakeUpGate>
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route path="/register" element={<RegisterPage />} />

              <Route
                path="/"
                element={
                  <ProtectedRoute>
                    <DashboardPage />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/compare"
                element={
                  <ProtectedRoute>
                    <ComparePage />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/papers/:paperId"
                element={
                  <ProtectedRoute>
                    <PaperPage />
                  </ProtectedRoute>
                }
              />

              {/* Anything unrecognised goes home rather than rendering nothing. */}
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </WakeUpGate>
        </Layout>
      </BrowserRouter>
    </AuthProvider>
  )
}
