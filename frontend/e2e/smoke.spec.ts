import { expect, test } from '@playwright/test'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

/**
 * Browser smoke test against a running server (built frontend served by the
 * backend, or the Vite dev server proxying to it). Uses real model quota:
 * one paper ingested and analysed, one question. Not part of `npm test`.
 *
 *   BASE_URL=http://localhost:8011 npx playwright test
 *
 * What it proves that unit tests cannot: the SPA, the SSE stream, the
 * citation chips, the PDF viewer and the httpOnly refresh cookie all work
 * together in an actual browser.
 */

const BASE = process.env.BASE_URL ?? 'http://localhost:8011'
const here = path.dirname(fileURLToPath(import.meta.url))
const FIXTURE = path.resolve(here, '../../backend/eval/fixture_paper.pdf')

test.setTimeout(6 * 60 * 1000)

test('register, upload, analyse, ask, open the cited page', async ({ page }) => {
  const email = `pw-${Date.now()}@example.com`

  await page.goto(`${BASE}/register`)
  await page.locator('#email').fill(email)
  await page.locator('#password').fill('supersecret1')
  await page.locator('button[type="submit"]').click()
  await expect(page).toHaveURL(/\/$/)

  // Upload the fixture paper.
  await page.locator('input[type="file"]').setInputFiles(FIXTURE)
  await expect(page.getByText(/RAGDoc|fixture_paper/i).first()).toBeVisible({ timeout: 20_000 })

  // Open it and wait for the analysis to finish.
  await page.getByRole('link', { name: /open|view progress/i }).first().click()
  await expect(page.getByRole('heading', { name: /analysis/i })).toBeVisible({ timeout: 5 * 60 * 1000 })

  // Reload: the access token is memory-only, so this exercises the refresh
  // cookie -- a broken cookie flow would land on /login here.
  await page.reload()
  await expect(page.getByRole('heading', { name: /analysis/i })).toBeVisible({ timeout: 30_000 })

  // Ask a question that has a numeric answer, so a quote is likely.
  await page.getByPlaceholder(/ask a question/i).fill('What learning rate was used during training?')
  await page.locator('form button[type="submit"]').click()

  const sources = page.getByText(/^sources$/i)
  await expect(sources).toBeVisible({ timeout: 90_000 })
  await expect(page.getByText('3e-4').first()).toBeVisible()

  // Click "open page" on the first citation: the viewer appears at that page.
  await page.getByRole('button', { name: /open page/i }).first().click()
  await expect(page.getByText(/page \d+ \/ \d+/)).toBeVisible({ timeout: 30_000 })
  await expect(page.locator('canvas')).toBeVisible()
  await page.waitForTimeout(1500) // let the page render before the screenshot
  await page.screenshot({ path: 'e2e/results/viewer.png', fullPage: false })

  // Claim mode returns a verdict badge.
  await page.getByRole('button', { name: /check claim/i }).click()
  await page.getByPlaceholder(/state a claim/i).fill('The system was trained on eight A100 GPUs.')
  await page.locator('form button[type="submit"]').click()
  await expect(page.getByText(/supported|contradicted|partly supported|not addressed/i).first())
    .toBeVisible({ timeout: 90_000 })

  // Log out clears the session; the dashboard is no longer reachable.
  await page.getByRole('button', { name: /log out/i }).click()
  await page.goto(`${BASE}/`)
  await expect(page).toHaveURL(/\/login/)
})
