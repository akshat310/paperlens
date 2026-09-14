import { defineConfig } from '@playwright/test'

// Browser smoke test only. Unit tests are Vitest (`npm test`); this needs a
// running server and real model quota: `BASE_URL=... npx playwright test`.
export default defineConfig({
  testDir: './e2e',
  timeout: 6 * 60 * 1000,
  retries: 0,
  use: {
    headless: true,
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  reporter: [['list']],
  outputDir: 'e2e/results',
})
