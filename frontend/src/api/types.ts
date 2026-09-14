/**
 * TypeScript mirrors of the Pydantic schemas in backend/app/schemas.py.
 *
 * Hand-written on purpose. Generating them from the OpenAPI spec is possible
 * (FastAPI publishes one at /openapi.json) but adds a build step and a generator
 * to explain; for a contract this small, typing it by hand keeps the whole data
 * flow readable. If the backend schema changes, change it here too -- that
 * coupling is the trade-off, and it is worth naming out loud.
 */

export interface User {
  id: string
  email: string
  full_name: string | null
  created_at: string
}

export interface Token {
  access_token: string
  token_type: string
  user: User
}

/** pending -> processing -> ready | failed */
export type PaperStatus = 'pending' | 'processing' | 'ready' | 'failed'

export interface Paper {
  id: string
  title: string
  filename: string
  /** Set when the paper was fetched from an arXiv/DOI link rather than uploaded. */
  source_url: string | null
  authors: string | null
  venue: string | null
  year: string | null
  abstract: string | null
  num_pages: number
  num_chunks: number
  /** Text came from model transcription of a scanned PDF, not a text layer. */
  ocr: boolean
  status: PaperStatus
  error_message: string | null
  created_at: string
}

export interface Section {
  id: string
  name: string
  kind: string
  ordinal: number
  page_start: number
  page_end: number
}

/** Progress of the background ingest + analysis run. */
export interface JobStatus {
  id: string
  status: 'queued' | 'running' | 'done' | 'failed'
  stage: 'queued' | 'parsing' | 'embedding' | 'analyzing' | 'synthesizing' | 'done'
  stage_index: number
  stage_count: number
  current: number
  total: number
  message: string
  error_message: string | null
}

export interface GlossaryTerm {
  term: string
  definition: string
}

/**
 * The deep-analysis report. Every field is rendered as its own collapsible
 * block, which is why they are separate fields rather than one blob of prose.
 */
export interface PaperReport {
  plain_language: string
  problem: string
  contributions: string[]
  method_walkthrough: string
  experimental_setup: string
  key_results: string[]
  results_interpretation: string
  limitations_stated: string[]
  limitations_observed: string[]
  assumptions: string[]
  prior_work: string
  reproducibility: string
  open_questions: string[]
  glossary: GlossaryTerm[]
}

/**
 * A pointer back into the paper. `label` is pre-rendered by the backend
 * ("page 4", "pages 4-6, section: Method") so this app does not reimplement
 * that formatting in a second language.
 */
export interface Citation {
  chunk_id: string
  page_start: number
  page_end: number
  section: string | null
  label: string
  snippet: string
  /**
   * The phrase the model said supports the claim -- present only if it was
   * found verbatim in the cited passage. `verified` is true exactly then; the
   * backend keeps them separate so this code never has to know the rule.
   */
  quote: string | null
  verified: boolean
}

export interface ChatResponse {
  answer: string
  citations: Citation[]
  /** false when the answer was not found in the paper, or had no valid citations */
  grounded: boolean
  followups: string[]
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  created_at: string
  citations: Citation[]
}

/** Events on the SSE chat stream. */
export type StreamEvent =
  | { type: 'status'; message: string }
  | { type: 'token'; text: string }
  | {
      type: 'done'
      answer: string
      citations: Citation[]
      grounded: boolean
      followups: string[]
    }
  | { type: 'error'; detail: string; retryable: boolean }

export type ClaimVerdict = 'supports' | 'contradicts' | 'partial' | 'not_addressed'

export interface ClaimEvidence {
  citation: Citation
  quote: string
  verified: boolean
}

/** A structured verdict on one claim; see check_claim in chat_service.py. */
export interface ClaimResponse {
  claim: string
  verdict: ClaimVerdict
  reasoning: string
  evidence: ClaimEvidence[]
  caveats: string
  citations: Citation[]
}

export interface CompareCitation extends Citation {
  paper_id: string
  paper_label: string
  paper_title: string
}

export interface CompareResponse {
  answer: string
  citations: CompareCitation[]
  grounded: boolean
  papers: { label: string; id: string; title: string }[]
}

export interface RelatedEntry {
  raw: string
  title: string | null
  year: string | null
  first_author: string | null
  openalex_id: string | null
  matched_title: string | null
  abstract: string | null
  venue: string | null
  cited_by: number | null
  url: string | null
}

export interface RelatedOut {
  status: 'ready' | 'pending'
  entries: RelatedEntry[]
}

/** Totals from the LLM call ledger, for one paper. */
export interface Usage {
  calls: number
  failed: number
  prompt_tokens: number
  output_tokens: number
  total_latency_ms: number
  by_purpose: Record<string, number>
}

export interface MemoryInfo {
  rss_mb: number
  vms_mb: number
  percent_of_limit: number
  limit_mb: number
  python_objects: number
  /** The applied threadpool cap. 40 here means the cap did NOT take effect. */
  thread_limit: number
  queued_jobs: number
}
