import { useRef, useState, type FormEvent } from 'react'

import { errorMessage, papersApi } from '../api/client'
import { Spinner } from './Spinner'

// Mirrors MAX_UPLOAD_BYTES in backend/app/routers/papers.py. Checking here saves
// uploading 40MB just to be told no; the backend check is the one that counts.
const MAX_BYTES = 25 * 1024 * 1024

export function UploadCard({ onUploaded }: { onUploaded: () => void }) {
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const [url, setUrl] = useState('')
  const [fetching, setFetching] = useState(false)

  // A ref to the hidden <input type="file">. The native file input cannot be
  // styled usefully, so the usual approach is to hide it and click it from a
  // nicer-looking element. A ref is how you reach a DOM node imperatively.
  const inputRef = useRef<HTMLInputElement>(null)

  async function upload(file: File) {
    setError(null)

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Only PDF files are accepted.')
      return
    }
    if (file.size > MAX_BYTES) {
      setError(`"${file.name}" is larger than 25 MB.`)
      return
    }

    setUploading(true)
    try {
      await papersApi.upload(file)
      // Tell the dashboard to refetch. The new paper arrives as "pending" and
      // the dashboard's polling takes over from there.
      onUploaded()
    } catch (err) {
      setError(errorMessage(err, 'Upload failed.'))
    } finally {
      setUploading(false)
      // Clear the input so picking the same file twice in a row still fires
      // onChange -- otherwise the value is unchanged and the event never fires.
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  async function fromUrl(event: FormEvent) {
    event.preventDefault()
    const link = url.trim()
    if (!link) return
    setError(null)
    setFetching(true)
    try {
      await papersApi.fromUrl(link)
      setUrl('')
      onUploaded()
    } catch (err) {
      setError(errorMessage(err, 'Could not fetch that link.'))
    } finally {
      setFetching(false)
    }
  }

  return (
    <div>
      <div
        onDragOver={(e) => {
          e.preventDefault() // default behaviour is to open the file, not drop it
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          const file = e.dataTransfer.files[0]
          if (file) void upload(file)
        }}
        onClick={() => inputRef.current?.click()}
        className={`cursor-pointer rounded-lg border-2 border-dashed p-8 text-center transition ${
          dragging
            ? 'border-slate-900 bg-slate-100'
            : 'border-slate-300 bg-white hover:border-slate-400'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) void upload(file)
          }}
        />

        {uploading ? (
          <p className="flex items-center justify-center gap-2 text-sm text-slate-600">
            <Spinner />
            Uploading...
          </p>
        ) : (
          <>
            <p className="text-sm font-medium text-slate-700">
              Drop a PDF here, or click to choose one
            </p>
            <p className="mt-1 text-xs text-slate-400">Research papers up to 25 MB</p>
          </>
        )}
      </div>

      {/* Or paste a link. The server fetches from arxiv.org / doi.org only --
          an allowlist, so it cannot be pointed at internal addresses -- and
          takes the title, authors and abstract from arXiv's own metadata. */}
      <form onSubmit={fromUrl} className="mt-3 flex gap-2">
        <input
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="…or paste an arXiv link, e.g. https://arxiv.org/abs/1706.03762"
          disabled={fetching || uploading}
          className="flex-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm outline-none focus:border-slate-900 disabled:bg-slate-50"
        />
        <button
          type="submit"
          disabled={fetching || uploading || !url.trim()}
          className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-800 disabled:opacity-40"
        >
          {fetching ? 'Fetching…' : 'Fetch'}
        </button>
      </form>

      {error && (
        <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>
      )}
    </div>
  )
}
