/** A small CSS-only loading spinner. No library needed for a rotating circle. */
export function Spinner({ className = 'h-4 w-4' }: { className?: string }) {
  return (
    <span
      className={`inline-block animate-spin rounded-full border-2 border-current border-t-transparent ${className}`}
      // Screen readers announce this instead of trying to describe a spinning div.
      role="status"
      aria-label="Loading"
    />
  )
}
