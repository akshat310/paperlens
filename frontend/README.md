# PaperLens — frontend

React + TypeScript + Tailwind, built with Vite. See the [root README](../README.md)
for the project as a whole.

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # type-check + production bundle into dist/
npx oxlint src   # lint
```

The dev server proxies `/api` to `http://localhost:8000` (see `vite.config.ts`),
so the backend must be running. Because both are same-origin from the browser's
point of view, CORS never applies and there is no API base URL to configure —
in production FastAPI serves this bundle itself, so they really are one origin.

## Structure

```
src/
├── api/
│   ├── client.ts        axios instance, auth interceptors, endpoint wrappers
│   └── types.ts         TypeScript mirrors of backend/app/schemas.py
├── auth/
│   └── AuthContext.tsx  the only React Context in the app
├── components/
│   ├── Layout.tsx           header shell
│   ├── ProtectedRoute.tsx   redirects logged-out users
│   ├── UploadCard.tsx       drag-and-drop PDF upload
│   ├── StatusBadge.tsx      pending / processing / ready / failed
│   ├── MessageBubble.tsx    renders [n] markers as clickable citation chips
│   ├── SummaryPanel.tsx     on-demand structured summary
│   └── Spinner.tsx
└── pages/
    ├── LoginPage.tsx
    ├── RegisterPage.tsx
    ├── DashboardPage.tsx    paper list + upload, polls while ingesting
    └── ChatPage.tsx         conversation + citations
```

## Two things worth reading first

**`MessageBubble.tsx`** is where citations become clickable. It splits the answer
on `/\[(\d+)\]/` — a capturing group in `split` keeps the captured text, so odd
indices in the resulting array are marker numbers and even indices are prose.
This works because the backend renumbers markers so that `[n]` is always
`citations[n - 1]`; without that guarantee the component would need the original
retrieval list to resolve a marker.

**`AuthContext.tsx`** is the only Context here, and deliberately so. The token
and current user are needed by the header, the route guard, and the login action
— components at unrelated depths of the tree, which is the specific problem
Context solves. Everything else uses plain `useState`, which is why there is no
Redux or React Query in this project.
