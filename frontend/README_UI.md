# Feedback Analytics UI (Frontend)

## Dev prerequisites

- Node.js + npm
- Backend running (FastAPI) on `http://127.0.0.1:8000`

## Install

From the repo root:

- Install frontend dependencies:
  - `cd frontend`
  - `npm install`

> Note: This UI uses Tailwind CSS v4 with the PostCSS plugin `@tailwindcss/postcss`.

## Run (recommended)

1. Start the backend (in another terminal)
2. Start the frontend dev server:

- `cd frontend`
- `npm run dev`

Then open the URL printed by Vite (typically `http://localhost:5173/`).

### If port 5173 is busy

Vite will automatically pick the next port (e.g. `5174`). Use the URL printed in the terminal.

## Tailwind / Google palette troubleshooting

If you see missing styles (e.g., button text not white, cards not white background):

1. **Restart Vite** after any Tailwind or PostCSS changes.
2. Ensure `@tailwindcss/postcss` is installed:
   - `npm ls @tailwindcss/postcss`
3. Ensure `frontend/postcss.config.js` uses:
   - `@tailwindcss/postcss`
4. Ensure `frontend/src/index.css` begins with:
   - `@import "tailwindcss";`
   - `@config "../tailwind.config.js";`

If you recently pulled changes:

- Run `npm install` again in `frontend/`.

If the browser is caching stale assets:

- Hard refresh (Ctrl+F5)
- Optionally delete `frontend/node_modules/.vite/` and restart `npm run dev`.

## API connectivity

The frontend calls the backend via a Vite proxy:

- Frontend requests `/api/...`
- Vite proxies to `http://127.0.0.1:8000`

This avoids CORS issues when developing locally.
