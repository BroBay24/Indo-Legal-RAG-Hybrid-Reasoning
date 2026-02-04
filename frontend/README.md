# RAG Hukum Indonesia - Frontend

Frontend chatbot modern untuk RAG Hukum Indonesia menggunakan Next.js 15 dan React 19.

## Fitur

- Chat interface modern dan responsif
- Integrasi dengan backend FastAPI RAG
- Tampilan sumber dokumen (sources) dari hasil retrieval
- Dark/Light mode support
- Streaming response support
- Mobile-friendly design

## Tech Stack

- **Framework**: Next.js 15 (App Router)
- **UI Library**: React 19
- **Styling**: Tailwind CSS
- **Components**: Radix UI primitives
- **Icons**: Lucide React

## Struktur Folder

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── api/chat/          # API route proxy ke FastAPI
│   │   ├── globals.css        # Global styles
│   │   ├── layout.tsx         # Root layout
│   │   └── page.tsx           # Homepage
│   ├── components/
│   │   ├── chat/              # Chat components
│   │   │   ├── chat-interface.tsx
│   │   │   └── sources-panel.tsx
│   │   └── ui/                # Reusable UI components
│   │       ├── button.tsx
│   │       ├── card.tsx
│   │       ├── textarea.tsx
│   │       └── ...
│   └── lib/
│       ├── types.ts           # TypeScript types
│       └── utils.ts           # Utility functions
├── .env.local                 # Environment variables
├── package.json
├── tailwind.config.ts
└── tsconfig.json
```

## Instalasi

1. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Setup environment:**
   Edit `.env.local`:
   ```env
   NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
   ```

3. **Jalankan development server:**
   ```bash
   npm run dev
   ```

4. **Buka browser:**
   ```
   http://localhost:3000
   ```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_BACKEND_URL` | URL backend FastAPI | `http://localhost:8000` |

## Penggunaan dengan Backend

Pastikan backend FastAPI sudah berjalan:
```bash
cd ../backend
python main.py
```

Frontend akan mengirim request ke endpoint `/chat` di backend.

## Build untuk Production

```bash
npm run build
npm run start
```

## Deployment

### Vercel (Recommended)
```bash
npm i -g vercel
vercel
```

### Docker
```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## API Integration

Frontend berkomunikasi dengan backend melalui:

- **Endpoint**: `POST /chat`
- **Request Body**:
  ```json
  {
    "pertanyaan": "string",
    "top_k": 5,
    "max_tokens": 800,
    "temperature": 0.7,
    "include_context": false
  }
  ```
- **Response**:
  ```json
  {
    "jawaban": "string",
    "sumber": [...],
    "konteks": "string | null",
    "pertanyaan": "string"
  }
  ```
