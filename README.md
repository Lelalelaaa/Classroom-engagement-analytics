# ReLi — Classroom Engagement Analytics

> Real-time, privacy-preserving classroom monitoring using computer vision.
> No video stored. No faces tracked. Only anonymous, aggregated insights.

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- Docker + Docker Compose
- PostgreSQL 16 (or use Docker)
- Redis 7 (or use Docker)

### 1. Clone & configure
```bash
git clone <repo-url>
cd ReLi
cp backend/.env.example backend/.env
# Edit backend/.env with your database credentials
```

### 2. Start with Docker Compose (easiest)
```bash
cd infra
docker compose up -d
```
This starts PostgreSQL, Redis, the FastAPI backend, and Next.js frontend.

### 3. Run migrations
```bash
docker compose exec backend alembic upgrade head
```

### 4. Open the app
- **Landing page:** http://localhost:3000
- **Teacher Dashboard:** http://localhost:3000/teacher
- **Admin Overview:** http://localhost:3000/admin
- **API Docs:** http://localhost:8000/docs

---

## Development (without Docker)

### Backend
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## Project Structure

```
ReLi/
├── backend/          # FastAPI + AI Pipeline
│   ├── app/
│   │   ├── pipeline/ # MediaPipe, Gaze, Yawn, Emotion, Engagement
│   │   ├── api/      # REST routes + WebSocket endpoints
│   │   ├── services/ # Redis, Alerts, Metric Writer
│   │   └── db/       # SQLModel models + repositories
│   └── tests/        # Pytest unit tests
├── frontend/         # Next.js 14 dashboard
│   └── src/
│       ├── app/      # Teacher + Admin pages
│       ├── components/
│       │   ├── charts/   # Recharts components
│       │   └── ui/       # KPI cards, alerts, session setup
│       ├── hooks/    # WebSocket + session hooks
│       └── store/    # Zustand state
└── infra/            # Docker Compose + NGINX
```

---

## Privacy Guarantees

- Raw video frames are **never written to disk** — processed in RAM only
- **No individual student data** is stored — only anonymous class-level percentages
- Only metrics cross the privacy boundary: `{ class_engagement: 78%, yawn_rate: 5%, ... }`
- PostgreSQL only stores aggregated snapshots — no face IDs, no emotion per student

---

## Engagement Score Formula

```
E = (0.50 × Gaze) + (0.30 × Emotion_Valence) + (0.20 × Yawn_Absence)
```

| Signal | Weight | Range |
|--------|--------|-------|
| Gaze (on-screen) | 50% | 0 or 1 |
| Emotion valence | 30% | 0.1 – 1.0 |
| Yawn absence | 20% | 0 or 1 |

---

## Running Tests

```bash
cd backend
pytest tests/ -v
```

---

## Deployment

See the [full tutorial document](../ReLi_Tutorial.md) for:
- Cloud hosting architecture (Vercel + AWS)
- On-premise deployment guide
- Hardware sizing recommendations
- Privacy compliance checklist
