# MiroFish Setup Guide

This file is a practical setup guide for the current state of this fork.
It is based on the main README, but focuses on the startup paths that are
working in this repository today.

## What Changed

MiroFish now supports two graph backends:

- `zep_cloud`: hosted Zep Cloud
- `graphiti_local`: local Graphiti + Neo4j

The local backend keeps all project graphs inside one Neo4j database and
isolates them with Graphiti `group_id`.

## Recommended Paths

Choose one of these:

- Docker: run frontend, backend, and Neo4j with `docker compose`
- Local development: run frontend/backend locally and Neo4j in Docker

## Prerequisites

For Docker:

- Docker Desktop or Docker Engine with Compose support

For local development:

- Node.js 18+
- Python 3.11 or 3.12
- `uv`
- Docker, if you want the local Neo4j service

## Environment File

Create the env file from the example:

```bash
cp .env.example .env
```

## Option 1: Docker Startup

This is the easiest way to run the full stack.

### 1. Configure `.env`

For the local Graphiti backend, a minimal working config looks like this:

```env
GRAPH_BACKEND=graphiti_local

LLM_API_KEY=your_llm_api_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o-mini

NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=mirofish-local-password
NEO4J_DATABASE=neo4j
```

Notes:

- `GRAPHITI_LLM_*`, `GRAPHITI_EMBEDDER_*`, and `GRAPHITI_RERANKER_*` are optional
- if they are omitted, the backend falls back to the main `LLM_*` settings

If you want to keep using hosted Zep Cloud instead, use:

```env
GRAPH_BACKEND=zep_cloud

LLM_API_KEY=your_llm_api_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o-mini

ZEP_API_KEY=your_zep_api_key
```

### 2. Build and start

```bash
docker compose up -d --build
```

### 3. Check status

```bash
docker compose ps
docker compose logs -f
curl http://localhost:5001/health
```

When healthy, the backend should answer with a payload that includes:

```json
{
  "status": "ok",
  "service": "MiroFish Backend",
  "graph_backend": "graphiti_local"
}
```

### 4. Open the app

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:5001`
- Neo4j Browser: `http://localhost:7474`

### Useful Docker commands

Stop the stack:

```bash
docker compose down
```

Stop and remove volumes too:

```bash
docker compose down -v
```

Rebuild after dependency or Dockerfile changes:

```bash
docker compose up -d --build
```

Restart only Neo4j:

```bash
docker compose up -d neo4j
```

## Option 2: Local Development Startup

Use this when you want hot reload or easier debugging.

### 1. Configure `.env`

For local Graphiti, use:

```env
GRAPH_BACKEND=graphiti_local

LLM_API_KEY=your_llm_api_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o-mini

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=mirofish-local-password
NEO4J_DATABASE=neo4j
```

### 2. Install dependencies

```bash
npm run setup:all
```

This does all of the following:

- installs root Node dependencies
- installs frontend dependencies
- creates and syncs the backend `uv` environment
- installs `graphiti-core==0.28.2` separately into the backend venv

### 3. Start Neo4j

```bash
docker compose up -d neo4j
```

### 4. Start frontend and backend

```bash
npm run dev
```

Or individually:

```bash
npm run backend
npm run frontend
```

## Current Neo4j Note

The local compose stack uses:

- `neo4j:5.26.22-enterprise`

This repo keeps the enterprise image as the default compose target because
existing local volumes may already use Neo4j block format. The application
logic itself is using a single Neo4j database plus Graphiti `group_id`
isolation, not one database per project.

## Troubleshooting

### Backend health is failing

Check:

- `LLM_API_KEY` is set
- `GRAPH_BACKEND` is correct
- if `GRAPH_BACKEND=graphiti_local`, `NEO4J_PASSWORD` is set
- Neo4j is running

### Docker app builds but does not start correctly

Watch logs:

```bash
docker compose logs -f mirofish neo4j
```

### Neo4j starts but the backend cannot connect

For Docker:

- use `NEO4J_URI=bolt://neo4j:7687`

For local development:

- use `NEO4J_URI=bolt://localhost:7687`

### You are on x86_64 and Docker build fails

The app service currently pins:

- `platform: linux/arm64`

in `docker-compose.yml`.

If your machine is not ARM64, remove or change that line before building.

## Fast Start

If you just want the shortest path for local Graphiti in Docker:

```bash
cp .env.example .env
```

Put this in `.env`:

```env
GRAPH_BACKEND=graphiti_local
LLM_API_KEY=your_llm_api_key
NEO4J_PASSWORD=mirofish-local-password
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_DATABASE=neo4j
```

Then run:

```bash
docker compose up -d --build
curl http://localhost:5001/health
```
