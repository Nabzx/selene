# 0001: Stack

## Status

Accepted

## Context

One day to build. The simulator, verifier and RL training want Python and numpy. The 3D view wants Three.js. Build tooling costs time and breaks at the worst moment.

## Decision

- `apps/sim`: Python with uv, FastAPI, numpy, rasterio for the data crop, `google-genai` for model calls. Optional stable-baselines3 for RL.
- `apps/web`: plain HTML, CSS and JavaScript, Three.js from a CDN import map, served as static files. It talks to the backend over HTTP and a WebSocket.

## Consequences

No bundler or `node_modules`. The view can be opened from any static server. Anything that needs a build step is out of scope.
