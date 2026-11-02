# Integration guide (read first)

**Status:** the Climate_Pulse repo could not be accessed from the environment this was built in
(private repo / no GitHub credentials). So this module was built and tested STANDALONE and has
NOT been checked against the repo's real framework, folder layout, naming conventions or
dependency versions. Please verify the points below.

## Assumptions to verify against the repo
1. Backend is **FastAPI** with **pydantic v2**. (If pydantic v1: `model_validator`/`field_validator`
   in `schemas.py` need small changes. If Flask/Django: only `router.py` must be rewritten; `engine.py`,
   `scoring.py`, `config.py` are framework-free apart from pydantic.)
2. API prefix convention. Current: `/api/climate-simulator`. Change `prefix=` in `router.py` if needed.
3. No name clash with Varad's modules.

## Steps
```bash
git fetch origin
git checkout backend && git pull            # read-only: see shared work
git checkout -b feature/climate-risk-adaptation-simulator   # or your existing feature branch
```
1. Copy `climate_risk_simulator/` into the backend source tree (e.g. `backend/app/routers/` or `backend/app/`).
   Fix imports if it lives under a different package path.
2. Copy `tests/` files into the repo's test directory (drop `conftest.py`'s `sys.path` hack if the repo has its own).
3. In the main app file add ONE line (do not touch other routers):
   ```python
   from climate_risk_simulator import router as climate_simulator_router
   app.include_router(climate_simulator_router)
   ```
4. `pytest`, then open `/docs` and try `POST /api/climate-simulator/simulate`.
5. Commit only to your feature branch; open a PR into `backend` (never push to `main`/`backend` directly).
