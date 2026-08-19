# Checkpoint Phase 4 D.2B: real production agent-system-a image. Build
# context is the REPOSITORY ROOT (root docker-compose.yml -- `context: .`,
# `dockerfile: orchestration/system_a/Dockerfile`): this service is the
# root-owned composition layer (docs/adr/0014, 0015) that imports
# services/planner-a's phase1/phase4 packages, the root providers/
# package, and this orchestration/ package itself, all through the real
# declared dependency path `pyproject.toml` establishes (Checkpoint
# D.2A) -- installed here via a real `pip install -e .`, never a
# PYTHONPATH or sys.path hack (see pyproject.toml's own docstring).
FROM python:3.12-slim

WORKDIR /app

RUN groupadd --system voyager && useradd --system --gid voyager --home /app voyager

COPY orchestration/requirements.lock ./orchestration/requirements.lock
RUN pip install --no-cache-dir -r orchestration/requirements.lock

# Only the exact directories pyproject.toml's own packages/package-dir
# mapping needs -- never the whole services/planner-a submodule (its
# other phases/tests are irrelevant to this image and would only bloat
# it), and never rag/ (this composition layer does not depend on it).
#
# Production real-mode incident P.1: `contracts/` is a runtime
# dependency of `providers/validation.py` (`_CONTRACTS_DIR` is computed
# as a sibling of the installed `providers/` package directory, via
# `__file__` -- correct, CWD-independent resolution that was never the
# bug) -- but this Dockerfile never copied it into the image, so every
# real provider call's `validate_envelope()` step failed with
# `FileNotFoundError` on first use inside this image specifically
# (never on a full host checkout, where `contracts/` already exists
# next to `providers/`). Copied here, at the same sibling path
# `_CONTRACTS_DIR` already expects.
COPY pyproject.toml ./pyproject.toml
COPY providers ./providers
COPY contracts ./contracts
COPY orchestration ./orchestration
COPY services/planner-a/phase1 ./services/planner-a/phase1
COPY services/planner-a/phase4 ./services/planner-a/phase4

RUN pip install --no-cache-dir --no-deps -e .

ENV PORT=8002
# Explicit configurable SQLite path (Checkpoint D.2A) -- pointed at a
# named volume's mount point in root docker-compose.yml so run/session/
# event history and the LangGraph checkpoint tables survive a container
# restart.
ENV VOYAGER_SYSTEM_A_DB_PATH=/data/system_a.db
# Explicit, safe default -- production/default mode. Overridden to
# "fixture" only via docker-compose.yml's fixture-mode configuration,
# never silently.
ENV VOYAGER_SYSTEM_A_MODE=real

RUN mkdir -p /data && chown -R voyager:voyager /app /data
USER voyager

EXPOSE 8002

HEALTHCHECK --interval=5s --timeout=3s --start-period=15s --retries=6 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8002\")}/health',timeout=2).status==200 else 1)"

CMD ["sh", "-c", "python -m orchestration.system_a.entrypoint"]
