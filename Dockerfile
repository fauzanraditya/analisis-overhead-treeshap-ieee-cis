# Copy only installed Python dependencies, not the earlier ULB application.
# On a new computer build this dependency image with Dockerfile.runtime first.
FROM fraud-benchmark:primary-20261016 AS installed
FROM python:3.12.15-slim
COPY --from=installed /usr/local /usr/local
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt models.json protocol.json Dockerfile Dockerfile.runtime run-docker.ps1 ./
COPY models ./models
COPY resources ./resources
COPY scripts ./scripts
CMD ["python", "scripts/verify_artifact.py"]
