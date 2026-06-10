# Build pre-indexed Docker images and push to Docker Hub.
#
# Prerequisites:
#   - Docker Desktop running
#   - Data files present in data/  (see README — 5 required files)
#   - Python + dependencies installed in your venv
#   - Logged into Docker Hub: docker login
#
# Usage (run from project root):
#   .\.venv\Scripts\Activate.ps1
#   cd "path\to\project"
#   docker\build-and-push.ps1 -DockerUser yourusername

param(
    [Parameter(Mandatory=$true)]
    [string]$DockerUser
)

$AppImage   = "${DockerUser}/recipe-recommender-app:latest"
$EsImage    = "${DockerUser}/recipe-recommender-es:latest"
$EsName     = "recipe_es_build"
$EsVolume   = "recipe_esdata_build"
$SnapshotDir = "docker\es-snapshot"

Write-Host ""
Write-Host "Building  $AppImage"
Write-Host "          $EsImage"
Write-Host ""

# ── 1. Clean up any leftover state ────────────────────────────────────────────
# Pipe through Out-Null — these are expected to fail on a clean run (no-op)
docker rm -f $EsName 2>&1 | Out-Null
docker volume rm $EsVolume 2>&1 | Out-Null
if (Test-Path $SnapshotDir) { Remove-Item -Recurse -Force $SnapshotDir }
New-Item -ItemType Directory -Force $SnapshotDir | Out-Null

# ── 2. Start Elasticsearch with a named volume ────────────────────────────────
Write-Host "==> Starting Elasticsearch..."
docker run -d --name $EsName `
    -e "discovery.type=single-node" `
    -e "xpack.security.enabled=false" `
    -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" `
    -p 9200:9200 `
    -v "${EsVolume}:/usr/share/elasticsearch/data" `
    docker.elastic.co/elasticsearch/elasticsearch:8.13.4

Write-Host "==> Waiting for Elasticsearch to become healthy..."
$maxRetries = 60
for ($i = 0; $i -lt $maxRetries; $i++) {
    Start-Sleep -Seconds 3
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:9200/_cluster/health" -UseBasicParsing -ErrorAction Stop
        if ($r.StatusCode -eq 200) { break }
    } catch {}
    if ($i -eq ($maxRetries - 1)) { throw "Elasticsearch failed to start after $($maxRetries * 3)s." }
}
Write-Host "==> Elasticsearch ready."

# ── 3. Index recipes ──────────────────────────────────────────────────────────
Write-Host "==> Indexing recipes (may take a few minutes)..."
python main.py index
Write-Host "==> Indexing complete."

# ── 4. Stop ES cleanly so node.lock is released ───────────────────────────────
Write-Host "==> Stopping Elasticsearch..."
docker stop $EsName | Out-Null
docker rm $EsName | Out-Null

# ── 5. Export the ES volume to a tar archive ──────────────────────────────────
Write-Host "==> Exporting index data from Docker volume..."
$AbsSnapshot = (Resolve-Path $SnapshotDir).Path
docker run --rm `
    -v "${EsVolume}:/source:ro" `
    -v "${AbsSnapshot}:/backup" `
    alpine tar czf /backup/esdata.tar.gz -C /source .
Write-Host "==> Snapshot saved to $SnapshotDir\esdata.tar.gz"

# ── 6. Build and push the app image (data/ is copied in by Dockerfile) ────────
Write-Host "==> Building app image (data files included)..."
docker build -f docker/Dockerfile -t $AppImage .
Write-Host "==> Pushing app image..."
docker push $AppImage

# ── 7. Build and push the pre-indexed ES image ────────────────────────────────
Write-Host "==> Building pre-indexed Elasticsearch image..."
docker build -f docker/Dockerfile.es -t $EsImage docker/
Write-Host "==> Pushing Elasticsearch image..."
docker push $EsImage

# ── 8. Write the reviewer compose file ────────────────────────────────────────
Write-Host "==> Writing docker/docker-compose.prebuilt.yml..."
@"
# Reviewer quick start — no data files or Python installation needed.
#
#   docker compose -f docker/docker-compose.prebuilt.yml up
#
# Then open http://localhost:8501
# First run: ES takes ~30s to start. No indexing step — index is pre-loaded.

services:
  elasticsearch:
    image: $EsImage
    container_name: recipe_es
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - ES_JAVA_OPTS=-Xms1g -Xmx1g
    ports:
      - "9200:9200"
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:9200/_cluster/health || exit 1"]
      interval: 15s
      timeout: 10s
      retries: 20
      start_period: 30s

  app:
    image: $AppImage
    container_name: recipe_app
    depends_on:
      elasticsearch:
        condition: service_healthy
    ports:
      - "8501:8501"
    environment:
      - ES_HOST=http://elasticsearch:9200
      - ES_INDEX=recipes
    restart: unless-stopped
"@ | Out-File -FilePath "docker/docker-compose.prebuilt.yml" -Encoding utf8

# ── 9. Cleanup build artifacts ────────────────────────────────────────────────
Write-Host "==> Cleaning up..."
docker volume rm $EsVolume 2>&1 | Out-Null
Remove-Item -Recurse -Force $SnapshotDir

Write-Host ""
Write-Host "Done."
Write-Host ""
Write-Host "Share docker/docker-compose.prebuilt.yml with your reviewer."
Write-Host "They only need Docker Desktop installed, then run:"
Write-Host ""
Write-Host "    docker compose -f docker/docker-compose.prebuilt.yml up"
Write-Host ""
Write-Host "App will be available at http://localhost:8501"
