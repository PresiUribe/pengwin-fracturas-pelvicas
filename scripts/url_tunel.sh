#!/usr/bin/env bash
# Imprime la URL pública del túnel de Cloudflare (espera hasta 60 s a que aparezca)
cd "$(dirname "$0")/.."
for i in $(seq 1 60); do
  url=$(docker compose logs tunnel 2>&1 | grep -o 'https://[-a-z0-9]*\.trycloudflare\.com' | tail -1)
  if [ -n "$url" ]; then echo "Dashboard público: $url"; echo "Local: http://localhost:7860"; exit 0; fi
  sleep 1
done
echo "No apareció la URL. Revisen: docker compose ps   y   docker compose logs tunnel"; exit 1
