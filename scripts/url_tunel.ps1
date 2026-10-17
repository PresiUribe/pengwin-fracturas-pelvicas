# Imprime la URL pública del túnel de Cloudflare (Windows PowerShell). Espera hasta 60 s.
Set-Location (Join-Path $PSScriptRoot "..")
for ($i = 0; $i -lt 60; $i++) {
    $url = docker compose logs tunnel 2>&1 | Select-String -Pattern 'https://[-a-z0-9]+\.trycloudflare\.com' -AllMatches |
           ForEach-Object { $_.Matches.Value } | Select-Object -Last 1
    if ($url) { Write-Host "Dashboard público: $url"; Write-Host "Local: http://localhost:7860"; exit 0 }
    Start-Sleep -Seconds 1
}
Write-Host "No apareció la URL. Revisen: docker compose ps   y   docker compose logs tunnel"; exit 1
