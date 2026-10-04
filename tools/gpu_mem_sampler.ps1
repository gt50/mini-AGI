# Usage: pwsh -File tools/gpu_mem_sampler.ps1 -ProcessId <train.py pid> [-Out runs/gpu-mem.csv]
#
# Samples one process's GPU memory from Windows' own counters every
# $Every seconds until it exits, appending CSV rows to $Out.
#   dedicated_gb - in the GPU's dedicated (VGM) memory
#   shared_gb    - spilled to shared system memory, which is far slower
param([int]$ProcessId, [string]$Out = "runs/gpu-mem.csv", [int]$Every = 30)

"time,dedicated_gb,shared_gb,committed_gb" | Out-File -FilePath $Out -Encoding utf8
while (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
    $pat = "pid_$($ProcessId)_"
    $s = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage',
                      '\GPU Process Memory(*)\Shared Usage',
                      '\GPU Process Memory(*)\Total Committed' `
          -ErrorAction SilentlyContinue).CounterSamples |
         Where-Object { $_.InstanceName -like "$pat*" }
    $sum = { param($kind) ($s | Where-Object { $_.Path -like "*\$kind" } |
                           Measure-Object CookedValue -Sum).Sum / 1GB }
    $row = "{0},{1:N2},{2:N2},{3:N2}" -f (Get-Date -Format s),
        (& $sum 'dedicated usage'), (& $sum 'shared usage'), (& $sum 'total committed')
    $row | Out-File -FilePath $Out -Append -Encoding utf8
    Start-Sleep -Seconds $Every
}
