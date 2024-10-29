$OutputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$openPoseDemoPath = "bin\OpenPoseDemo.exe"


$dataDir = "C:\project\data"

cd openpose\

$files = Get-ChildItem -Path $dataDir -Filter "*.avi"

foreach ($file in $files) {

    $baseName = $file.BaseName
    

    $targetDir = Join-Path -Path $dataDir -ChildPath $baseName
    if (-not (Test-Path $targetDir)) {
        New-Item -ItemType Directory -Path $targetDir | Out-Null
    }
    

    $openPoseCommand = "$openPoseDemoPath --video `"$($file.FullName)`" --write_json `"$targetDir`" --display 0 --render_pose 0"
    

    Invoke-Expression $openPoseCommand
    
    Start-Sleep -Seconds 60
}

Write-Host "all ok"