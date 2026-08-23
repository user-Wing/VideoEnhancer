$ErrorActionPreference = 'Stop'

$versionRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$pluginPath = Join-Path $versionRoot 'videoenhancer.3fui.dll'
$hostBin = 'C:\Users\ARXChem\Documents\LakeUI-2\FFmpegFreeUI\FFmpegFreeUI\bin\Debug\net10.0-windows10.0.26100.0'
$cliPath = 'C:\PortableSoft\VideoEnhancer-CLI\videoenhancer.exe'

foreach ($requiredPath in @(
    $pluginPath,
    (Join-Path $hostBin 'FFmpegFreeUI.dll'),
    (Join-Path $hostBin 'LakeUI.dll'),
    $cliPath
)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required test dependency not found: $requiredPath"
    }
}

function Get-ExpectedModels([string] $backend) {
    $json = @(& $cliPath --list-interp-models -interp-backend $backend --json)
    if ($LASTEXITCODE -ne 0) {
        throw "$backend model listing failed with exit code $LASTEXITCODE"
    }
    return @($json[-1] | ConvertFrom-Json)
}

function Wait-InterpLoad($panel, $loadingField) {
    $deadline = [DateTime]::UtcNow.AddSeconds(60)
    while ($loadingField.GetValue($panel) -and [DateTime]::UtcNow -lt $deadline) {
        [System.Windows.Forms.Application]::DoEvents()
        Start-Sleep -Milliseconds 25
    }
    [System.Windows.Forms.Application]::DoEvents()
    if ($loadingField.GetValue($panel)) {
        throw 'Timed out while waiting for the interpolation model list.'
    }
}

$expectedModels = @(Get-ExpectedModels 'ncnn')
$expectedCudaModels = @(Get-ExpectedModels 'cuda')
if ($expectedModels.Count -eq 0 -or $expectedCudaModels.Count -eq 0) {
    throw 'The regression test requires at least one installed NCNN interpolation model.'
}

$configDir = Join-Path ([System.IO.Path]::GetTempPath()) ('videoenhancer-plugin-test-' + [Guid]::NewGuid().ToString('N'))
$env:VIDEOENHANCER_CONFIG_DIR = $configDir

try {
    [void][System.Reflection.Assembly]::LoadFrom((Join-Path $hostBin 'FFmpegFreeUI.dll'))
    [void][System.Reflection.Assembly]::LoadFrom((Join-Path $hostBin 'LakeUI.dll'))
    $pluginAssembly = [System.Reflection.Assembly]::LoadFrom($pluginPath)
    $configType = $pluginAssembly.GetType('videoenhancer.PluginConfig', $true)
    $panelType = $pluginAssembly.GetType('videoenhancer.PluginPanel', $true)
    $config = [Activator]::CreateInstance($configType)
    $config.ExePath = $cliPath
    $config.Enabled = $true
    $config.InterpEnabled = $true
    $config.InterpBackend = 'ncnn'
    $config.InterpModel = ''

    $constructor = $panelType.GetConstructor(@($configType, [bool]))
    $panel = $constructor.Invoke(@($config, $true))
    try {
        $null = $panel.Handle
        $startLoad = $panelType.GetMethod('StartInterpModelLoad', [System.Reflection.BindingFlags]'Instance,NonPublic')
        $refreshModels = $panelType.GetMethod('RefreshInterpModels', [System.Reflection.BindingFlags]'Instance,NonPublic')
        $loadingField = $panelType.GetField('_loadingInterpModels', [System.Reflection.BindingFlags]'Instance,NonPublic')
        $comboField = $panelType.GetField('_cmbInterp', [System.Reflection.BindingFlags]'Instance,NonPublic')
        $startLoad.Invoke($panel, @())
        Wait-InterpLoad $panel $loadingField

        $actualModels = @($comboField.GetValue($panel).Items | ForEach-Object { $_.ToString() })
        $unexpectedModels = @($actualModels | Where-Object { $_ -notin $expectedModels })
        if ($unexpectedModels.Count -gt 0) {
            throw ('NCNN model list contains models from another backend: ' + ($unexpectedModels -join ', '))
        }
        if ($config.InterpBackend -ne 'ncnn') {
            throw "Selecting NCNN was overridden to '$($config.InterpBackend)'."
        }
        if (@($actualModels).Count -ne @($expectedModels).Count) {
            throw "NCNN model list count mismatch. Expected $($expectedModels.Count), got $($actualModels.Count)."
        }

        # Reproduce a slow CUDA request finishing after the user has selected NCNN.
        $config.InterpBackend = 'cuda'
        $startLoad.Invoke($panel, @())
        $config.InterpBackend = 'ncnn'
        $refreshModels.Invoke($panel, @())
        Wait-InterpLoad $panel $loadingField

        $actualModels = @($comboField.GetValue($panel).Items | ForEach-Object { $_.ToString() })
        if ($config.InterpBackend -ne 'ncnn' -or @($actualModels).Count -ne @($expectedModels).Count -or
            @($actualModels | Where-Object { $_ -notin $expectedModels }).Count -gt 0) {
            throw "A stale CUDA request overrode the final NCNN selection. Backend=$($config.InterpBackend); Models=$($actualModels -join ', ')"
        }

        $config.InterpBackend = 'cuda'
        $refreshModels.Invoke($panel, @())
        Wait-InterpLoad $panel $loadingField
        $actualCudaModels = @($comboField.GetValue($panel).Items | ForEach-Object { $_.ToString() })
        if ($config.InterpBackend -ne 'cuda' -or @($actualCudaModels).Count -ne @($expectedCudaModels).Count -or
            @($actualCudaModels | Where-Object { $_ -notin $expectedCudaModels }).Count -gt 0) {
            throw "CUDA model list does not match the CUDA CLI catalog. Models=$($actualCudaModels -join ', ')"
        }

        Write-Host "PASS: NCNN=$($actualModels.Count), CUDA=$($actualCudaModels.Count); stale requests cannot override the selected backend."
    }
    finally {
        if ($null -ne $panel) {
            $panel.Dispose()
        }
    }
}
finally {
    Remove-Item Env:VIDEOENHANCER_CONFIG_DIR -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $configDir -Recurse -Force -ErrorAction SilentlyContinue
}
