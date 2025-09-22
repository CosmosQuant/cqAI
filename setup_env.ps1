# cqAI 自动设置虚拟环境脚本
Write-Host "Setting up cqAI (Crypto Quantitative AI) environment..." -ForegroundColor Green

# 检查并激活虚拟环境
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    .\venv\Scripts\Activate.ps1
    Write-Host "cqAI Virtual environment activated!" -ForegroundColor Green
    Write-Host "Python path: $(python -c 'import sys; print(sys.executable)')" -ForegroundColor Cyan
} else {
    Write-Host "cqAI Virtual environment not found!" -ForegroundColor Red
}

# 显示项目信息
Write-Host "`nProject: cqAI - Crypto Quantitative AI" -ForegroundColor Yellow
Write-Host "Location: $(Get-Location)" -ForegroundColor Yellow
Write-Host "cqAI Ready to work! 🚀🤖" -ForegroundColor Green
