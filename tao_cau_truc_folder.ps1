# Script tao cau truc folder cho project World Model (RSSM) + Actor-Critic - MetaDrive
# Cach dung: mo PowerShell trong thu muc goc project, chay: .\tao_cau_truc_folder.ps1

$folders = @(
    "configs",
    "data\raw_rollouts",
    "data\replay_buffer",
    "models",
    "envs",
    "training",
    "evaluation",
    "utils",
    "notebooks",
    "checkpoints\world_model",
    "checkpoints\actor_critic",
    "logs",
    "scripts"
)

foreach ($f in $folders) {
    New-Item -ItemType Directory -Force -Path $f | Out-Null
    Write-Host "Da tao: $f"
}

# Tao cac file __init__.py rong cho cac package Python
$initFiles = @("models\__init__.py", "envs\__init__.py", "training\__init__.py", "evaluation\__init__.py", "utils\__init__.py")
foreach ($f in $initFiles) {
    New-Item -ItemType File -Force -Path $f | Out-Null
}

# Tao file .gitignore
@"
checkpoints/
logs/
data/
__pycache__/
*.pyc
.ipynb_checkpoints/
"@ | Out-File -FilePath ".gitignore" -Encoding utf8

# Tao file requirements.txt khung san
@"
metadrive-simulator
torch
numpy
matplotlib
jupyter
wandb
pyyaml
"@ | Out-File -FilePath "requirements.txt" -Encoding utf8

Write-Host ""
Write-Host "Hoan tat! Cau truc folder da san sang." -ForegroundColor Green