param(
    [string]$RepoPath = '.\external\DiffSynth-Studio',
    [string]$DatasetBasePath = '.\training\dataset\qwen_image_lora\train',
    [string]$OutputRoot = '.\training\outputs\qwen_image_lora_lowvram',
    [string]$ModelId = 'Qwen/Qwen-Image',
    [string]$AccelerateConfig = '.\training\configs\accelerate_single_gpu.yaml',
    [int]$DataProcessRepeat = 1,
    [int]$TrainRepeat = 20,
    [int]$Epochs = 4,
    [double]$LearningRate = 1e-4,
    [int]$LoraRank = 32,
    [int]$MaxPixels = 1048576,
    [switch]$OnlyDataProcess,
    [switch]$OnlyTrain
)

$repoResolved = Resolve-Path $RepoPath
$datasetResolved = Resolve-Path $DatasetBasePath
$outputResolved = Join-Path (Resolve-Path '.\training').Path ('outputs\qwen_image_lora_lowvram')
New-Item -ItemType Directory -Force -Path $outputResolved | Out-Null
$cachePath = Join-Path $outputResolved 'cache'
$modelPath = Join-Path $outputResolved 'model'
$accelerateResolved = Resolve-Path $AccelerateConfig
$trainPy = Join-Path $repoResolved 'examples\qwen_image\model_training\train.py'

$commonLoraArgs = @(
    '--learning_rate', $LearningRate,
    '--num_epochs', $Epochs,
    '--remove_prefix_in_ckpt', 'pipe.dit.',
    '--lora_base_model', 'dit',
    '--lora_target_modules', 'to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1',
    '--lora_rank', $LoraRank,
    '--use_gradient_checkpointing',
    '--use_gradient_checkpointing_offload',
    '--dataset_num_workers', '2',
    '--find_unused_parameters'
)

$processCommand = @(
    'accelerate', 'launch', '--config_file', $accelerateResolved, $trainPy,
    '--dataset_base_path', $datasetResolved,
    '--dataset_metadata_path', (Join-Path $datasetResolved 'metadata.csv'),
    '--max_pixels', $MaxPixels,
    '--dataset_repeat', $DataProcessRepeat,
    '--model_id_with_origin_paths', "$ModelId`:text_encoder/model*.safetensors,$ModelId`:vae/diffusion_pytorch_model.safetensors",
    '--fp8_models', "$ModelId`:text_encoder/model*.safetensors,$ModelId`:vae/diffusion_pytorch_model.safetensors",
    '--output_path', $cachePath,
    '--task', 'sft:data_process'
) + $commonLoraArgs

$trainCommand = @(
    'accelerate', 'launch', '--config_file', $accelerateResolved, $trainPy,
    '--dataset_base_path', $cachePath,
    '--max_pixels', $MaxPixels,
    '--dataset_repeat', $TrainRepeat,
    '--model_id_with_origin_paths', "$ModelId`:transformer/diffusion_pytorch_model*.safetensors",
    '--fp8_models', "$ModelId`:transformer/diffusion_pytorch_model*.safetensors",
    '--output_path', $modelPath,
    '--task', 'sft:train'
) + $commonLoraArgs

Push-Location $repoResolved
try {
    if (-not $OnlyTrain) {
        Write-Host "Starting low-VRAM data_process stage..."
        Write-Host ($processCommand -join ' ')
        & $processCommand[0] $processCommand[1..($processCommand.Length - 1)]
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    if (-not $OnlyDataProcess) {
        Write-Host "Starting low-VRAM train stage..."
        Write-Host ($trainCommand -join ' ')
        & $trainCommand[0] $trainCommand[1..($trainCommand.Length - 1)]
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
finally {
    Pop-Location
}
