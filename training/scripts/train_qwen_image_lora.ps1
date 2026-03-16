param(
    [string]$DatasetBasePath = '.\training\dataset\qwen_image_lora\train',
    [string]$OutputPath = '.\training\outputs\qwen_image_lora',
    [string]$ModelId = 'Qwen/Qwen-Image',
    [int]$Epochs = 10,
    [double]$LearningRate = 1e-4,
    [int]$LoraRank = 32,
    [int]$Height = 1024,
    [int]$Width = 1024,
    [int]$DatasetRepeat = 10
)

$metadata = Join-Path $DatasetBasePath 'metadata.csv'
$command = @(
    'accelerate', 'launch', 'examples/qwen_image/model_training/train.py',
    '--dataset_base_path', $DatasetBasePath,
    '--dataset_metadata_path', $metadata,
    '--height', $Height,
    '--width', $Width,
    '--dataset_repeat', $DatasetRepeat,
    '--model_id_with_origin_paths', "$ModelId`:transformer/diffusion_pytorch_model*.safetensors,$ModelId`:text_encoder/model*.safetensors,$ModelId`:vae/diffusion_pytorch_model.safetensors",
    '--learning_rate', $LearningRate,
    '--num_epochs', $Epochs,
    '--output_path', $OutputPath,
    '--lora_base_model', 'dit',
    '--lora_target_modules', 'to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1',
    '--lora_rank', $LoraRank,
    '--dataset_num_workers', '2',
    '--use_gradient_checkpointing',
    '--find_unused_parameters'
)

Write-Host 'Running:'
Write-Host ($command -join ' ')
& $command[0] $command[1..($command.Length - 1)]
