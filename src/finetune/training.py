import yaml

# Specify the path to your YAML configuration file
config_file_path = "config.yaml"

# Load the YAML file
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)

# Access individual configuration parameters
# Access all configuration parameters
max_seq_length = config.get('max_seq_length')
dtype = config.get('dtype')
load_in_4bit = config.get('load_in_4bit')
model_name = config.get('model_name')
target_modules = config.get('target_modules')
ra_alpha = config.get('ra_alpha')
lora_dropout = config.get('lora_dropout')
bias = config.get('bias')
use_gradient_checkpointing = config.get('use_gradient_checkpointing')
random_state = config.get('random_state')
use_rslora = config.get('use_rslora')
loftq_config = config.get('loftq_config')
dataset_text_field = config.get('dataset_text_field')
dataset_num_proc = config.get('dataset_num_proc')  
packing = config.get('packing')
per_device_train_batch_size = config.get('per_device_train_batch_size')
gradient_accumulation_steps = config.get('gradient_accumulation_steps')
warmup_steps = config.get('warmup_steps')
hub_strategy = config.get('hub_strategy')
num_train_epochs = config.get('num_train_epochs')
push_to_hub = config.get('push_to_hub')
push_to_hub_model_id = config.get('push_to_hub_model_id')
learning_rate = config.get('learning_rate')
resume_from_checkpoint = config.get('resume_from_checkpoint')
fp16 = config.get('fp16')
bf16 = config.get('bf16')
logging_steps = config.get('logging_steps')
optim = config.get('optim')
weight_decay = config.get('weight_decay')
save_total_limit = config.get('save_total_limit')
save_steps = config.get('save_steps')
lr_scheduler_type = config.get('lr_scheduler_type')
seed = config.get('seed')
