from unsloth import FastLanguageModel
import argparse
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

def loading_model(model_name, max_seq_length, dtype, load_in_4bit):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    ) 
    return model, tokenizer

def format_chat_ml(tokenizer, dataset):
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",  
        mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
        map_eos_token=True,  
    )

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset

def get_peft_model(model, r, lora_alpha):
    peft_model = FastLanguageModel.get_peft_model(
        model,
        r=r, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
        lora_alpha=lora_alpha,
        lora_dropout=0, 
        bias="none",    
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,  
        loftq_config=None, 
    )
    return peft_model

def training(model, tokenizer, dataset, max_seq_length, dataset_num_proc, save_total_limit, output_dir, args):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=dataset_num_proc,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            hub_token = args.token,
            push_to_hub=True,
            hub_strategy="checkpoint",
            push_to_hub_model_id=args.push_to_hub_model,
            learning_rate=args.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=args.logging_steps,
            save_total_limit=save_total_limit,
            save_steps=args.save_steps,
            optim=args.optim,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            seed=args.seed,
            output_dir=output_dir,
        ),
    )
    trainer_stats = trainer.train()

def main(args):
    model, tokenizer = loading_model(args.modelname, args.max_seq_length, args.dtype, args.load_in_4bit)
    model = get_peft_model(model,args.r, args.lora_alpha)
    if args.dataset:
        dataset = load_dataset(args.dataset, split="train")
        dataset = dataset.select(range(args.range))
        dataset = dataset.shuffle(seed=76)
        dataset = format_chat_ml(tokenizer, dataset)
        training(model, tokenizer, dataset, args.max_seq_length, args.dataset_num_proc, args.save_total_limit, args.output_dir, args)
    else:
        print("No dataset specified")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str, default="unsloth/mistral-7b-bnb-4bit")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--range", type=int, default=10000)
    parser.add_argument("--dataset_num_proc", type=int, default=2)
    parser.add_argument("--save_total_limit", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="output-dir")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--push_to_hub_model", type=str, default="last-hopefully")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--optim", type=str, default="adamw_8bit")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)

    args = parser.parse_args()
    main(args)