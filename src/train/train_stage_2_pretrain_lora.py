# train_stage_2_pretrain_lora.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataclasses import dataclass, field
import os
from datasets import load_dataset
from special_tokens_vocab import SpecialTokens

# --- 配置参数 ---
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="./checkpoints/stage1_alignment")

@dataclass
class DataArguments:
    data_path: str = field(default="./data/stage2_pretrain.jsonl")
    image_folder: str = field(default="./images")

@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(default="./checkpoints/stage2_pretrain_lora")
    num_train_epochs: int = field(default=5)
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.0)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=10)
    save_strategy: str = field(default="epoch")
    bf16: bool = field(default=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False)
    fp16: bool = field(default=False if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else True)
    report_to: str = field(default="none")

class PretrainDataCollator:
    def __init__(self, tokenizer, image_folder):
        self.tokenizer = tokenizer
        self.image_folder = image_folder

    def __call__(self, features):
        prompts = [f"Picture 1:<img>{os.path.join(self.image_folder, f['image'])}</img>\n{f['text']}" for f in features]
        inputs = self.tokenizer(prompts, return_tensors='pt', padding='longest', truncation=True, max_length=2048)
        labels = inputs.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {'input_ids': inputs.input_ids, 'attention_mask': inputs.attention_mask, 'labels': labels}

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 加载阶段一训练好的模型和Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        device_map="auto"
    )
    # 启用梯度检查点以节省显存
    model.gradient_checkpointing_enable()

    # 1. 配置LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=['c_attn', 'attn.c_proj', 'mlp.w1', 'mlp.w2'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 2. 应用LoRA到模型
    model = get_peft_model(model, lora_config)
    
    # 确保新词的embedding仍然是可训练的
    for name, param in model.named_parameters():
        if 'embed_tokens' in name or 'lm_head' in name:
             if param.shape[0] == len(tokenizer):
                param.requires_grad = True
    
    model.print_trainable_parameters()

    # 加载数据集
    dataset = load_dataset("json", data_files=data_args.data_path, split="train")

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=PretrainDataCollator(tokenizer, data_args.image_folder)
    )

    # 开始训练
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    trainer.train(resume_from_checkpoint=last_checkpoint if last_checkpoint else None)

    # 保存最终的LoRA适配器
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
