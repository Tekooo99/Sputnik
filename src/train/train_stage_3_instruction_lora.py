import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, PeftModel
from dataclasses import dataclass, field
import os
from datasets import load_dataset

@dataclass
class ModelArguments:
    base_model_path: str = field(default="Qwen/Qwen-VL-Chat")
    lora_adapter_path: str = field(default="./checkpoints/stage2_pretrain_lora")

@dataclass
class DataArguments:
    data_path: str = field(default="./data/stage3_instruction.jsonl")
    image_folder: str = field(default="./images")

@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(default="./checkpoints/stage3_instruction_lora")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=16)
    learning_rate: float = field(default=1e-5)
    weight_decay: float = field(default=0.0)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=5)
    save_strategy: str = field(default="epoch")
    bf16: bool = field(default=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False)
    fp16: bool = field(default=False if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else True)
    report_to: str = field(default="none")

class InstructionDataCollator:
    def __init__(self, tokenizer, image_folder):
        self.tokenizer = tokenizer
        self.image_folder = image_folder

    def __call__(self, features):
        # Qwen-VL Chat的模板是 '<img>path</img>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>'
        prompts = []
        for f in features:
            image_path = os.path.join(self.image_folder, f['image'])
            human_prompt = f['conversations'][0]['value']
            gpt_answer = f['conversations'][1]['value']
            
            prompt = f"Picture 1:<img>{image_path}</img>\n"
            prompt += f"<|im_start|>user\n{human_prompt}<|im_end|>\n<|im_start|>assistant\n{gpt_answer}"
            prompts.append(prompt)
            
        inputs = self.tokenizer(prompts, return_tensors='pt', padding='longest', truncation=True, max_length=2048)
        labels = inputs.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {'input_ids': inputs.input_ids, 'attention_mask': inputs.attention_mask, 'labels': labels}

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 加载基础模型和Tokenizer
    # 注意：Tokenizer从阶段二的checkpoint加载，因为它包含了special tokens
    tokenizer = AutoTokenizer.from_pretrained(model_args.lora_adapter_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.base_model_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        device_map="auto"
    )
    # 确保embedding大小匹配
    model.resize_token_embeddings(len(tokenizer))
    
    # 加载并合并阶段二训练好的LoRA适配器
    print(f"Loading LoRA adapter from {model_args.lora_adapter_path}")
    model = PeftModel.from_pretrained(model, model_args.lora_adapter_path)
    model.gradient_checkpointing_enable()

    # 打印可训练参数，确认LoRA模块已加载
    model.print_trainable_parameters()

    # 加载数据集
    dataset = load_dataset("json", data_files=data_args.data_path, split="train")

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=InstructionDataCollator(tokenizer, data_args.image_folder)
    )

    # 开始训练
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    trainer.train(resume_from_checkpoint=last_checkpoint if last_checkpoint else None)

    # 保存最终的LoRA适配器
    print("Saving final LoRA adapter...")
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
