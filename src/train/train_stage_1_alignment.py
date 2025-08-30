import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint
from dataclasses import dataclass, field
import os
from PIL import Image
from datasets import load_dataset
from special_tokens_vocab import SpecialTokens

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen-VL-Chat")

@dataclass
class DataArguments:
    data_path: str = field(default="./data/stage1_alignment.jsonl")
    image_folder: str = field(default="./images")

@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(default="./checkpoints/stage1_alignment")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.01)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=10)
    save_strategy: str = field(default="epoch")
    bf16: bool = field(default=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False)
    fp16: bool = field(default=False if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else True)
    report_to: str = field(default="none")

class AlignmentDataCollator:
    def __init__(self, tokenizer, image_folder):
        self.tokenizer = tokenizer
        self.image_folder = image_folder

    def __call__(self, features):
        image_paths = [f['image'] for f in features]
        texts = [f['text'] for f in features]

        # 注意: Qwen-VL的预处理比较特殊，它将图片路径和文本编码在一起
        # <img>image_path</img>text
        prompts = [f"Picture 1:<img>{os.path.join(self.image_folder, path)}</img>\n{text}" for path, text in zip(image_paths, texts)]
        
        # 使用Qwen的分词器进行编码
        inputs = self.tokenizer(prompts, return_tensors='pt', padding='longest', truncation=True, max_length=2048)
        
        # 创建 labels
        labels = inputs.input_ids.clone()
        # 将padding部分的label设为-100，使其在损失计算中被忽略
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'labels': labels
        }

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 加载模型和Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, pad_token='<|endoftext|>')
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        device_map="auto"
    )

    # 1. 添加并调整Special Tokens
    vocab = SpecialTokens()
    new_tokens = vocab.get_all_tokens()
    tokenizer.add_tokens(new_tokens, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    # 2. 冻结参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 解冻Visual Projector和新添加的Token Embeddings
    # Qwen-VL的visual projector名为 'visual.proj'
    for name, param in model.named_parameters():
        if 'visual.proj' in name:
            param.requires_grad = True
    
    # 解冻新词的embedding
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data
    
    input_embeddings_avg = input_embeddings[:-len(new_tokens)].mean(dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings[:-len(new_tokens)].mean(dim=0, keepdim=True)
    
    input_embeddings[-len(new_tokens):] = input_embeddings_avg
    output_embeddings[-len(new_tokens):] = output_embeddings_avg
    
    model.get_input_embeddings().weight.requires_grad = True
    model.get_output_embeddings().weight.requires_grad = True

    # 打印可训练参数
    model.print_trainable_parameters()

    # 加载数据集
    dataset = load_dataset("json", data_files=data_args.data_path, split="train")
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=AlignmentDataCollator(tokenizer, data_args.image_folder)
    )

    # 开始训练
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    trainer.train(resume_from_checkpoint=last_checkpoint if last_checkpoint else None)

    # 保存最终模型
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
