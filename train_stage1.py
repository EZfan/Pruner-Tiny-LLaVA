import os
import torch
from transformers import AutoTokenizer, SiglipImageProcessor, TrainingArguments, Trainer
from model import PrunerTinyLLaVA
from dataset import LLaVADataset, collate_fn, COCOPretrainDataset


MODEL_ID = "D:\D11PMINDER\Pruner-Tiny-llava\local_models\qwen2.5-0.5b-instruct"
VISION_ID = "D:\D11PMINDER\Pruner-Tiny-llava\local_models\siglip-so400m"
DATA_PATH = "data/captions_train2017.json"
IMG_DIR = "data/coco/train2017"
OUTPUT_DIR = "output_stage1"


def train():
    # 1. 准备组件
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    processor = SiglipImageProcessor.from_pretrained(
        VISION_ID,
        size={"height": 378, "width": 378},
        do_resize=True,
        do_center_crop=False
    )

    model = PrunerTinyLLaVA(MODEL_ID, VISION_ID)

    model.vision_tower.requires_grad_(False)
    model.llm.requires_grad_(False)
    model.projector.requires_grad_(True)
    model.pruner.requires_grad_(True)

    print(f"Stage 1参数设置完成,开始训练 Projector + Pruner...")

    dataset = COCOPretrainDataset(DATA_PATH, IMG_DIR, tokenizer, processor, max_samples=5000)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=1e-3,
        bf16=True,
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collate_fn)
    trainer.train()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "stage1.pt"))
    print("Stage 1训练完成！")


if __name__ == "__main__":
    train()
