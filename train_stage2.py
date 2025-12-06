import os
import torch
from transformers import AutoTokenizer, SiglipImageProcessor, TrainingArguments, Trainer
from model import PrunerTinyLLaVA
from dataset import LLaVADataset, collate_fn
from torch.utils.data import Subset

MODEL_ID = "local_models\qwen2.5-0.5b-instruct"
VISION_ID = "local_models\siglip-so400m"
DATA_PATH = "data/llava_instruct_150k.json"
IMG_DIR = "data/coco/train2017"
STAGE1_CKPT = "output_stage1/stage1.pt"
OUTPUT_DIR = "output_stage2_fast"


def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    processor = SiglipImageProcessor.from_pretrained(
        VISION_ID,
        size={"height": 378, "width": 378},
        do_resize=True,
        do_center_crop=False
    )

    model = PrunerTinyLLaVA(MODEL_ID, VISION_ID)

    if os.path.exists(STAGE1_CKPT):
        print(f"加载 Stage 1 权重: {STAGE1_CKPT}")
        model.load_state_dict(torch.load(STAGE1_CKPT), strict=False)
    else:
        print(f"找不到 {STAGE1_CKPT}，将跳过加载 Stage 1 权重")

    model.vision_tower.requires_grad_(False)
    model.projector.requires_grad_(True)
    model.pruner.requires_grad_(True)
    model.llm.requires_grad_(True)

    print("Stage 2...")

    full_dataset = LLaVADataset(DATA_PATH, IMG_DIR, tokenizer, processor)

    TEST_SIZE = 1000
    if len(full_dataset) > TEST_SIZE:
        print(f"仅使用前 {TEST_SIZE} 条数据 (原数据 {len(full_dataset)} 条)")
        dataset = Subset(full_dataset, range(TEST_SIZE))
    else:
        dataset = full_dataset
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-5,
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        save_safetensors=False,
        report_to="none"
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collate_fn)
    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model_fast.pt"))
    print("Stage 2训练完成！")


if __name__ == "__main__":
    train()
