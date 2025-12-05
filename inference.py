import torch
from PIL import Image
from transformers import AutoTokenizer, SiglipImageProcessor
from model import PrunerTinyLLaVA

MODEL_ID = "D:\D11PMINDER\Pruner-Tiny-llava\local_models\qwen2.5-0.5b-instruct"
VISION_ID = "D:\D11PMINDER\Pruner-Tiny-llava\local_models\siglip-so400m"
CKPT_PATH = "output_stage2_fast/final_model_fast.pt"
IMG_PATH = "data/coco/train2017/000000000034.jpg"


def run_inference():
    device = "cuda"
    model = PrunerTinyLLaVA(MODEL_ID, VISION_ID, device=device)
    model.load_state_dict(torch.load(CKPT_PATH), strict=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    processor = SiglipImageProcessor.from_pretrained(
        VISION_ID,
        size={"height": 378, "width": 378},
        do_resize=True,
        do_center_crop=False
    )
    try:
        image = Image.open(IMG_PATH).convert('RGB')
    except:
        print("找不到图片，请检查路径")
        return

    pixel_values = processor(image, return_tensors='pt')['pixel_values'].to(device)

    question = "Describe this image in detail."
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    print("正在推理...")
    with torch.no_grad():
        vision_out = model.vision_tower.vision_model(pixel_values).last_hidden_state
        pruned_feats = model.pruner(vision_out)

        print(f"原始 Vision Tokens: {vision_out.shape[1]}")
        print(f"剪枝后输入 LLM Tokens: {pruned_feats.shape[1]}")

        img_embeds = model.projector(pruned_feats.to(torch.bfloat16))
        txt_embeds = model.llm.model.embed_tokens(input_ids)
        combined_embeds = torch.cat([img_embeds, txt_embeds], dim=1)

        generated_ids = model.llm.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=100,
            do_sample=False
        )

    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("\nPrunerTinyLLaVA 回答:", output_text)


if __name__ == "__main__":
    run_inference()
