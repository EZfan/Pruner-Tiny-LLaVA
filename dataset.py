import torch
import json
import os
import random
from torch.utils.data import Dataset
from PIL import Image

from conversation import conv_templates, SeparatorStyle


class COCOPretrainDataset(Dataset):
    def __init__(self, coco_annotation_path, img_folder, tokenizer, processor, max_samples=None):

        self.img_folder = img_folder
        self.tokenizer = tokenizer
        self.processor = processor
        self.conv_template = "chatml_direct"

        print(f"正在加载 COCO 标注: {coco_annotation_path} ...")
        with open(coco_annotation_path, 'r') as f:
            coco_data = json.load(f)

        self.id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        self.data_items = []
        annotations = coco_data['annotations']

        if max_samples:
            random.shuffle(annotations)
            annotations = annotations[:max_samples]

        for ann in annotations:
            img_id = ann['image_id']
            caption = ann['caption']
            filename = self.id_to_filename.get(img_id)
            if filename and os.path.exists(os.path.join(img_folder, filename)):
                self.data_items.append({
                    "file_name": filename,
                    "caption": caption
                })
        print(f"COCO 数据加载完毕，有效数据: {len(self.data_items)} 条")

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        item = self.data_items[idx]
        img_path = os.path.join(self.img_folder, item['file_name'])
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            return self.__getitem__((idx + 1) % len(self))
        pixel_values = self.processor(image, return_tensors='pt')['pixel_values'].squeeze(0)
        conv = conv_templates[self.conv_template].copy()
        user_input = "Describe this image."
        assistant_response = item['caption']
        conv.append_message(conv.roles[0], user_input)
        conv.append_message(conv.roles[1], assistant_response)

        prompt = conv.get_prompt()
        prompt = prompt.replace("<image>", "").replace("<image>\n", "")

        encodings = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        input_ids = encodings.input_ids.squeeze(0)
        attention_mask = encodings.attention_mask.squeeze(0)

        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values
        }


class LLaVADataset(Dataset):
    def __init__(self, json_path, img_folder, tokenizer, processor, max_samples=None):

        self.data = json.load(open(json_path, 'r'))
        self.img_folder = img_folder
        self.tokenizer = tokenizer
        self.processor = processor
        self.conv_template = "chatml_direct"
        if max_samples:
            self.data = self.data[:max_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img_name = os.path.basename(item['image'])
        if "COCO" in img_name: img_name = img_name.split('_')[-1]
        img_path = os.path.join(self.img_folder, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            return self.__getitem__((idx + 1) % len(self))

        pixel_values = self.processor(image, return_tensors='pt')['pixel_values'].squeeze(0)

        conv = conv_templates[self.conv_template].copy()

        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        source_conversations = item['conversations']

        for turn in source_conversations:
            role = turn['from']
            content = turn['value']
            content = content.replace("<image>", "").strip()
            if role == "human" or role == "user":
                conv.append_message(conv.roles[0], content)
            elif role == "gpt" or role == "assistant":
                conv.append_message(conv.roles[1], content)

        prompt = conv.get_prompt()
        prompt = prompt.replace("<image>", "").replace("<image>\n", "")

        encodings = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256
        )

        input_ids = encodings.input_ids.squeeze(0)
        attention_mask = encodings.attention_mask.squeeze(0)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x['input_ids'] for x in batch]),
        "attention_mask": torch.stack([x['attention_mask'] for x in batch]),
        "labels": torch.stack([x['labels'] for x in batch]),
        "pixel_values": torch.stack([x['pixel_values'] for x in batch])
    }
