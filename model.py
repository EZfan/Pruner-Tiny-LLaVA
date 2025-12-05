import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, SiglipVisionModel


class HybridTokenPruner(nn.Module):
    def __init__(self, hidden_dim, keep_num=144, compress_num=36):
        super().__init__()
        self.keep_num = keep_num
        self.compress_num = compress_num
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        self.compressor = nn.AdaptiveAvgPool1d(compress_num)

    def forward(self, x):
        batch, seq_len, dim = x.shape

        scores = self.scorer(x).squeeze(-1)

        _, topk_indices = torch.topk(scores, self.keep_num, dim=1)
        topk_indices, _ = torch.sort(topk_indices, dim=1)

        batch_idx = torch.arange(batch, device=x.device).unsqueeze(1).expand(-1, self.keep_num)
        kept_tokens = x[batch_idx, topk_indices]

        mask = torch.ones((batch, seq_len), dtype=torch.bool, device=x.device)
        mask.scatter_(1, topk_indices, False)
        rejected = x[mask].view(batch, -1, dim)

        compressed = self.compressor(rejected.transpose(1, 2)).transpose(1, 2)

        return torch.cat([kept_tokens, compressed], dim=1)


class PrunerTinyLLaVA(nn.Module):
    def __init__(self, llm_id, vision_id, device="cuda"):
        super().__init__()
        self.device = device

        print("Loading Vision Tower...")
        self.vision_tower = SiglipVisionModel.from_pretrained(vision_id).to(device)
        if hasattr(self.vision_tower.config, 'vision_model'):
            self.vision_dim = self.vision_tower.config.vision_model.hidden_size
        else:
            self.vision_dim = self.vision_tower.config.hidden_size

        self.pruner = HybridTokenPruner(self.vision_dim, keep_num=144, compress_num=36).to(device)

        print("Loading LLM & Projector...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_id, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)
        self.llm_dim = self.llm.config.hidden_size

        self.projector = nn.Sequential(
            nn.Linear(self.vision_dim, self.llm_dim),
            nn.GELU(),
            nn.Linear(self.llm_dim, self.llm_dim)
        ).to(device).to(torch.bfloat16)

    def forward(self, input_ids, pixel_values, labels=None, attention_mask=None):

        with torch.no_grad():
            img_feats = self.vision_tower(pixel_values).last_hidden_state

        img_feats_pruned = self.pruner(img_feats)

        img_embeds = self.projector(img_feats_pruned.to(torch.bfloat16))

        txt_embeds = self.llm.model.embed_tokens(input_ids)
        combined_embeds = torch.cat([img_embeds, txt_embeds], dim=1)

        if labels is not None:
            img_labels = torch.full((img_embeds.shape[0], img_embeds.shape[1]), -100, device=self.device)
            combined_labels = torch.cat([img_labels, labels], dim=1)
        else:
            combined_labels = None

        if attention_mask is not None:
            img_mask = torch.ones((img_embeds.shape[0], img_embeds.shape[1]), device=self.device)
            combined_mask = torch.cat([img_mask, attention_mask], dim=1)
        else:
            combined_mask = None

        return self.llm(
            inputs_embeds=combined_embeds,
            labels=combined_labels,
            attention_mask=combined_mask
        )
