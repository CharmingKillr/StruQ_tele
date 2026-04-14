# modeling_role_llama.py
from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaConfig

class RoleLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig, num_roles: int = 4):
        super().__init__(config)
        self.num_roles = num_roles
        self.role_embeddings = nn.Embedding(num_roles, config.hidden_size)
        # 可选：初始化成 0，避免一开始破坏原模型分布
        nn.init.zeros_(self.role_embeddings.weight)
        self.config.num_roles = num_roles  # 方便保存 / 重新加载

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, num_roles: int = 4, **kwargs):
        # 先用父类加载，再注入 role embedding
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        config = model.config
        config.num_roles = num_roles
        # 把父类对象“升级”为 RoleLlamaForCausalLM
        role_model = cls(config, num_roles=num_roles)
        role_model.load_state_dict(model.state_dict(), strict=False)
        return role_model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        role_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # 默认 role=0（普通 token）
        if role_ids is None:
            role_ids = torch.zeros_like(input_ids)

        # 原始 token embedding
        inputs_embeds = self.model.embed_tokens(input_ids)
        # role embedding
        role_embeds = self.role_embeddings(role_ids)
        inputs_embeds = inputs_embeds + role_embeds

        # 把 inputs_embeds 喂给父类，避开再次 lookup
        return super().forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
