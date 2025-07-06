from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
from transformers import AutoModel, AutoTokenizer


class HFEmbedder(nn.Module):
    def __init__(self, tokenizer_path: str, text_model_path: str,  max_length: int, is_clip: bool, **hf_kwargs):
        super().__init__()
        self.is_clip = is_clip
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, max_length=max_length,  local_files_only=True)
            self.hf_module: CLIPTextModel = AutoModel.from_pretrained(text_model_path,  local_files_only=True, **hf_kwargs)
        else:
            self.tokenizer: T5Tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, max_length=max_length,  local_files_only=True)
            self.hf_module: T5EncoderModel = AutoModel.from_pretrained(text_model_path, local_files_only=True, **hf_kwargs)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key].bfloat16()
