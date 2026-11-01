import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Reranker:
    """Cross-encoder: reads query and passage together, returns relevance in 0-1."""

    def __init__(self, model_name: str, max_tokens: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.max_tokens = max_tokens

    @torch.inference_mode()
    def score(self, query: str, passages: list[str], batch_size: int = 16) -> list[float]:
        scores: list[float] = []
        for i in range(0, len(passages), batch_size):
            batch = passages[i : i + batch_size]
            inputs = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt",
            )
            logits = self.model(**inputs).logits.view(-1)
            scores += torch.sigmoid(logits).tolist()  # logits are unbounded, thresholds need 0-1
        return scores
