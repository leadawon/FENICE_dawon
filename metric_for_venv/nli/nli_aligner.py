from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from metric.utils.utils import chunks


class NLIAligner:
    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        batch_size: int = 32,
        device: str = "cuda",
        max_length: int = 512,
        **kwargs,
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            device
        )
        self.batch_size = batch_size
        self.max_length = max_length

    def score(self, premises, hypothesis, lower=False, disable_prog_bar: bool = False):
        if isinstance(hypothesis, str):
            hypothesis = [hypothesis] * len(premises)
        if lower:
            premises = [p.lower() for p in premises]
            hypothesis = [h.lower() for h in hypothesis]
        prem_hyp_pairs = list(zip(premises, hypothesis))
        return self.process_batch(prem_hyp_pairs, disable_prog_bar=disable_prog_bar)

    def process_batch(self, prem_hyp_pairs, disable_prog_bar=False):
        batches = list(chunks(prem_hyp_pairs, self.batch_size))
        all_entailment_probs = []
        all_contradiction_probs = []
        all_neutral_probs = []
        for batch in tqdm(
            batches,
            desc="Computing alignments...",
            total=len(batches),
            disable=disable_prog_bar,
        ):
            entailment_dists = self.score_sample(batch)
            all_entailment_probs.extend([d[0] for d in entailment_dists])
            all_neutral_probs.extend([d[1] for d in entailment_dists])
            all_contradiction_probs.extend(d[2] for d in entailment_dists)
        return all_entailment_probs, all_contradiction_probs, all_neutral_probs

    # def score_sample(self, batch):
    #     tokenized_input_seq_pairs = self.tokenizer.batch_encode_plus(
    #         batch,
    #         return_token_type_ids=True,
    #         padding=True,
    #         return_tensors="pt",
    #         truncation=True,
    #         max_length=self.max_length,
    #     ).to(self.device)
    #     input_ids = tokenized_input_seq_pairs["input_ids"]
    #     token_type_ids = tokenized_input_seq_pairs["token_type_ids"]
    #     attention_mask = tokenized_input_seq_pairs["attention_mask"]
    #     with torch.no_grad():
    #         outputs = self.model(
    #             input_ids,
    #             attention_mask=attention_mask,
    #             token_type_ids=token_type_ids,
    #             labels=None,
    #         )

    #     entailment_dists = torch.softmax(outputs.logits, dim=1)
    #     return entailment_dists
    def score_sample(self, batch):
        tokenized_input_seq_pairs = self.tokenizer.batch_encode_plus(
            batch,
            return_token_type_ids=True,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        input_ids = tokenized_input_seq_pairs["input_ids"]
        token_type_ids = tokenized_input_seq_pairs["token_type_ids"]
        attention_mask = tokenized_input_seq_pairs["attention_mask"]

        with torch.no_grad():
            batch_size = input_ids.size(0)

            # ✅ Dynamic split size 결정
            if batch_size <= 2:
                split = 1
            elif batch_size <= 4:
                split = 2
            elif batch_size <= 8:
                split = 4
            elif batch_size <= 16:
                split = 8
            elif batch_size <= 32:
                split = 16
            else:
                split = 32  # or more depending on GPU
            
            chunk_size = (batch_size + split - 1) // split  # 올림 나누기
            print(chunk_size)
            logits_list = []

            for i in range(0, batch_size, chunk_size):
                input_chunk = input_ids[i:i+chunk_size]
                attention_chunk = attention_mask[i:i+chunk_size]
                token_type_chunk = token_type_ids[i:i+chunk_size]

                outputs = self.model(
                    input_chunk,
                    attention_mask=attention_chunk,
                    token_type_ids=token_type_chunk,
                    labels=None,
                )

                logits_list.append(outputs.logits)

            # 모든 조각을 concat
            logits = torch.cat(logits_list, dim=0)

        entailment_dists = torch.softmax(logits, dim=1)
        return entailment_dists
