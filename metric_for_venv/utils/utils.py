import hashlib
import pickle
from typing import List, Tuple, Union
import spacy
import sys
import os
from spacy.cli import download
from tqdm import tqdm

try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    os.environ["SPACY_WARNING_IGNORE"] = "true"  # Prevent interactive prompts
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def flatten(two_d_list: List[List]) -> List:
    """
    Flattens a 2D Python list into a 1D list using list comprehension.
    """
    return [item for sublist in two_d_list for item in sublist]


def chunks(lst: List, n: int) -> List[List]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def sliding_chunks(lst: List, n: int, sliding_stride: int = 1) -> List[List]:
    """Yield sliding windows of n-sized chunks from lst."""
    for i in range(len(lst) - n + 1):
        if i % sliding_stride == 0:
            yield lst[i : i + n]


def distinct(input_list: List) -> List:
    seen = set()
    return [x for x in input_list if not (x in seen or seen.add(x))]


def split_into_sentences(
    text: str, return_offsets: bool = False
) -> Union[List[Tuple[str, str, str]], List[str]]:
    # Process the text with spaCy
    doc = nlp(text)
    return get_sentences(doc, return_offsets)


def get_sentences(doc, return_offsets):
    # Initialize a list to store sentences and their offsets
    sentences_with_offsets = []
    # Iterate over the sentences in the processed text
    for sent in doc.sents:
        # Get the sentence text and character offsets
        sentence_text = sent.text
        start_offset = sent.start_char
        end_offset = sent.end_char

        # Add the sentence text and offsets to the list as a tuple
        sentences_with_offsets.append((sentence_text, start_offset, end_offset))
    if return_offsets:
        return sentences_with_offsets
    else:
        # Return only the sentences
        return [sentence[0] for sentence in sentences_with_offsets]


def split_into_sentences_batched(
    texts: List[str], return_offsets: bool = False, batch_size=32
) -> List[List[Tuple[str, str, str]]]:
    # Process the text with spaCy
    batches = list(chunks(texts, batch_size))
    sentences = []
    for b in tqdm(
        batches, total=len(batches), desc="splitting document batches into sentences"
    ):
        docs = nlp.pipe(b, disable=["attribute_ruler", "lemmatizer", "ner"])
        for doc in docs:
            doc_sentences = get_sentences(doc, return_offsets=return_offsets)
            sentences.append(doc_sentences)
    return sentences

# def dawon_split_into_sentences_batched(
#     texts: List[str], return_offsets: bool = False, batch_size=32, claims = None
# ) -> List[List[Tuple[str, str, str]]]:
#     # Process the text with spaCy
#     batches = list(chunks(texts, batch_size))
#     sentences = []
#     for b in tqdm(
#         batches, total=len(batches), desc="splitting document batches into sentences"
#     ):
#         docs = nlp.pipe(b, disable=["attribute_ruler", "lemmatizer", "ner"])
#         for doc in docs:
#             doc_sentences = get_sentences(doc, return_offsets=return_offsets)
#             sentences.append(doc_sentences)
#     ###############################################################################here dawon code




#     import copy
#     from metric.claim_extractor.claim_extractor import ClaimExtractor
#     from rouge_score import rouge_scorer
#     from sentence_transformers import SentenceTransformer, util
#     import numpy as np

#     # SBERT 모델 로드
#     sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

#     # 원본 sentences 리스트 (깊은 복사)
#     modified_sentences = copy.deepcopy(sentences)

#     # ROUGE Scorer 초기화
#     scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

#     # 선택된 sentence를 저장할 집합 (중복 제거 목적)
#     selected_sentences = set()

#     # Step 1: claims 리스트에서 가장 유사한 sentence 찾기
#     for claim in claims[0]:
#         best_score = 0
#         best_sentence_idx = None

#         for i, (sentence, _, _) in enumerate(sentences[0]):  # sentences 리스트의 첫 번째 요소 접근
#             # ROUGE-L 점수 계산
#             rouge_score = scorer.score(claim, sentence)['rougeL'].fmeasure

#             # SBERT 코사인 유사도 계산
#             claim_emb = sbert_model.encode(claim, convert_to_tensor=True)
#             sentence_emb = sbert_model.encode(sentence, convert_to_tensor=True)
#             sbert_score = util.pytorch_cos_sim(claim_emb, sentence_emb).item()

#             # 최종 점수 (ROUGE와 SBERT를 1:1 비율로 합산)
#             score = 0.5 * rouge_score + 0.5 * sbert_score

#             if score > best_score:
#                 best_score = score
#                 best_sentence_idx = i

#         if best_sentence_idx is not None:
#             selected_sentences.add(best_sentence_idx)  # 선택된 sentence 인덱스 저장

#     # ClaimExtractor 초기화
#     claim_extractor = ClaimExtractor(batch_size=128, device="cuda")

#     # Step 2: 선택된 sentence를 atomic하게 쪼개기
#     for idx in sorted(selected_sentences, reverse=True):  # 뒤에서부터 처리하여 index shift 방지
#         sentence, start_idx, end_idx = modified_sentences[0][idx]  # 원본 sentence 가져오기
        
#         # ClaimExtractor를 사용하여 atomic fact로 분할
#         atomic_facts = claim_extractor.process_batch([sentence])  # 리스트로 감싸야 함
#         atomic_facts = atomic_facts[0]  # 결과 리스트에서 첫 번째 요소 가져오기
        
#         # 분할된 atomic fact를 기존 sentences 리스트에 적용
#         new_sentences = [(fact, start_idx, end_idx) for fact in atomic_facts]  # 새로운 튜플 리스트 생성
#         modified_sentences[0] = modified_sentences[0][:idx] + new_sentences + modified_sentences[0][idx+1:]

#     # 결과 반환
#     return modified_sentences


    ###############################################################################here dawon code
    return sentences

def dawon_split_into_sentences_batched_no_discards(
    texts: List[str], return_offsets: bool = False, batch_size=32, claims=None, args=None
) -> List[List[Tuple[str, str, str]]]:
    # Process the text with spaCy
    batches = list(chunks(texts, batch_size))
    sentences = []
    for b in tqdm(
        batches, total=len(batches), desc="splitting document batches into sentences"
    ):
        docs = nlp.pipe(b, disable=["attribute_ruler", "lemmatizer", "ner"])
        for doc in docs:
            doc_sentences = get_sentences(doc, return_offsets=return_offsets)
            sentences.append(doc_sentences)

    ###############################################################################
    # Dawon code with BERTScore and no_discards
    import copy
    from metric.claim_extractor.claim_extractor import ClaimExtractor
    from rouge_score import rouge_scorer
    from bert_score import BERTScorer

    # BERTScore 모델 로드
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    # 원본 sentences 리스트 (깊은 복사)
    modified_sentences = copy.deepcopy(sentences)

    # ROUGE Scorer 초기화
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # 선택된 sentence를 저장할 집합 (중복 제거 목적)
    selected_sentences = set()

    # Step 1: claims 리스트에서 가장 유사한 sentence 찾기 (top-k 지원)
    for claim in claims[0]:
        scored_sentences = []

        for i, (sentence, _, _) in enumerate(sentences[0]):  # sentences 리스트의 첫 번째 요소 접근
            # ROUGE-L 점수 계산
            rouge_score = scorer.score(claim, sentence)['rougeL'].fmeasure

            # BERTScore 계산
            P, R, F1 = bert_scorer.score([sentence], [claim])
            bert_score = F1.item()

            # 최종 점수 (ROUGE와 BERTScore를 가중합)
            score = args.weight_rouge * rouge_score + args.weight_bertscore * bert_score

            scored_sentences.append((score, i))

        # 점수 내림차순 정렬
        scored_sentences.sort(reverse=True)

        # top-k 선택 (단, available한 sentence 개수보다 top_k가 크면 자동 조정)
        top_k = min(args.num_of_top_k, len(scored_sentences))
        for j in range(top_k):
            selected_sentences.add(scored_sentences[j][1])  # top-k sentence 인덱스 추가

    # ClaimExtractor 초기화
    claim_extractor = ClaimExtractor(batch_size=128, device="cuda")

    # Step 2: 선택된 sentence를 atomic하게 쪼개기 (no_discards 방식)
    for idx in sorted(selected_sentences, reverse=True):  # 뒤에서부터 처리하여 index shift 방지
        sentence, start_idx, end_idx = modified_sentences[0][idx]  # 원본 sentence 가져오기
        
        # ClaimExtractor를 사용하여 atomic fact로 분할
        atomic_facts = claim_extractor.process_batch([sentence])  # 리스트로 감싸야 함
        atomic_facts = atomic_facts[0]  # 결과 리스트에서 첫 번째 요소 가져오기

        # 선택된 문장 먼저 추가 (discard하지 않음)
        new_sentences = [(sentence, start_idx, end_idx)]  # 선택된 원본 문장 추가
        new_sentences += [(fact, start_idx, end_idx) for fact in atomic_facts]  # atomic fact 추가

        # 분할된 atomic fact와 선택된 문장을 기존 sentences 리스트에 적용
        modified_sentences[0] = (
            modified_sentences[0][:idx] + new_sentences + modified_sentences[0][idx+1:]
        )

    # 결과 반환
    return modified_sentences
    ###############################################################################



def split_into_paragraphs(
    sentences: List[str],
    num_sent_per_paragraph: int,
    sliding_paragraphs=True,
    sliding_stride: int = 1,
) -> List[str]:
    if len(sentences) < num_sent_per_paragraph:
        return [" ".join(sentences)]
    paragraphs = (
        list(sliding_chunks(sentences, num_sent_per_paragraph, sliding_stride))
        if sliding_paragraphs
        else list(chunks(sentences, num_sent_per_paragraph))
    )
    for i, par in enumerate(paragraphs):
        paragraphs[i] = " ".join(par)
    return paragraphs


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_pickle(path: str):
    # Open the file in binary read mode
    with open(path, "rb") as file:
        # Load the object from the file
        return pickle.load(file)


def dump_pickle(path: str, data):
    # Open the file in binary write mode
    with open(path, "wb") as file:
        # Dump the object to the file
        pickle.dump(data, file)
