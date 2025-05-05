import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
from metric.FENICE import FENICE
import time
fenice = FENICE()

result_str = '../results/aggre_cnndm/fenice_SB.json'

# Load the xsumfaith.json file
with open('../data/aggre_fact_cnndm_sota.json', 'r', encoding="utf-8") as f:
    factcc_data = json.load(f)
factcc_data = factcc_data#[320:]
# JSON 파일을 새로 생성 (초기화)
with open(result_str, 'w', encoding="utf-8") as f:
    f.write("[\n")  # JSON 배열 시작
# Compute the scores for each document-summary pair
for i, item in enumerate(factcc_data):
    batch = [{"document": item["document"], "summary": item["claim"]}]
    result = fenice.score_batch(batch)

    # 저장할 데이터 구조
    result_entry = {
        "score": result[0]["score"],
        "label": item["label"],
        "document": item["document"],
        "summary": item["claim"],
        "cut": item["cut"]
    }

    # JSON 파일에 한 줄씩 추가 (마지막 요소인지 확인하여 쉼표 처리)
    with open(result_str, 'a', encoding="utf-8") as f:
        json.dump(result_entry, f, indent=4)
        if i < len(factcc_data) - 1:  # 마지막 요소가 아니면 쉼표 추가
            f.write(",\n")

# JSON 파일 닫기 (배열 닫기)
with open(result_str, 'a', encoding="utf-8") as f:
    f.write("\n]")

print("Results saved to fenice_original.json")
