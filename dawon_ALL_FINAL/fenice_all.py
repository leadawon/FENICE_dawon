import os
import argparse
parser = argparse.ArgumentParser(description="FENICE Scoring Script")

parser.add_argument("--result_file", type=str, required=True, help="Path to save the result JSON file.")
parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON file.")

parser.add_argument("--weight_rouge", type=float, default=False, help="Weight for ROUGE score.")
parser.add_argument("--weight_bertscore", type=float, default=False, help="Weight for BERTScore.")
parser.add_argument("--weight_compare_cont", type=float, default=False, help="Whether to use ECN score.")
parser.add_argument("--weight_cont", type=float, default=False, help="Whether to use ECN score.")
parser.add_argument("--weight_min", type=float, default=0.5, help="Whether to apply weight_min strategy.")
parser.add_argument("--weight_mean", type=float, default=0.5, help="Whether to apply weight_mean strategy.")
parser.add_argument("--num_of_top_k", type=int, default=1, help="Number of top-k candidates to consider.")

parser.add_argument("--cuda_device", type=str, default="0", help="CUDA device id(s) to use (e.g., '0' or '0,1').")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
import json
import time

from metric.FENICE import FENICE

def main(args):
    
    
    fenice = FENICE(use_coref = True, args = args)

    # Load input file
    with open(args.input_file, 'r', encoding="utf-8") as f:
        factcc_data = json.load(f)

    # Create or reset result file
    with open(args.result_file, 'w', encoding="utf-8") as f:
        f.write("[\n")  # Start JSON array

    # Compute scores
    for i, item in enumerate(factcc_data):
        print(f"Processing item {i + 1}/{len(factcc_data)}: {args.result_file}")
        batch = [{"document": item["document"], "summary": item["summary"]}]
        result = fenice.score_batch(batch)

        result_entry = {
            "score": result[0]["score"],
            "label": item["label"],
            "document": item["document"],
            "summary": item["summary"], # 이거 claim으로 바꾸고 dataset도 claim version으로 바꿔야함
            "cut": item["cut"],
            "model_group": item["model_group"],
        }

        # Write result entry
        with open(args.result_file, 'a', encoding="utf-8") as f:
            json.dump(result_entry, f, indent=4)
            if i < len(factcc_data) - 1:
                f.write(",\n")

    # Close JSON array
    with open(args.result_file, 'a', encoding="utf-8") as f:
        f.write("\n]")

    print(f"Results saved to {args.result_file}")

if __name__ == "__main__":
    
    main(args)
