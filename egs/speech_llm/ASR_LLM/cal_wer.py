import numpy as np
from tqdm import tqdm
import sys

def calculate_wer_overall(references, hypotheses):
    """
    전체 WER을 계산.
    """
    total_subs, total_dels, total_ins = 0, 0, 0
    total_words = 0

    for ref, hyp in tqdm(zip(references, hypotheses)):
        ref_words = ref.split()
        hyp_words = hyp.split()

        # Levenshtein distance로 오류 카운트
        dp = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)
        for i in range(len(ref_words) + 1):
            dp[i][0] = i
        for j in range(len(hyp_words) + 1):
            dp[0][j] = j
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

        # 오류 집계
        i, j = len(ref_words), len(hyp_words)
        subs, dels, ins = 0, 0, 0
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                subs += 1
                i -= 1
                j -= 1
            elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
                ins += 1
                j -= 1
            else:
                dels += 1
                i -= 1

        total_subs += subs
        total_dels += dels
        total_ins += ins
        total_words += len(ref_words)

    total_errors = total_subs + total_dels + total_ins
    return total_errors, total_subs, total_dels, total_ins, total_errors / total_words

def calculate_wer_from_file(file_path):
    """
    파일에서 참조(ref)와 가설(hyp)을 읽어 WER을 계산.
    파일 형식은 다음과 같아야 함:
    ref: <reference sentence>
    hyp: <hypothesis sentence>
    """
    references = []
    hypotheses = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("ref:"):
                references.append(line.replace("ref:", "").strip())
            elif line.startswith("hyp:"):
                hypotheses.append(line.replace("hyp:", "").strip())

    if len(references) != len(hypotheses):
        raise ValueError("Mismatch between number of references and hypotheses.")

    return calculate_wer_overall(references, hypotheses)

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        tot_err, tot_sub, tot_del, tot_ins, wer = calculate_wer_from_file(file_path)
        print("total_error: ", tot_err)
        print("total_subs: ", tot_sub)
        print("total_dels: ", tot_del)
        print("total_ins: ", tot_ins)
        print(f"Overall WER: {wer:.2%}")
    except Exception as e:
        print(f"Error: {e}")
