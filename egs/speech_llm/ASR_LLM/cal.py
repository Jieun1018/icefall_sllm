import numpy as np
import sys

def calculate_wer_overall(reference, hypothesis):
    """
    Calculate the Word Error Rate (WER).
    :param reference: List of words (ground truth).
    :param hypothesis: List of words (predicted).
    :return: Total errors and reference length for cumulative WER calculation.
    """
    r = reference.split()
    h = hypothesis.split()
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i - 1][j] + 1,      # deletion
                         d[i][j - 1] + 1,      # insertion
                         d[i - 1][j - 1] + cost)  # substitution

            if i > 1 and j > 1 and r[i - 1] == h[j - 2] and r[i - 2] == h[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)  # transposition

    return d[len(r)][len(h)], len(r)

def calculate_wer_from_file(file_path):
    """
    파일에서 참조(ref)와 가설(hyp)을 읽어 전체 WER을 계산.
    파일 형식은 다음과 같아야 함:
    ref: <reference sentence>
    hyp: <hypothesis sentence>
    """
    total_errors = 0
    total_ref_length = 0

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("ref:"):
                reference = line.replace("ref:", "").strip()
            elif line.startswith("hyp:"):
                hypothesis = line.replace("hyp:", "").strip()
                errors, ref_length = calculate_wer_overall(reference, hypothesis)
                total_errors += errors
                total_ref_length += ref_length

    if total_ref_length == 0:
        raise ValueError("No valid ref/hyp pairs found in the file.")

    return total_errors / total_ref_length

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        wer = calculate_wer_from_file(file_path)
        print(f"Overall WER: {wer:.2%}")
    except Exception as e:
        print(f"Error: {e}")
