import numpy as np
from tqdm import tqdm
import sys

def levenshtein_distance(ref, hyp):
    """
    Calculate the Levenshtein distance between two strings.
    """
    ref_words = ref.split()
    hyp_words = hyp.split()
    len_ref = len(ref_words)
    len_hyp = len(hyp_words)

    # Initialize the distance matrix
    dp = np.zeros((len_ref + 1, len_hyp + 1), dtype=int)
    for i in range(len_ref + 1):
        dp[i][0] = i
    for j in range(len_hyp + 1):
        dp[0][j] = j

    # Compute distances
    for i in range(1, len_ref + 1):
        for j in range(1, len_hyp + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    return dp[len_ref][len_hyp]

def calculate_wer(ref, hyp):
    """
    Calculate the Word Error Rate (WER) between two strings.
    """
    ref_words = ref.split()
    return levenshtein_distance(ref, hyp) / len(ref_words) if ref_words else 0

def process_file(file_path):
    """
    Process a text file and calculate WER for reference and hypothesis pairs.
    """
    references = []
    hypotheses = []

    # Read the file
    with open(file_path, 'r') as file:
        for line in tqdm(file):
            if line.startswith("ref:"):
                references.append(line.replace("ref:", "").strip())
            elif line.startswith("hyp:"):
                hypotheses.append(line.replace("hyp:", "").strip())

    # Ensure the number of references and hypotheses match
    if len(references) != len(hypotheses):
        raise ValueError("Mismatch between reference and hypothesis counts.")

    # Calculate WER for each pair
    individual_wers = [calculate_wer(ref, hyp) for ref, hyp in tqdm(zip(references, hypotheses))]

    # Calculate overall WER
    overall_wer = calculate_wer(" ".join(references), " ".join(hypotheses))

    return individual_wers, overall_wer

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    individual_wers, overall_wer = process_file(file_path)
    print("Individual WERs:", individual_wers)
    print("Overall WER:", overall_wer)
