#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

# run step 0 to step 5 by default
stage=-1
stop_stage=4

dl_dir=$PWD/download
fbank_dir=data/fbank

# we assume that you have your downloaded the AudioSet and placed
# it under $dl_dir/audioset, the folder structure should look like
# this:
# - $dl_dir/audioset
#       - balanced
#       - eval
#       - unbalanced
# If you haven't downloaded the AudioSet, please refer to
# https://github.com/RicherMans/SAT/blob/main/datasets/audioset/1_download_audioset.sh.

. shared/parse_options.sh || exit 1

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "Running prepare.sh"

log "dl_dir: $dl_dir"

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  log "Stage 0: Download the necessary csv files"
  if [ ! -e $dl_dir/audioset/.csv.done]; then
    wget --continue "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv" -O "${dl_dir}/audioset/class_labels_indices.csv"
    wget --continue http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv -O "${dl_dir}/audioset/balanced_train_segments.csv"
    wget --continue http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv -O "${dl_dir}/audioset/eval_segments.csv"
    touch $dl_dir/audioset/.csv.done
  fi
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Construct the audioset manifest and compute the fbank features for balanced set"
  if [! -e $fbank_dir/.balanced.done]; then
    python local/generate_audioset_manifest.py \
      --dataset-dir $dl_dir/audioset \
      --split balanced \
      --feat-output-dir $fbank_dir
    touch $fbank_dir/.balanced.done
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Construct the audioset manifest and compute the fbank features for unbalanced set"
  fbank_dir=data/fbank
  if [! -e $fbank_dir/.unbalanced.done]; then
    python local/generate_audioset_manifest.py \
      --dataset-dir $dl_dir/audioset \
      --split unbalanced \
      --feat-output-dir $fbank_dir
    touch $fbank_dir/.unbalanced.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Construct the audioset manifest and compute the fbank features for eval set"
  fbank_dir=data/fbank
  if [! -e $fbank_dir/.eval.done]; then
    python local/generate_audioset_manifest.py \
      --dataset-dir $dl_dir/audioset \
      --split eval \
      --feat-output-dir $fbank_dir
    touch $fbank_dir/.eval.done
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Prepare musan manifest"
  # We assume that you have downloaded the musan corpus
  # to $dl_dir/musan
  mkdir -p data/manifests
  if [ ! -e data/manifests/.musan.done ]; then
    lhotse prepare musan $dl_dir/musan data/manifests
    touch data/manifests/.musan.done
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute fbank for musan"
  mkdir -p data/fbank
  if [ ! -e data/fbank/.musan.done ]; then
    ./local/compute_fbank_musan.py
    touch data/fbank/.musan.done
  fi
fi

# The following stages are required to do weighted-sampling training
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Prepare for weighted-sampling training"
  if [ ! -e $fbank_dir/cuts_audioset_full.jsonl.gz ]; then
    lhotse combine $fbank_dir/cuts_audioset_balanced.jsonl.gz $fbank_dir/cuts_audioset_unbalanced.jsonl.gz $fbank_dir/cuts_audioset_full.jsonl.gz
  fi
  python ./local/compute_weight.py \
    --input-manifest $fbank_dir/cuts_audioset_full.jsonl.gz \
    --output $fbank_dir/sampling_weights_full.txt
fi
