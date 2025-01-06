#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

stage=0
stop_stage=100
sampling_rate=24000
nj=32

dl_dir=$PWD/download

. shared/parse_options.sh || exit 1

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"

  # If you have pre-downloaded it to /path/to/LibriTTS,
  # you can create a symlink
  #
  #   ln -sfv /path/to/LibriTTS $dl_dir/LibriTTS
  #
  if [ ! -d $dl_dir/LibriTTS ]; then
    lhotse download libritts $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare LibriTTS manifest"
  # We assume that you have downloaded the LibriTTS corpus
  # to $dl_dir/LibriTTS
  mkdir -p data/manifests
  if [ ! -e data/manifests/.libritts.done ]; then
    lhotse prepare libritts --num-jobs ${nj} $dl_dir/LibriTTS data/manifests
    touch data/manifests/.libritts.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Compute Spectrogram for LibriTTS"
  mkdir -p data/spectrogram
  if [ ! -e data/spectrogram/.libritts.done ]; then
    ./local/compute_spectrogram_libritts.py --sampling-rate $sampling_rate 
    touch data/spectrogram/.libritts.done
  fi

  # Here we shuffle and combine the train-clean-100, train-clean-360 and 
  # train-other-500 together to form the training set.
  if [ ! -f data/spectrogram/libritts_cuts_train-all-shuf.jsonl.gz ]; then
    cat <(gunzip -c data/spectrogram/libritts_cuts_train-clean-100.jsonl.gz) \
      <(gunzip -c data/spectrogram/libritts_cuts_train-clean-360.jsonl.gz) \
      <(gunzip -c data/spectrogramlibritts_cuts_train-other-500.jsonl.gz) | \
      shuf | gzip -c > data/spectrogram/libritts_cuts_train-all-shuf.jsonl.gz
  fi

  if [ ! -e data/spectrogram/.libritts-validated.done ]; then
    log "Validating data/spectrogram for LibriTTS"
    ./local/validate_manifest.py \
      data/spectrogram/libritts_cuts_train-all-shuf.jsonl.gz
    touch data/spectrogram/.libritts-validated.done
  fi
fi

