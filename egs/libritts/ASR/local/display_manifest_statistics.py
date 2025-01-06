#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.             (authors: Zengwei Yao)
#              2024  The Chinese Univ. of HK  (authors: Zengrui Jin)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file displays duration statistics of utterances in a manifest.
You can use the displayed value to choose minimum/maximum duration
to remove short and long utterances during the training.
"""


from lhotse import load_manifest_lazy


def main():
    paths = [
        "./data/fbank/libritts_cuts_train-clean-100.jsonl.gz",
        "./data/fbank/libritts_cuts_train-clean-360.jsonl.gz",
        "./data/fbank/libritts_cuts_train-other-500.jsonl.gz",
        "./data/fbank/libritts_cuts_dev-clean.jsonl.gz",
        "./data/fbank/libritts_cuts_dev-other.jsonl.gz",
        "./data/fbank/libritts_cuts_test-clean.jsonl.gz",
        "./data/fbank/libritts_cuts_test-other.jsonl.gz",
    ]
    for path in paths:
        cuts = load_manifest_lazy(path)
        cuts.describe()


if __name__ == "__main__":
    main()

"""
./data/fbank/libritts_cuts_train-clean-100.jsonl.gz statistics:
________________________________________
_ Cuts count:               _ 33236    _
________________________________________
_ Total duration (hh:mm:ss) _ 53:47:18 _
________________________________________
_ mean                      _ 5.8      _
________________________________________
_ std                       _ 4.6      _
________________________________________
_ min                       _ 0.2      _
________________________________________
_ 25%                       _ 2.4      _
________________________________________
_ 50%                       _ 4.5      _
________________________________________
_ 75%                       _ 7.9      _
________________________________________
_ 99%                       _ 21.4     _
________________________________________
_ 99.5%                     _ 23.7     _
________________________________________
_ 99.9%                     _ 27.8     _
________________________________________
_ max                       _ 33.2     _
________________________________________
_ Recordings available:     _ 33236    _
________________________________________
_ Features available:       _ 33236    _
________________________________________
_ Supervisions available:   _ 33236    _
________________________________________
SUPERVISION custom fields:
Speech duration statistics:
__________________________________________________________________
_ Total speech duration        _ 53:47:18 _ 100.00% of recording _
__________________________________________________________________
_ Total speaking time duration _ 53:47:18 _ 100.00% of recording _
__________________________________________________________________
_ Total silence duration       _ 00:00:01 _ 0.00% of recording   _
__________________________________________________________________

./data/fbank/libritts_cuts_train-clean-360.jsonl.gz statistics:
_________________________________________
_ Cuts count:               _ 116500    _
_________________________________________
_ Total duration (hh:mm:ss) _ 191:17:42 _
_________________________________________
_ mean                      _ 5.9       _
_________________________________________
_ std                       _ 4.6       _
_________________________________________
_ min                       _ 0.1       _
_________________________________________
_ 25%                       _ 2.4       _
_________________________________________
_ 50%                       _ 4.6       _
_________________________________________
_ 75%                       _ 8.1       _
_________________________________________
_ 99%                       _ 21.3      _
_________________________________________
_ 99.5%                     _ 23.4      _
_________________________________________
_ 99.9%                     _ 27.4      _
_________________________________________
_ max                       _ 40.4      _
_________________________________________
_ Recordings available:     _ 116500    _
_________________________________________
_ Features available:       _ 116500    _
_________________________________________
_ Supervisions available:   _ 116500    _
_________________________________________
SUPERVISION custom fields:
Speech duration statistics:
___________________________________________________________________
_ Total speech duration        _ 191:17:42 _ 100.00% of recording _
___________________________________________________________________
_ Total speaking time duration _ 191:17:42 _ 100.00% of recording _
___________________________________________________________________
_ Total silence duration       _ 00:00:01  _ 0.00% of recording   _
___________________________________________________________________

./data/fbank/libritts_cuts_train-other-500.jsonl.gz statistics:
_________________________________________
_ Cuts count:               _ 205043    _
_________________________________________
_ Total duration (hh:mm:ss) _ 310:04:36 _
_________________________________________
_ mean                      _ 5.4       _
_________________________________________
_ std                       _ 4.4       _
_________________________________________
_ min                       _ 0.1       _
_________________________________________
_ 25%                       _ 2.3       _
_________________________________________
_ 50%                       _ 4.2       _
_________________________________________
_ 75%                       _ 7.3       _
_________________________________________
_ 99%                       _ 20.6      _
_________________________________________
_ 99.5%                     _ 22.8      _
_________________________________________
_ 99.9%                     _ 27.4      _
_________________________________________
_ max                       _ 43.9      _
_________________________________________
_ Recordings available:     _ 205043    _
_________________________________________
_ Features available:       _ 205043    _
_________________________________________
_ Supervisions available:   _ 205043    _
_________________________________________
SUPERVISION custom fields:
Speech duration statistics:
___________________________________________________________________
_ Total speech duration        _ 310:04:36 _ 100.00% of recording _
___________________________________________________________________
_ Total speaking time duration _ 310:04:36 _ 100.00% of recording _
___________________________________________________________________
_ Total silence duration       _ 00:00:01  _ 0.00% of recording   _
___________________________________________________________________

./data/fbank/libritts_cuts_dev-clean.jsonl.gz statistics:
________________________________________
_ Cuts count:               _ 5736     _
________________________________________
_ Total duration (hh:mm:ss) _ 08:58:13 _
________________________________________
_ mean                      _ 5.6      _
________________________________________
_ std                       _ 4.3      _
________________________________________
_ min                       _ 0.3      _
________________________________________
_ 25%                       _ 2.4      _
________________________________________
_ 50%                       _ 4.4      _
________________________________________
_ 75%                       _ 7.8      _
________________________________________
_ 99%                       _ 19.9     _
________________________________________
_ 99.5%                     _ 21.9     _
________________________________________
_ 99.9%                     _ 26.3     _
________________________________________
_ max                       _ 30.1     _
________________________________________
_ Recordings available:     _ 5736     _
________________________________________
_ Features available:       _ 5736     _
________________________________________
_ Supervisions available:   _ 5736     _
________________________________________
SUPERVISION custom fields:
Speech duration statistics:
__________________________________________________________________
_ Total speech duration        _ 08:58:13 _ 100.00% of recording _
__________________________________________________________________
_ Total speaking time duration _ 08:58:13 _ 100.00% of recording _
__________________________________________________________________
_ Total silence duration       _ 00:00:01 _ 0.00% of recording   _
__________________________________________________________________

./data/fbank/libritts_cuts_dev-other.jsonl.gz statistics:
________________________________________
_ Cuts count:               _ 4613     _
________________________________________
_ Total duration (hh:mm:ss) _ 06:25:52 _
________________________________________
_ mean                      _ 5.0      _
________________________________________
_ std                       _ 4.1      _
________________________________________
_ min                       _ 0.3      _
________________________________________
_ 25%                       _ 2.2      _
________________________________________
_ 50%                       _ 3.8      _
________________________________________
_ 75%                       _ 6.5      _
________________________________________
_ 99%                       _ 19.7     _
________________________________________
_ 99.5%                     _ 24.5     _
________________________________________
_ 99.9%                     _ 31.0     _
________________________________________
_ max                       _ 32.6     _
________________________________________
_ Recordings available:     _ 4613     _
________________________________________
_ Features available:       _ 4613     _
________________________________________
_ Supervisions available:   _ 4613     _
________________________________________
SUPERVISION custom fields:
Speech duration statistics:
__________________________________________________________________
_ Total speech duration        _ 06:25:52 _ 100.00% of recording _
__________________________________________________________________
_ Total speaking time duration _ 06:25:52 _ 100.00% of recording _
__________________________________________________________________
_ Total silence duration       _ 00:00:01 _ 0.00% of recording   _
__________________________________________________________________

./data/fbank/libritts_cuts_test-clean.jsonl.gz statistics:
________________________________________
_ Cuts count:               _ 4837     _
________________________________________
_ Total duration (hh:mm:ss) _ 08:34:09 _
________________________________________
_ mean                      _ 6.4      _
________________________________________
_ std                       _ 5.1      _
________________________________________
_ min                       _ 0.3      _
________________________________________
_ 25%                       _ 2.4      _
________________________________________
_ 50%                       _ 4.8      _
________________________________________
_ 75%                       _ 8.9      _
________________________________________
_ 99%                       _ 22.6     _
________________________________________
_ 99.5%                     _ 24.4     _
________________________________________
_ 99.9%                     _ 29.6     _
________________________________________
_ max                       _ 36.7     _
________________________________________
_ Recordings available:     _ 4837     _
________________________________________
_ Features available:       _ 4837     _
________________________________________
_ Supervisions available:   _ 4837     _
________________________________________
SUPERVISION custom fields:
Speech duration statistics:
__________________________________________________________________
_ Total speech duration        _ 08:34:09 _ 100.00% of recording _
__________________________________________________________________
_ Total speaking time duration _ 08:34:09 _ 100.00% of recording _
__________________________________________________________________
_ Total silence duration       _ 00:00:01 _ 0.00% of recording   _
__________________________________________________________________

./data/fbank/libritts_cuts_test-other.jsonl.gz statistics:
________________________________________
_ Cuts count:               _ 5120     _
________________________________________
_ Total duration (hh:mm:ss) _ 06:41:31 _
________________________________________
_ mean                      _ 4.7      _
________________________________________
_ std                       _ 3.8      _
________________________________________
_ min                       _ 0.3      _
________________________________________
_ 25%                       _ 1.8      _
________________________________________
_ 50%                       _ 3.6      _
________________________________________
_ 75%                       _ 6.5      _
________________________________________
_ 99%                       _ 17.8     _
________________________________________
_ 99.5%                     _ 20.4     _
________________________________________
_ 99.9%                     _ 23.8     _
________________________________________
_ max                       _ 27.3     _
________________________________________
_ Recordings available:     _ 5120     _
________________________________________
_ Features available:       _ 5120     _
________________________________________
_ Supervisions available:   _ 5120     _
________________________________________
SUPERVISION custom fields:
Speech duration statistics:
__________________________________________________________________
_ Total speech duration        _ 06:41:31 _ 100.00% of recording _
__________________________________________________________________
_ Total speaking time duration _ 06:41:31 _ 100.00% of recording _
__________________________________________________________________
_ Total silence duration       _ 00:00:01 _ 0.00% of recording   _
__________________________________________________________________
"""
