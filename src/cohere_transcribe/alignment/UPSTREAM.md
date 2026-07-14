# ctc-forced-aligner utilities

These files preserve the normalization and span behavior used by this project without requiring the maintained repository to publish under the unrelated `ctc-forced-aligner` name on PyPI.

Source: <https://github.com/MahmoudAshraf97/ctc-forced-aligner>

Revision: `11855d1de76af2b490dd2e8e2db2661805ae90a0`

Author: Mahmoud Ashraf

License: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

`punctuations.lst` is an exact copy. `norm_config.py` and `text_utils.py` retain the upstream Arabic/English normalization and timestamp behavior while removing unsupported language modes, split modes, non-romanized paths, unused metadata, and confidence aggregation. `alignment_utils.py` retains the upstream `Segment`, `merge_repeats`, and `get_spans` behavior without the unused `Segment.length` property. The application uses TorchAudio for the alignment kernel and does not need the upstream C++ extension or model-loading helpers.
