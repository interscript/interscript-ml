# ara-diac-news-1.0

Arabic **news-register specialist** (run-029): byt5-base seq2seq,
register-pure training — 900K QCRI-silver units (~4.5M words of
Wikipedia news-domain labels, published with EMNLP 2025) + WikiNews
gold upsampling, NO classical tashkeela dilution, 2 epochs, r7-init.
IMF v1 int8, 399 MB.

## Metrics

| benchmark | this model | ara-diac-2.0 (generalist) |
|---|---|---|
| WikiNews-2024 multiref WER/DER | **10.13 / 8.98** | 17.38 / 11.83 |
| SadeedDiac-25 Total DER | 5.5008 | **2.2864** |

Use this model for news-domain text; use ara-diac-2.0 for
classical/MSA. Two families, never averaged (RESULTS.md: the
dose-response curve and the output-space oracle both close generalist
blending).

## Provenance

- Training: TODO.sota/25 (run-029, HF Jobs a100-large).
- sha256: `e5a52b1530dfa6d6c7ef8970443206ce33af865d0337a58d145945adab33484e`.
- License: BSD-3-Clause (code + weights).
