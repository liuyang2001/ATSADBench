# ATSADBench
The appendix has been uploaded.

How to reproduce the results in the paper:

**Direct paradigm:**

1. `cd direct`
2. `pip install -r requirement.txt`
3. Run `python get_min_max_values_json.py`. This will generate `dataset/min_max_values.json`. Copy its contents into `config.py` (we use all zeros as placeholders).
4. Add your API key in `src/model_handler.py`.
5. For DeepSeek-V3, run `bash deepseek-chat.bash`; for Qwen3, run `bash qwen.bash`.
6. Run `python window-related-metric.py` to obtain the window-related metrics.
7. To reproduce the few-shot results, set `POSITIVE_SAMPLE_NUMBER=1` and `NEGATIVE_SAMPLE_NUMBER=3` in `config.py`, then repeat steps 5 and 6.
8. To reproduce the RAG results, set `RAG_NUMBER=1` in `config.py`, then repeat steps 5 and 6.

