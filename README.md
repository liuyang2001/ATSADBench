# ATSADBench

Our benchmark is mainly intended to serve as a common evaluation reference for industry practitioners working on satellite anomaly detection who can access such data under confidentiality arrangements.

If it is helpful for researchers who can not access such data under confidentiality arrangements, we can try to support you in alternative ways:

We can request an internal review to see whether we may provide a very small, declassified subset (e.g., a limited portion from 1–2 tasks) for reference; and/or

If you would like to evaluate specific methods on this benchmark, you may share your code (or a runnable package/container). We can run the evaluation on our side and share the resulting metrics and logs in a reproducible format.


## How to Reproduce the Results

First, install the required dependencies:

```bash
pip install -r requirement.txt
```

---

### Direct Paradigm

1. Change directory:

   ```bash
   cd direct
   ```
2. Run the following script to generate min-max values:

   ```bash
   python get_min_max_values_json.py
   ```

   This will generate `dataset/min_max_values.json`. Copy its contents into `config.py` (we use all zeros as placeholders).
3. Add your API key in `src/model_handler.py`.
4. For **DeepSeek-V3**, run:

   ```bash
   bash deepseek-chat.bash
   ```

   For **Qwen3**, run:

   ```bash
   bash qwen.bash
   ```
5. Run:

   ```bash
   python window-related-metric.py
   ```

   to obtain window-related metrics.
6. **To reproduce the few-shot results:**
   Set `POSITIVE_SAMPLE_NUMBER=1` and `NEGATIVE_SAMPLE_NUMBER=3` in `config.py`, then repeat steps 4 and 5.
7. **To reproduce the RAG results:**
   Set `RAG_NUMBER=1` in `config.py`, then repeat steps 4 and 5.

---

### Prediction-Based Paradigm

1. Change directory:

   ```bash
   cd prediction-based
   ```
2. Run the following script to generate min-max values:

   ```bash
   python get_min_max_values_json.py
   ```

   This will generate `dataset/min_max_values.json`. Copy its contents into `config.py` (we use all zeros as placeholders).
3. Add your API key in `src/model_handler.py`.
4. For **DeepSeek-V3**, run:

   ```bash
   bash deepseek-chat.bash
   ```

   For **Qwen3**, run:

   ```bash
   bash qwen.bash
   ```
5. Run:

   ```bash
   python get_auc.py
   ```

   to obtain AUROC and AUPRC.
6. Run:

   ```bash
   python window-related-metric.py
   ```

   to obtain window-related metrics.
7. **To reproduce the few-shot results:**
   Set `POSITIVE_SAMPLE_NUMBER=1` in `config.py`, then repeat steps 4, 5, and 6.
8. **To reproduce the RAG results:**
   Set `RAG_NUMBER=1` in `config.py`, then repeat steps 4, 5, and 6.

---

### baseline

For **GCAD**, run:

   ```bash
   bash GCAD.sh
   ```

For **Sub-Adjacent**, run:

   ```bash
   bash sub.sh
   ```
For **TFMAE**, run:

   ```bash
   bash TFMAE.sh
   ```
