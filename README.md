# ATSADBench

*The appendix has been uploaded.*

The dataset should be stored in the `direct/dataset` and `prediction-based/dataset` folders. Details are provided in the `dataset_illustration.txt` file within each folder. Due to data confidentiality, the dataset is not directly included in the GitHub repository. If you require access to the ATSADBench dataset for academic research, please send an email to **[liuyang12339@163.com]**. We will provide the dataset package.


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
