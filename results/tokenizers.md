| label | source | vocab_size | identical_to | status |
| :-- | :-- | --: | :-- | :-- |
| GPT-2 | `openai-community/gpt2@607a30d` | 50257 |  | ok |
| cl100k_base | `tiktoken:cl100k_base` | 100277 |  | ok |
| o200k_base | `tiktoken:o200k_base` | 200019 |  | ok |
| Llama 3.x | `unsloth/Llama-3.2-1B-Instruct@5a8abab` | 128256 |  | ok |
| Qwen 2.5 | `Qwen/Qwen2.5-7B-Instruct@a09a354` | 151665 |  | ok |
| Qwen 3 | `Qwen/Qwen3-8B@b968826` | 151669 | Qwen 2.5 | ok |
| Gemma 2 | `unsloth/gemma-2-2b-it@457f2e1` | 256000 |  | ok |
| Gemma 3 | `unsloth/gemma-3-1b-it@5b11413` | 262145 |  | ok |
| Mistral 7B v0.3 | `mistralai/Mistral-7B-Instruct-v0.3@c170c70` | 32768 |  | ok |
| Mistral Nemo (Tekken) | `mistralai/Mistral-Nemo-Instruct-2407@04d8a90` | 131072 |  | ok |
| BLOOM | `bigscience/bloom@7f10a99` | 250680 |  | ok |
| mT5 | `google/mt5-base@2eb1546` | 250100 |  | ok |
| XLM-R | `FacebookAI/xlm-roberta-base@e73636d` | 250002 |  | ok |
| AfroXLMR | `Davlan/afro-xlmr-base@25f2729` | 250002 | XLM-R | ok |
| AfriBERTa | `castorini/afriberta_large@231d924` | 70006 |  | ok |

Skipped:

| repo | reason |
| :-- | :-- |
| `meta-llama/Llama-3.2-1B-Instruct` | gated (manual approval); measured through the unsloth mirror instead |
| `google/gemma-2-2b-it` | gated (manual approval); measured through the unsloth mirror instead |
| `google/gemma-3-1b-it` | gated (manual approval); measured through the unsloth mirror instead |
| `lelapa/InkubaLM-0.4B` | gated (requires a Hugging Face login); no ungated copy found |
| `openlanguagedata/flores_plus` | dataset is gated; mteb/flores mirror used |
| `facebook/flores` | dataset is gated; mteb/flores mirror used |
