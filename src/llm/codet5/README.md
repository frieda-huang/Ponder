### Draft

We follow the architecture of [CodeT5+ Bimodal model](https://github.com/salesforce/CodeT5/tree/main/CodeT5%2B) because this model can be used for code summarization and code retrieval in a zero-shot manner. Interestingly, this model's encoder and projection layer share the same weights with the CodeT5+ 110M embedding model. Note that CodeT5+ uses T5's architecture but RoBERTa's tokenizer. T5 implements both encoder and decoder.

-   tokenizer: RobertaTokenizer
-   dataset: https://huggingface.co/datasets/code-search-net/code_search_net (focus on Python for now)
