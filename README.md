# Ponder

A tool for developers to understand codebases efficiently.

## Features

**Code Parsing**

-   Extracts file structures, classes, functions, and dependencies.

**Code Q&A**

Ask questions like:

-   What does this function do?
-   Which modules does this class depend on?
-   Where is this function used?

The system provides:

-   Contextual answers from the codebase.
-   Links to relevant file locations or dependency visualizations.
-   Step-by-step explanations of complex interactions.

## Examples

**Query**

How does the `retrieve_relevant_pages` function work, and how does it leverage ColPali’s indexing strategies within the retrieval architecture?

**Expected Answer**
The `retrieve_relevant_pages` function takes a query and retrieves the top relevant PDF pages by:

1. Embedding the query using ColPali.
2. Performing an approximate nearest neighbor search based on the specified indexing strategy (HNSW or IVFFlat) to identify candidate pages.
3. Re-ranking the candidates using ColPali’s late interaction scoring, which fine-tunes relevance by comparing embeddings at a more granular level.

_Design Patterns_

1. Strategy Pattern
   ...
2. Dependency Injection
   ...

## Tech stack

| **Category**                       | **Tools/Technologies**                                                                                                     |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Frontend**                       | • Next.js <br> • Typescript <br> • Shadcn                                                                                  |
| **LLM Orchestration**              | [LiteLLM](https://github.com/BerriAI/litellm/tree/main)                                                                    |
| **LLM Model**                      | • Claude 3.5 Sonnet <br> • [DeepSeek-Coder-V2](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct)              |
| **Model Training and Fine-tuning** | [LLM Foundry](https://github.com/mosaicml/llm-foundry)                                                                     |
| **Performance Optimization**       | • [flash-attention](https://github.com/Dao-AILab/flash-attention) <br> • [Triton](https://triton-lang.org/main/index.html) |
| **Synthetic Data Generation**      | GPT-o1                                                                                                                     |
| **Evaluation**                     | [CodeXGLUE](https://github.com/microsoft/CodeXGLUE)                                                                        |
| **Code Parsing**                   | [tree-sitter](https://github.com/tree-sitter/tree-sitter?tab=readme-ov-file)                                               |

## Experimentation Plan

1. Fine-tune DeepSeek-Coder-V2-Instruct
2. Patch the model architecture (namely DeepSeek-Coder-V2-Instruct) to use the Flash Attention v2 Triton kernel
3. Use MosaicML with FSDP

## Resources

For implementation, we follow:

-   [Replit’s LLM Training Blog](https://blog.replit.com/llm-training)
-   [Replit’s Code Repair Blog](https://blog.replit.com/code-repair)
