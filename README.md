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
-   [Agents for Software Development and Web Browsing (Graham Neubig)](https://phontron.com/class/anlp-fall2024/assets/slides/anlp-16-softwarewebagents.pdf)

## Notes

LLM-based Localization is a very interesting problem

-   Finding the correct files given user intent
-   **I will focus on an unsolved issue: when to perform RAG in agent**
-   Use GraphRAG when user intent requires multiple pieces of information, as it can handle multi-hop queries efficiently
-   Use PageRank to prioritize which nodes (i.e., files or functions) to explore
-   A high PageRank score indicates that an existing knowledge graph may suffice to generate accurate responses without external retrieval.
    This is because entities with high PageRank scores are typically well-connected and central to the structure of the knowledge graph.

## Implemenation

Objective: Fine-tune DeepSeek-Coder-V2-Instruct to retrieve the most relevant code snippets for a user query

-   Build a codebase graph to identify relationships between files, functions, classes, and dependencies
-   Generate synthetic user queries related to code functionality
-   Use PageRank scores to rank the most relevant code snippets for each query
-   Extract additional context for each snippet
-   Use PageRank scores during training to help determine code importance and during inference to help decide retrieval strategy (direct vs GraphRAG)
    -   Nodes are source files and the edges are the references between methods/classes in the source files
