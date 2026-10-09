# Call Me Maybe

> **Status:** Work in progress. This repository and its documentation will be continuously updated throughout development.

## Description
This project implements **Function Calling** for small language models (Qwen3-0.6B) from scratch. 

The goal is to translate natural language user prompts into structured, executable function calls with typed arguments. Instead of relying purely on prompt engineering, the project implements **Constrained Decoding** by masking raw logits with NumPy, guaranteeing 100% valid JSON output that strictly follows predefined schemas.

Loading...
