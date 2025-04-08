# research-llm-exercise
## Introduction
- This is a boilerplate for using our Azure OpenAI API.

- This is a demonstration. You can use the provided keys with any other framework / API.

- We provide access to two models:  
  - GPT-4o
  - Ada-2

- We will provide you a `.env` file containing the relevant deployment names, keys, and other parameter files, allowing you to use it.

- This is a suggestion; you can use any other model.

## Task
Write a chatbot application which works with a GitHub repository (a cloned folder can work as well) as a context. It gets a question about the code from the user as an input, then answers him.

Here are some examples of questions:
1. Is there any authentication-related logic in the code? If so, where is it?
2. What does the following application do?
3. What is the flow (classes and function calls) of an executable signing in this application?

The application should leverage an LLM model in order to answer the question. Embeddings can also be used if necessary.

The UI part is not important, it can also be an interactive CLI.

### Notes
1. For a Java application, use the [`jsign`](https://github.com/ebourg/jsign) application repository. 
2. For a Python application, use the [`signify`](https://github.com/ralphje/signify) Python module repository.