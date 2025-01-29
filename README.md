# 📜 ZenML RAG Pipeline Template

This repository contains a starter template for building Retrieval Augmented Generation (RAG) 
pipelines with ZenML. It provides a complete implementation of a RAG pipeline that includes:

- Document ingestion and chunking
- Vector store integration for efficient retrieval
- LLM integration for generation using LangGraph for composable pipelines
- API endpoint deployment
- Evaluation metrics and monitoring

The template contains all the necessary steps, pipeline configurations, stack components and
other artifacts to get you started with building production-ready RAG applications. The implementation
in the `template/` directory showcases a LangGraph-based assistant that:

- Ingests and embeds documents into a vector store
- Uses a composable graph structure for flexible RAG workflows
- Provides evaluation capabilities to assess assistant performance
- Can be deployed as an API endpoint

🔥 **Looking to build a RAG application with ZenML?**

This template provides a solid foundation for building RAG applications with proper MLOps
practices. Whether you're building a chatbot, question-answering system, or any other
RAG-based application, this template will help you get started quickly. If you have
questions or want to share your RAG use case, please [join our Slack](https://zenml.io/slack/)
and let us know!

## 📦 Prerequisites

To use the templates, you need to have Zenml and its `templates` extras
installed: 

```bash
pip install "zenml[templates]"
```

## 🚀 Generate a ZenML Project

You can generate a project from one of the existing templates by using the
`--template` flag with the `zenml init` command:

```bash
zenml init --template
```

Under the hood, ZenML uses the popular [Copier](https://copier.readthedocs.io/en/stable/)
library and a set of Jinja2 templates to generate the project. So you may also
interact with Copier directly to generate a project, e.g.:

```bash
copier gh:zenml-io/template-rag <directory>
```

You will be prompted to select the project template and enter various values for
the template variables. Once you have entered them, the project will be
generated in the indicated path.

To update an already generated project, with different parameters you can run
the same command again. If you want to skip the prompts to use the values you
already entered and overwrite all files in the existing project, you can run:

```bash
copier -wf gh:zenml-io/template-rag <directory>
```
