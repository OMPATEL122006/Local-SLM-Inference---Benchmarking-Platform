# Local SLM Inference & Benchmarking Platform

A local AI inference platform for running Small Language Models (SLMs) entirely offline using **Ollama**, exposing inference through a **FastAPI** backend, and benchmarking multiple models on the same consumer hardware.

The project focuses on understanding the practical trade-offs between **model quality, inference speed, latency, memory consumption, privacy, and cost** when running language models locally.

## Project Goals

* Run language models completely locally using Ollama.
* Provide a clean REST API for local model inference.
* Dynamically discover locally installed models.
* Support structured LLM outputs using Pydantic and evaluate the usefulness of Instructor.
* Benchmark multiple models under consistent hardware and workload conditions.
* Measure inference latency and generation throughput.
* Monitor system resource usage such as RAM and GPU VRAM.
* Compare model quality against speed and resource requirements.
* Document the practical trade-offs of local inference.

## Current Hardware

The initial experiments are being performed on a consumer laptop with:

* **CPU:** AMD Ryzen 5 6600H
* **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU
* **GPU VRAM:** 4 GB
* **System RAM:** 8 GB DDR5
* **OS:** Windows

The limited 8 GB system RAM and 4 GB GPU VRAM make this a useful environment for investigating the practical constraints of local SLM inference.

## Technology Stack

| Technology          | Purpose                                                |
| ------------------- | ------------------------------------------------------ |
| Python              | Application and benchmarking language                  |
| Ollama              | Local model runtime and model management               |
| Ollama Python SDK   | Communication with the local Ollama service            |
| FastAPI             | REST API layer                                         |
| Uvicorn             | ASGI server for FastAPI                                |
| Pydantic            | Request/response schemas and validation                |
| Instructor          | Structured LLM output and validation/retry experiments |
| GGUF / Quantization | Efficient local model representation                   |
| Git                 | Version control                                        |

## Architecture

```text
                         Client / Frontend
                                |
                                | HTTP
                                v
                         +-------------+
                         |   FastAPI   |
                         +------+------+
                                |
                     +----------+----------+
                     |                     |
                     v                     v
                Pydantic              Ollama Client
                Validation                  |
                                            v
                                      +-----------+
                                      |  Ollama   |
                                      +-----+-----+
                                            |
                         +------------------+------------------+
                         |                  |                  |
                         v                  v                  v
                      Model A            Model B            Model C
                         |                  |                  |
                         +------------------+------------------+
                                            |
                                            v
                                  Benchmarking Engine
                                            |
                           +----------------+----------------+
                           |                |                |
                           v                v                v
                         Speed           Resources        Quality
```

## Current API

The backend currently provides:

### `GET /`

Checks whether the local inference API is running.

### `GET /models`

Dynamically retrieves the models currently installed in Ollama.

### `POST /generate`

Runs a prompt against a selected local model and returns the generated response together with inference metrics.

Example request:

```json
{
  "prompt": "Explain how HTTP works.",
  "model": "phi3:mini"
}
```

The response includes metrics such as:

* Total inference duration
* Model load duration
* Prompt token count
* Prompt evaluation duration
* Generated token count
* Generation evaluation duration
* Tokens per second

## Initial Model

The first model tested is:

**Phi-3 Mini**

* Parameters: 3.8B
* Quantization: Q4_0
* Context length: 131,072 tokens

The model successfully runs locally with GPU acceleration on the RTX 3050.

## Initial Performance Observation

An initial API experiment produced approximately:

```text
Model:              Phi-3 Mini
Quantization:       Q4_0
Output tokens:      352
Generation speed:   ~18.86 tokens/sec
```

A subsequent request measured approximately:

```text
Output tokens:      129
Total latency:      ~11.80 seconds
Generation speed:   ~23.28 tokens/sec
```

These are **preliminary observations**, not final benchmark results. The final benchmark will use repeated, controlled workloads to reduce measurement noise.

## Benchmarking Plan

The final benchmark will compare three locally runnable models on the same hardware.

Measurements will include:

### Performance

* Total latency
* Cold-start latency
* Warm-request latency
* Prompt processing time
* Generation time
* Tokens per second

### Resource Usage

* System RAM
* GPU VRAM
* CPU utilization
* GPU utilization
* Model size

### Quality

Depending on the benchmark task:

* Factual accuracy
* Instruction following
* Summarization quality
* Structured-output reliability
* Reasoning performance
* Coding performance

The same prompts, evaluation criteria, and hardware conditions will be used across models wherever applicable.

## Project Structure

```text
Local SLM Inference & Benchmarking Platform/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── ollama_client.py
│   ├── schemas.py
│   └── config.py
│
├── benchmark/
│   ├── runner.py
│   ├── prompts.json
│   └── results/
│
├── tests/
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Design Principles

### Domain agnostic

The platform is not tied to a particular subject or type of problem. Example prompts used during development are only smoke tests.

The benchmark will use a diverse set of tasks to evaluate general-purpose local inference.

### Reproducible benchmarking

Models will be tested under controlled conditions rather than comparing isolated responses.

### Separate application and experimentation

The API implementation and benchmark framework are intentionally separated so that application behavior does not interfere with experimental measurement.

### Local-first

Inference is performed through the local Ollama runtime rather than a cloud LLM API.

## Future Work

* [ ] Implement structured output endpoint
* [ ] Evaluate native Ollama structured outputs
* [ ] Integrate and evaluate Instructor
* [ ] Build automated benchmark runner
* [ ] Select final three benchmark models
* [ ] Add repeated benchmark trials
* [ ] Add RAM/CPU monitoring
* [ ] Add GPU/VRAM monitoring
* [ ] Implement quality evaluation
* [ ] Build model comparison dashboard
* [ ] Build lightweight frontend
* [ ] Generate benchmark graphs
* [ ] Document final findings
* [ ] Add automated tests
* [ ] Finalize project documentation

## Objective

The ultimate goal is to determine:

> **When is running an SLM locally a better engineering choice than using a cloud-based LLM, and what compromises are required in terms of quality, speed, memory, privacy, and cost?**
