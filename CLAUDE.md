# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

llama.cpp is a high-performance C/C++ implementation of Large Language Model inference with minimal dependencies. The project focuses on enabling LLM inference across diverse hardware platforms (CPU, CUDA, Metal, Vulkan, etc.) with extensive optimization and quantization support.

## Development Commands

### Building
```bash
# Standard build
cmake -B build
cmake --build build --config Release

# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# With parallel jobs
cmake --build build --config Release -j 8
```

### Testing
```bash
# Run all main tests
ctest -L main --verbose

# Run specific test pattern
ctest -R "test-tokenizer" --verbose

# Run tests with timeout
ctest -L main --verbose --timeout 900
```

### Code Formatting
```bash
# Format C/C++ code
clang-format -i src/file.cpp

# Format all files
find . -name "*.cpp" -o -name "*.h" -o -name "*.c" | xargs clang-format -i

# Run static analysis
clang-tidy src/file.cpp
```

## Architecture

### Core Components
- **GGML Library** (`ggml/`): Low-level tensor operations and ML primitives
- **LLAMA Library** (`src/`): High-level LLM implementation built on GGML
- **Common Library** (`common/`): Shared utilities (argument parsing, logging, sampling)

### Key Directories
- `src/`: Core llama.cpp implementation with modular design
- `ggml/src/`: Hardware-specific backends (CPU, CUDA, Metal, Vulkan, etc.)
- `tools/`: CLI applications (main, server, quantize, bench, etc.)
- `common/`: Shared utilities and helper functions
- `tests/`: Comprehensive test suite
- `examples/`: Sample implementations
- `docs/`: Documentation including build guides

### Backend Architecture
The project uses a sophisticated multi-backend architecture:
- **CPU**: Highly optimized with SIMD (AVX, NEON) and threading
- **NVIDIA**: CUDA kernels for tensor operations
- **Apple**: Metal framework for Apple Silicon
- **AMD**: HIP support for ROCm
- **Intel**: SYCL backend for Intel GPUs
- **Cross-platform**: Vulkan and OpenCL support

## Code Style

### Formatting (enforced by .clang-format)
- 120 character line limit
- 4 spaces for indentation (no tabs)
- Pointer style: `void * ptr`
- Reference style: `int & ref`
- Brackets on same line
- Extensive vertical alignment

### Naming Conventions
- Functions/variables: `snake_case`
- Enum values: `UPPER_CASE` with prefix
- Pattern: `<class>_<method>` where method is `<action>_<noun>`
- Files: lowercase with dashes (C/C++), underscores (Python)

### Development Principles
- Avoid fancy STL constructs
- Use basic for loops and simple patterns
- Maintain cross-platform compatibility
- Use sized integer types (`int32_t`, `uint64_t`)
- Keep code simple and readable

## Model Conversion

### Convert HuggingFace models
```bash
# Convert to GGUF format
python convert_hf_to_gguf.py /path/to/model

# Convert LoRA adapters
python convert_lora_to_gguf.py /path/to/lora
```

### Quantization
```bash
# Quantize model
./build/bin/llama-quantize model.gguf model_q4_0.gguf q4_0
```

## Testing Strategy

### Test Categories
- **Tokenizer tests**: Various vocabulary formats (BPE, SentencePiece)
- **Grammar tests**: Structured output parsing
- **Backend tests**: Hardware-specific functionality
- **API tests**: C/C++ interface validation
- **Model tests**: Loading and inference validation

### Test Labels
- `main`: Standard CI tests
- `model`: Tests requiring model files
- `curl`: Network-dependent tests

## Performance Considerations

### Key Optimizations
- Memory-mapped I/O for large models
- Quantization support (1.5-bit to 8-bit)
- SIMD instruction utilization
- Multi-threading and parallel processing
- Speculative decoding capabilities

### Benchmarking
```bash
# Run performance benchmarks
./build/bin/llama-bench -m model.gguf

# Measure perplexity
./build/bin/llama-perplexity -m model.gguf -f test.txt
```

## Common File Patterns

- `llama-*.cpp/.h`: Modular LLM components (vocab, grammar, sampling)
- `ggml-*.cpp/.h`: Backend implementations
- `test-*.cpp`: Unit tests
- `*-cli.cpp`: Command-line interface tools
- `convert_*.py`: Model conversion scripts