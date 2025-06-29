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

## RTX 4090 / Ada Lovelace Optimization Notes

### Background (January 2025)
Attempted to optimize llama.cpp flash attention for RTX 4090 (Ada Lovelace architecture) with enhanced capabilities:
- 544 4th-gen Tensor Cores
- 96MB L2 cache (16x larger than previous gen)
- 128 CUDA cores per SM (vs 64 on Ampere)
- 1008 GB/s memory bandwidth
- Enhanced occupancy characteristics

### What We Learned

#### CUDA Template System Constraints
The flash attention implementation uses a complex template instantiation system that is extremely fragile:
- Template parameters like `nwarps_max`, `nstages_target`, `nbatch_fa` affect kernel instantiation
- Kernel variants are pre-compiled in `template-instances/` directory
- Changing config parameters breaks template matching, causing build failures
- Runtime variables cannot be used in compile-time template expressions (`std::conditional`)

#### Failed Approaches
1. **Direct Config Modification**: Modifying `fattn_mma_f16_config<128,128>` parameters broke template system
2. **Conditional Templates**: Using `std::conditional` with runtime `cc` variable caused compile errors
3. **Incremental Changes**: Even small increases to batch sizes caused template mismatches

#### Successful Framework Creation
Created comprehensive Ada Lovelace optimization framework in `rtx4090` branch:
- `fattn-ada.cuh`: Enhanced flash attention configurations with 2x-4x improvements
- `mma-ada.cuh`: Tensor core enhancements with BF16 support  
- `l2-cache-ada.cuh`: L2 cache optimization for 96MB cache
- `ada-thread-config.cuh`: Thread block optimization for Ada architecture
- `ada-memory.cuh`: Memory bandwidth optimization (vectorized operations)
- `ada-benchmark.cu/cuh`: Benchmarking suite
- `fattn.cu`: Added Ada Lovelace detection and dispatch

#### Key Insights
- BF16 inference is supported (GGML_TYPE_BF16, type 30) and would likely provide 5-15% speedup on RTX 4090
- Ada architecture detection works correctly (compute capability 8.9)
- Diagnostic output confirms optimization path selection
- Framework is in place but kernel still uses standard parameters due to template constraints

#### Future Approaches
To achieve actual performance improvements, consider:
1. **New Template Instantiations**: Add Ada-specific template variants to `template-instances/`
2. **Dynamic Kernel Selection**: Modify kernel generation to support runtime parameter selection
3. **Separate Ada Kernels**: Create entirely separate optimized kernels for Ada Lovelace
4. **Build System Changes**: Modify CMake/build system to generate Ada-optimized variants

#### Commands Used
```bash
# Create optimization branch
git checkout -b rtx4090

# Build and test (requires CUDA support)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

# Push optimizations
git push brayniac rtx4090
```

The optimization framework provides a solid foundation for future Ada Lovelace enhancements once the template system constraints are properly addressed.