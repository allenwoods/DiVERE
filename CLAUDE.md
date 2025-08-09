# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DiVERE is a professional film color grading tool built on ACEScg Linear workflow. It provides a density-based workflow for film digitization post-processing with AI-powered auto color correction.

## Common Development Commands

### Running the Application
```bash
# Run with python directly
python -m divere

# Or if using uv (preferred):
uv run python -m divere
```

### Package Management
```bash
# Install dependencies
pip install -r requirements.txt

# For development with Poetry (if pyproject.toml is configured):
poetry install
```

### Building and Packaging
```bash
# Build for macOS
./packaging/scripts/build_macos.sh

# Build for Windows
packaging\scripts\build_windows.bat

# Build for Linux
./packaging/scripts/build_linux.sh

# Build for all platforms
./packaging/scripts/build_all.sh
```

### Code Quality Tools
The project has Poetry dev dependencies configured but no specific lint/typecheck commands in scripts. Use:
```bash
# Format code
black divere/

# Check imports
isort divere/

# Lint code
flake8 divere/

# Type checking
mypy divere/
```

## High-Level Architecture

### Core Processing Pipeline
The application follows a linear density-based workflow:

1. **Input Stage**: Load image → Convert to ACEScg Linear working space
2. **Density Processing**: Linear values → Density space (negative log10)
3. **Color Correction**: Apply corrections in density space (matrix, gains, curves)
4. **Output Stage**: Convert back to linear → Apply output color space

### Key Components

#### Image Processing Core (`divere/core/`)
- **ImageManager**: Handles image loading, proxy generation, and caching. Supports TIFF/JPEG/PNG/RAW formats with proper channel ordering.
- **ColorSpaceManager**: Manages color space conversions using colour-science. Central to the ACEScg Linear workflow.
- **TheEnlarger**: Main processing engine implementing the density-based pipeline. Includes AI auto-correction via Deep White Balance model.
- **LUTProcessor**: Generates 3D/1D LUTs with caching for real-time preview performance.

#### User Interface (`divere/ui/`)
- **MainWindow**: Qt-based main application window coordinating all components
- **PreviewWidget**: Real-time preview with zoom/pan capabilities
- **ParameterPanel**: Controls for all processing parameters (density, matrix, curves)
- **CurveEditorWidget**: Custom curve editor for density-based adjustments

#### AI Components (`divere/colorConstancyModels/`)
- Deep White Balance integration via ONNX model (`net_awb.onnx`)
- Wrapper around the AI model for automatic color correction
- Falls back gracefully if model is unavailable

### Configuration System
- User configs stored in platform-specific directories
- JSON-based configuration for color spaces, curves, and matrices
- Enhanced config manager supports user overrides of built-in configs

### Critical Processing Details
- All processing done in 32-bit float for precision
- Density space allows natural film-like adjustments
- Status M to Print Density matrix for accurate film emulation
- Monotonic cubic interpolation for smooth curves

## Important Notes
- The AI model file (`net_awb.onnx`) is required for auto color correction
- Processing is optimized for real-time preview with proxy images
- Full resolution processing maintains complete precision
- User configurations in `config/` override built-in defaults