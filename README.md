# AI Drafting Agent

An intelligent automated system for generating professional 2D technical drawings 
from 3D CAD models using FreeCAD's TechDraw workbench.

## Overview

The AI Drafting Agent bridges the gap between 3D CAD geometry and manufacturing-ready 
2D documentation by programmatically interpreting 3D data and generating professional 
technical drawings.

### Key Features

- **Automatic View Generation**: Creates optimal orthographic views (Front, Top, Right)
- **Intelligent Dimensioning**: Automatically places 3-5+ critical dimensions with proper spacing
- **Hole Detection**: Identifies and dimensions circular features
- **Title Block Integration**: Populates metadata based on part analysis
- **Multiple Output Formats**: Generates FreeCAD (.FCStd), PDF, and PNG files

## Requirements

- Python 3.8+
- FreeCAD 0.19+ (with Python API)
- NumPy >= 1.20.0
- Matplotlib >= 3.4.0

## Installation

```bash
# Clone or download the repository
git clone https://github.com/your-org/ai-drafting-agent.git
cd ai-drafting-agent

# Install Python dependencies
pip install numpy matplotlib

freecadcmd main.py <input.step> <output_name>

## How to use

# Process a STEP file
freecadcmd main.py bracket.step bracket_drawing

# Process an IGES file  
freecadcmd main.py housing.iges housing_drawing

# Output files created:
#   - bracket_drawing.FCStd  (FreeCAD document)
#   - bracket_drawing.pdf    (Technical drawing)
#   - bracket_drawing.png    (Preview image)


## how to agent

from main import AIDraftingAgent

agent = AIDraftingAgent()
results = agent.run("input.step", "output_drawing")

print(f"Created {results['dimensions_techdraw']} TechDraw dimensions")
print(f"Created {results['dimensions_pdf']} PDF dimensions")

##expected output files

Output Files
File	Description
{name}.FCStd	FreeCAD document with TechDraw page, views, and dimensions
{name}.pdf	High-resolution PDF technical drawing (300 DPI)
{name}.png	Preview image (150 DPI)

## project strructure
ai_drafting_agent/
├── README.md        # This file - documentation and architecture
├── main.py          # Entry point, CLI, and main Agent class
├── core.py          # Data structures, geometry analysis, utilities
└── generators.py    # TechDraw and PDF rendering engines

##Module Descriptions
Module	Purpose
main.py	CLI parsing, validation, orchestrates the complete pipeline
core.py	Data models, geometry analyzer, title block generator, config
generators.py	TechDraw dimension strategies, view projection, PDF rendering

### pipleline

┌─────────────────────────────────────────────────────────────────┐
│                        INPUT STAGE                               │
├─────────────────────────────────────────────────────────────────┤
│  STEP/IGES/FCStd  ──▶  FreeCAD Import  ──▶  Shape Extraction   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ANALYSIS STAGE                              │
├─────────────────────────────────────────────────────────────────┤
│  Bounding Box  ──▶  Hole Detection  ──▶  Edge Classification   │
│       │                    │                     │               │
│       ▼                    ▼                     ▼               │
│  Scale Selection    Diameter List         Direction Map         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GENERATION STAGE                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  TechDraw    │    │    Views     │    │  Dimensions  │      │
│  │    Page      │───▶│  Front/Top/  │───▶│  Vertex or   │      │
│  │  + Template  │    │    Right     │    │  Edge-based  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OUTPUT STAGE                               │
├─────────────────────────────────────────────────────────────────┤
│  FCStd Document  ◀──  PDF Renderer  ◀──  View Projector        │
│        │                   │                    │                │
│        ▼                   ▼                    ▼                │
│   .FCStd file          .pdf file           .png file            │
└─────────────────────────────────────────────────────────────────┘

##
