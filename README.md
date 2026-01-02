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

- Python 3.11
- Freecad 1.0.0
- Conda

## Installation

```bash
# Clone or download the repository
git clone https://github.com/your-org/ai-drafting-agent.git
cd ai-drafting-agent

# Install Python dependencies
conda env create -f environment.yml
conda activate ai_drafting_agent

# To run the main code:
'''

freecadcmd main.py <input.step> <output_name>
'''

## How to use

# Process a STEP file

freecadcmd main.py bracket.step bracket_drawing

# Process an IGES file

freecadcmd main.py housing.iges housing_drawing

# Output files created:
#   - bracket_drawing.FCStd  (FreeCAD document)
#   - bracket_drawing.pdf    (Technical drawing)
#   - bracket_drawing.png    (Preview image)



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
├── machine_part.STEP  # input data
└──environment.yml   # for installations



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
