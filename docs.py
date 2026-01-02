"""
================================================================================
AI DRAFTING AGENT - DOCUMENTATION & ARCHITECTURE
================================================================================

An intelligent automated system for generating professional 2D technical 
drawings from 3D CAD models using FreeCAD's TechDraw workbench.

================================================================================
OVERVIEW
================================================================================

The AI Drafting Agent bridges the gap between 3D CAD geometry and 
manufacturing-ready 2D documentation by programmatically interpreting 3D data 
and generating professional technical drawings.

Key Features:
    - Automatic View Generation: Creates optimal orthographic views (Front, Top, Right)
    - Intelligent Dimensioning: Automatically places 3-5+ critical dimensions
    - Hole Detection: Identifies and dimensions circular features
    - Title Block Integration: Populates metadata based on part analysis
    - Multiple Output Formats: Generates FreeCAD (.FCStd), PDF, and PNG files

================================================================================
REQUIREMENTS
================================================================================

- Python 3.8+
- FreeCAD 0.19+ (with Python API)
- NumPy >= 1.20.0
- Matplotlib >= 3.4.0

================================================================================
INSTALLATION
================================================================================

    # Clone or download the repository
    git clone https://github.com/your-org/ai-drafting-agent.git
    cd ai-drafting-agent

    # Install Python dependencies
    pip install numpy matplotlib

================================================================================
USAGE
================================================================================

Command Line (via FreeCAD):
    
    freecadcmd main.py <input.step> <output_name>

Examples:

    # Process a STEP file
    freecadcmd main.py bracket.step bracket_drawing

    # Process an IGES file  
    freecadcmd main.py housing.iges housing_drawing

    # Output files created:
    #   - bracket_drawing.FCStd  (FreeCAD document)
    #   - bracket_drawing.pdf    (Technical drawing)
    #   - bracket_drawing.png    (Preview image)

Python API:

    from main import AIDraftingAgent

    agent = AIDraftingAgent()
    results = agent.run("input.step", "output_drawing")

    print(f"Created {results['dimensions_techdraw']} TechDraw dimensions")
    print(f"Created {results['dimensions_pdf']} PDF dimensions")

================================================================================
PROJECT STRUCTURE
================================================================================

    ai_drafting_agent/
    ├── docs.py          # This file - documentation and architecture
    ├── main.py          # Entry point, CLI, and main Agent class
    ├── core.py          # Data structures, geometry analysis, utilities
    └── generators.py    # TechDraw generators, view projection, PDF rendering

Module Descriptions:

    docs.py       : Documentation, architecture notes, usage examples
    main.py       : CLI parsing, validation, orchestrates the complete pipeline
    core.py       : Config, data models, GeometryAnalyzer, TitleBlockGenerator
    generators.py : TechDraw dimension strategies, ViewProjector, PDFRenderer

================================================================================
SYSTEM ARCHITECTURE
================================================================================

1. PROBLEM SOLVED
-----------------
This system automates the conversion of 3D CAD models into professional 2D 
technical drawings, addressing the labor-intensive manual drafting process 
in manufacturing workflows.

2. ARCHITECTURAL APPROACH
-------------------------

2.1 Hybrid Heuristic/Rule-Based System

The agent employs a geometric reasoning engine that applies manufacturing logic:

    - Bounding Box Analysis: Determines optimal scale and view positioning
    - Feature Detection: Identifies holes, cylindrical surfaces, critical edges
    - Edge Classification: Categorizes edges by orientation (X, Y, Z)
    - Multi-Strategy Dimensioning: Falls back through multiple approaches

2.2 Spatial Readability Strategy

To prevent cluttered drawings, the system implements:

    - Dimension Offset Management: Systematic spacing of dimension lines
    - View-Specific Logic: Different dimensioning strategies per view
    - Deduplication: Prevents redundant dimensions across views
    - Validation Pipeline: Tests dimension validity before final placement

3. PROCESSING PIPELINE
----------------------

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
    │  TechDraw Page  ──▶  Views (F/T/R)  ──▶  Dimensions            │
    └─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                       OUTPUT STAGE                               │
    ├─────────────────────────────────────────────────────────────────┤
    │  .FCStd file  ◀──  PDF Renderer  ◀──  View Projector           │
    │  .pdf file                                                       │
    │  .png file                                                       │
    └─────────────────────────────────────────────────────────────────┘

4. DIMENSION PLACEMENT ALGORITHM
--------------------------------

The system uses a multi-strategy fallback approach:

    Strategy 1: Vertex-Based Method
    ───────────────────────────────
    • Find vertex pairs matching bounding box dimensions
    • Use DistanceX/DistanceY dimensions
    • Most reliable for overall dimensions
            │
            │ If fails
            ▼
    Strategy 2: Edge-Based Method
    ─────────────────────────────
    • Map 3D edges to TechDraw indices
    • Use Distance dimensions on edges
    • Good for specific features
            │
            │ If fails
            ▼
    Strategy 3: Direct Enumeration
    ──────────────────────────────
    • Try edges 0, 1, 2... sequentially
    • Maximum compatibility fallback
    • Always produces some dimensions

5. KEY DESIGN DECISIONS
-----------------------

    Decision              | Rationale
    ──────────────────────|─────────────────────────────────────────
    FreeCAD TechDraw      | Industry-standard output, native STEP support
    Matplotlib PDF        | Cross-platform fallback, publication quality
    Multiple strategies   | Robustness across different part geometries
    Modular architecture  | Maintainability, testing, extensibility
    Dataclass models      | Type safety, clean data containers

6. VIEW CONFIGURATION
---------------------

    View   | Direction    | Shows              | Typical Dimensions
    ───────|──────────────|────────────────────|────────────────────
    Front  | (0, -1, 0)   | X width, Z height  | Overall W × H
    Top    | (0, 0, 1)    | X width, Y depth   | Overall W × D
    Right  | (1, 0, 0)    | Y depth, Z height  | D × H, Hole diameters

7. EXTENSIBILITY POINTS
-----------------------

    - Custom View Configurations: Modify Config.VIEWS dictionary
    - New Dimension Strategies: Subclass BaseTechDrawGenerator
    - Template Customization: Add paths to Config.TEMPLATE_NAMES
    - Material Database: Extend TitleBlockGenerator

8. LIMITATIONS & FUTURE WORK
----------------------------

Current Limitations:
    - Section views not yet implemented
    - No GD&T (Geometric Dimensioning & Tolerancing)
    - Limited to orthographic projections

Planned Enhancements:
    - AI-based feature recognition using Vision LLMs
    - Automatic section view generation for complex internals
    - GD&T annotation support
    - BOM (Bill of Materials) generation

================================================================================
LICENSE
================================================================================

MIT License - See LICENSE file for details.

================================================================================
"""

__version__ = "1.0.0"
__author__ = "AI Engineering Team"
__doc_version__ = "2024.1"


def print_help():
    """Print usage help to console."""
    help_text = """
    AI Drafting Agent - Automated Technical Drawing Generation
    ==========================================================
    
    Usage:
        freecadcmd main.py <input_file> <output_name>
    
    Arguments:
        input_file   : Path to CAD file (STEP, IGES, or FCStd)
        output_name  : Base name for output files (no extension)
    
    Examples:
        freecadcmd main.py bracket.step bracket_drawing
        freecadcmd main.py housing.iges housing_drawing
    
    Output Files:
        {output_name}.FCStd  - FreeCAD document with TechDraw page
        {output_name}.pdf    - Technical drawing PDF (300 DPI)
        {output_name}.png    - Preview image (150 DPI)
    
    For more information, see the module docstring:
        python -c "import docs; print(docs.__doc__)"
    """
    print(help_text)


def print_architecture():
    """Print architecture summary to console."""
    arch_text = """
    AI Drafting Agent - Architecture Summary
    ========================================
    
    Processing Pipeline:
    
        1. INPUT     : Import STEP/IGES/FCStd file via FreeCAD
        2. ANALYSIS  : Extract bounding box, detect holes, classify edges
        3. METADATA  : Generate title block data, select scale
        4. TECHDRAW  : Create page, views, and dimensions
        5. OUTPUT    : Save FCStd, render PDF and PNG
    
    Dimensioning Strategy:
    
        Primary   : Vertex-based (DistanceX/Y between vertex pairs)
        Fallback  : Edge-based (Distance on single edges)
        Last      : Direct enumeration (edges 0, 1, 2...)
    
    Key Components:
    
        - AIDraftingAgent    : Main orchestrator (main.py)
        - GeometryAnalyzer   : 3D analysis (core.py)
        - TechDrawGenerator  : FreeCAD output (generators.py)
        - PDFRenderer        : Matplotlib output (generators.py)
    """
    print(arch_text)


if __name__ == '__main__':
    print(__doc__)