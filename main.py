#!/usr/bin/env python3
"""
AI Drafting Agent - Main Entry Point
=====================================

Automated technical drawing generation from 3D CAD models.

This module provides:
    - Command-line interface
    - Main AIDraftingAgent orchestrator class
    - Input validation and error handling
    - Dependency checking

Usage:
    freecadcmd main.py <input.step> <output_name>

Example:
    freecadcmd main.py bracket.step bracket_drawing
    
    Outputs:
        bracket_drawing.FCStd  - FreeCAD document
        bracket_drawing.pdf    - Technical drawing PDF
        bracket_drawing.png    - Preview image

Author: AI Engineering Team
Version: 1.0.0
"""

import sys
import os

__version__ = "1.0.0"
__author__ = "AI Engineering Team"


# =============================================================================
# BANNER AND STARTUP
# =============================================================================

def print_banner():
    """Print application banner."""
    print("=" * 70)
    print("  AI DRAFTING AGENT - Automated Technical Drawing Generation")
    print(f"  Version {__version__}")
    print("=" * 70)


def check_dependencies():
    """
    Check and import required dependencies.
    
    Validates that FreeCAD, Part, TechDraw, matplotlib, and numpy
    are available for import.
    
    Returns:
        tuple: (FreeCAD, Part, TechDraw) modules
        
    Raises:
        SystemExit: If required modules are not available
    """
    print("\n[CHECKING DEPENDENCIES]")
    
    # Check FreeCAD
    try:
        import FreeCAD
        version = f"{FreeCAD.Version()[0]}.{FreeCAD.Version()[1]}"
        print(f"  ✓ FreeCAD {version}")
    except ImportError as e:
        print(f"  ✗ FreeCAD not found: {e}")
        print("    Run this script using: freecadcmd main.py <args>")
        sys.exit(1)
    
    # Check Part module
    try:
        import Part
        print("  ✓ Part module")
    except ImportError:
        print("  ✗ Part module not found")
        sys.exit(1)
    
    # Check TechDraw module
    try:
        import TechDraw
        print("  ✓ TechDraw module")
    except ImportError:
        print("  ✗ TechDraw module not found")
        sys.exit(1)
    
    # Check matplotlib
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        print("  ✓ Matplotlib")
    except ImportError:
        print("  ✗ Matplotlib not found - PDF output will be disabled")
    
    # Check numpy
    try:
        import numpy
        print("  ✓ NumPy")
    except ImportError:
        print("  ✗ NumPy not found")
        sys.exit(1)
    
    return FreeCAD, Part, TechDraw


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments(args: list) -> tuple:
    """
    Parse command line arguments.
    
    Handles FreeCAD's argument passing conventions where arguments
    appear after the script name.
    
    Args:
        args: List of command line arguments (sys.argv)
        
    Returns:
        tuple: (input_file, output_base) paths
        
    Raises:
        SystemExit: If arguments are invalid or missing
    """
    # Find arguments after the Python script
    script_args = []
    
    for i, arg in enumerate(args):
        if arg.endswith('.py'):
            script_args = args[i + 1:]
            break
    
    # Check for help flag
    if '-h' in script_args or '--help' in script_args:
        print_usage()
        sys.exit(0)
    
    # Validate argument count
    if len(script_args) < 2:
        print_usage()
        sys.exit(1)
    
    input_file = os.path.abspath(script_args[0])
    output_base = os.path.abspath(script_args[1])
    
    return input_file, output_base


def print_usage():
    """Print usage information."""
    usage = """
Usage: freecadcmd main.py <input_file> <output_name>

Arguments:
    input_file   : Path to CAD file (STEP, IGES, or FCStd)
    output_name  : Base name for output files (without extension)

Options:
    -h, --help   : Show this help message

Examples:
    freecadcmd main.py bracket.step bracket_drawing
    freecadcmd main.py housing.iges housing_drawing
    freecadcmd main.py assembly.fcstd assembly_drawing

Output Files:
    {output_name}.FCStd  - FreeCAD document with TechDraw page
    {output_name}.pdf    - Technical drawing PDF
    {output_name}.png    - Preview image
"""
    print(usage)


def validate_input(input_file: str) -> None:
    """
    Validate input file exists and has supported extension.
    
    Args:
        input_file: Path to input CAD file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    # Check file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Check extension
    supported_extensions = {'.step', '.stp', '.iges', '.igs', '.fcstd'}
    ext = os.path.splitext(input_file)[1].lower()
    
    if ext not in supported_extensions:
        raise ValueError(
            f"Unsupported file format: {ext}\n"
            f"Supported formats: {', '.join(sorted(supported_extensions))}"
        )


# =============================================================================
# MAIN AGENT CLASS
# =============================================================================

class AIDraftingAgent:
    """
    Main orchestrator for the AI Drafting Agent system.
    
    Coordinates the complete pipeline from CAD import through
    technical drawing generation.
    
    Pipeline Steps:
        1. Import CAD file into FreeCAD
        2. Analyze geometry (bounding box, holes, edges)
        3. Generate metadata (title block, scale selection)
        4. Create TechDraw views and dimensions
        5. Render PDF/PNG outputs
    
    Attributes:
        config (Config): Configuration object
        doc: FreeCAD document (set during run)
        part: Part object (set during run)
        analysis (AnalysisResult): Analysis results (set during run)
        title_data (TitleBlockData): Title block data (set during run)
        
    Example:
        >>> agent = AIDraftingAgent()
        >>> results = agent.run("input.step", "output")
        >>> print(f"Created {results['dimensions_techdraw']} dimensions")
    """
    
    def __init__(self, config=None):
        """
        Initialize the drafting agent.
        
        Args:
            config: Optional Config object. Uses defaults if None.
        """
        # Lazy import to avoid issues when module is imported
        from core import Config
        
        self.config = config or Config()
        self.FreeCAD = None
        self.Part = None
        self.TechDraw = None
        self.doc = None
        self.part = None
        self.analysis = None
        self.title_data = None
    
    def run(self, input_file: str, output_base: str) -> dict:
        """
        Execute the complete drafting pipeline.
        
        Args:
            input_file: Path to input CAD file
            output_base: Base path for output files (without extension)
            
        Returns:
            dict: Results dictionary containing:
                - input: Input file path
                - output_files: List of generated file paths
                - dimensions_techdraw: Number of TechDraw dimensions
                - dimensions_pdf: Number of PDF dimensions
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If no valid shape found in file
        """
        # Import FreeCAD modules
        import FreeCAD
        import Part
        import TechDraw
        
        self.FreeCAD = FreeCAD
        self.Part = Part
        self.TechDraw = TechDraw
        
        print("\n" + "=" * 60)
        print("  PROCESSING PIPELINE")
        print("=" * 60)
        
        # Initialize results
        results = {
            'input': input_file,
            'output_files': [],
            'dimensions_techdraw': 0,
            'dimensions_pdf': 0,
        }
        
        # Step 1: Import CAD file
        self._import_cad(input_file)
        
        # Step 2: Analyze geometry
        self._analyze_geometry()
        
        # Step 3: Generate metadata
        self._generate_metadata(input_file)
        
        # Step 4: Create TechDraw
        techdraw_dims = self._create_techdraw(output_base)
        results['dimensions_techdraw'] = techdraw_dims
        results['output_files'].append(f"{output_base}.FCStd")
        
        # Step 5: Render PDF
        pdf_dims = self._render_pdf(output_base)
        results['dimensions_pdf'] = pdf_dims
        results['output_files'].extend([
            f"{output_base}.pdf",
            f"{output_base}.png"
        ])
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _import_cad(self, input_file: str) -> None:
        """
        Import CAD file into FreeCAD.
        
        Creates a new document and imports the specified file.
        Locates the first object with valid 3D geometry.
        
        Args:
            input_file: Path to CAD file
            
        Raises:
            ValueError: If no valid 3D shape found
        """
        print("\n[STEP 1: IMPORT]")
        
        # Create new document
        self.doc = self.FreeCAD.newDocument("Drawing")
        
        # Import the file
        self.Part.insert(input_file, self.doc.Name)
        
        # Find the part object with valid geometry
        self.part = None
        for obj in self.doc.Objects:
            if hasattr(obj, 'Shape') and obj.Shape.Volume > 0:
                self.part = obj
                break
        
        if self.part is None:
            raise ValueError("No valid 3D shape found in file")
        
        print(f"  ✓ File: {os.path.basename(input_file)}")
        print(f"  ✓ Part: {self.part.Name}")
        print(f"  ✓ Type: {self.part.TypeId}")
    
    def _analyze_geometry(self) -> None:
        """
        Analyze the imported geometry.
        
        Extracts bounding box dimensions, detects holes,
        and classifies edges by direction.
        """
        from core import GeometryAnalyzer
        
        analyzer = GeometryAnalyzer(self.part.Shape, self.config)
        self.analysis = analyzer.analyze()
    
    def _generate_metadata(self, input_file: str) -> None:
        """
        Generate title block metadata.
        
        Creates part name, number, scale, and other metadata
        based on filename and geometry analysis.
        
        Args:
            input_file: Path to input file (for naming)
        """
        from core import TitleBlockGenerator
        
        print("\n[STEP 3: METADATA]")
        
        generator = TitleBlockGenerator(self.config)
        self.title_data = generator.generate(input_file, self.analysis)
        
        print(f"  ✓ Part Name: {self.title_data.part_name}")
        print(f"  ✓ Part Number: {self.title_data.part_number}")
        print(f"  ✓ Scale: {self.title_data.scale}")
        print(f"  ✓ Weight: {self.title_data.weight}")
    
    def _create_techdraw(self, output_base: str) -> int:
        """
        Create TechDraw document with views and dimensions.
        
        Tries vertex-based dimensioning first, then falls back
        to simpler methods if needed.
        
        Args:
            output_base: Base path for output file
            
        Returns:
            int: Number of dimensions created
        """
        from generators import TechDrawGeneratorVertex, TechDrawSimpleDirect
        
        # Try vertex-based method first (most reliable)
        generator = TechDrawGeneratorVertex(
            self.doc, self.part, self.analysis,
            self.title_data, self.config,
            self.FreeCAD, self.TechDraw
        )
        generator.generate()
        dim_count = generator.dim_count
        
        # Fallback to simple method if no dimensions created
        if dim_count == 0:
            print("\n  Retrying with fallback method...")
            fallback = TechDrawSimpleDirect(
                self.doc, self.part, self.analysis,
                self.title_data, self.config,
                self.FreeCAD, self.TechDraw
            )
            fallback.generate()
            dim_count = fallback.dim_count
        
        # Save the FreeCAD document
        generator.save(output_base)
        
        return dim_count
    
    def _render_pdf(self, output_base: str) -> int:
        """
        Render PDF output with projected views and dimensions.
        
        Uses matplotlib to create publication-quality PDF
        with orthographic views and dimensions.
        
        Args:
            output_base: Base path for output files
            
        Returns:
            int: Number of dimensions rendered
        """
        from generators import ViewProjector, PDFRenderer
        
        print("\n[STEP 5: PDF RENDERING]")
        
        # Create projector and renderer
        projector = ViewProjector(self.part.Shape, self.config)
        renderer = PDFRenderer(self.analysis, self.title_data, self.config)
        
        # Project standard views
        for view_name in ['front', 'top']:
            view = projector.project(view_name)
            renderer.add_view(view)
            print(f"  ✓ Projected {view_name}: {len(view.edges)} edges")
        
        # Right view includes circle detection for holes
        view = projector.project('right', include_circles=True)
        renderer.add_view(view)
        print(f"  ✓ Projected right: {len(view.edges)} edges, {len(view.circles)} circles")
        
        # Render to files
        renderer.render(output_base)
        
        return renderer.dim_count
    
    def _print_summary(self, results: dict) -> None:
        """
        Print processing summary.
        
        Args:
            results: Results dictionary from processing
        """
        print("\n" + "=" * 60)
        print("  PROCESSING COMPLETE")
        print("=" * 60)
        
        bbox = self.analysis.bbox
        
        print(f"""
  Part Information:
    Name: {self.title_data.part_name}
    Number: {self.title_data.part_number}
    Dimensions: {bbox['dx']:.1f} x {bbox['dy']:.1f} x {bbox['dz']:.1f} mm
    Weight: {self.title_data.weight}
    Scale: {self.title_data.scale}
  
  Results:
    TechDraw Dimensions: {results['dimensions_techdraw']}
    PDF Dimensions: {results['dimensions_pdf']}
    Holes Detected: {self.analysis.hole_count}
  
  Output Files:""")
        
        for filepath in results['output_files']:
            exists = "✓" if os.path.exists(filepath) else "○"
            print(f"    {exists} {filepath}")
        
        print()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for the AI Drafting Agent.
    
    Orchestrates the complete workflow:
    1. Print banner
    2. Check dependencies
    3. Parse and validate arguments
    4. Run the drafting agent
    5. Handle errors gracefully
    """
    print_banner()
    
    try:
        # Check dependencies first
        check_dependencies()
        
        # Parse and validate arguments
        input_file, output_base = parse_arguments(sys.argv)
        validate_input(input_file)
        
        print(f"\n  Input:  {input_file}")
        print(f"  Output: {output_base}.*")
        
        # Create and run the agent
        agent = AIDraftingAgent()
        results = agent.run(input_file, output_base)
        
        # Exit with success
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"\n✗ File Error: {e}")
        sys.exit(1)
        
    except ValueError as e:
        print(f"\n✗ Validation Error: {e}")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n✗ Operation cancelled by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n✗ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# =============================================================================
# MODULE EXECUTION
# =============================================================================

if __name__ == '__main__':
    main()