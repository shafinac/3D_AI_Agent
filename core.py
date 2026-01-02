"""
AI Drafting Agent - Core Module
================================

This module contains foundational components:
    - Configuration management (Config, ViewConfig)
    - Data structures (AnalysisResult, TitleBlockData, ProjectedView, EdgeInfo)
    - Geometry analysis (GeometryAnalyzer)
    - Title block generation (TitleBlockGenerator)
    - Utility functions (to_float, find_template)

All core functionality that doesn't depend on specific 
output generation (TechDraw or PDF).

Author: AI Engineering Team
Version: 1.0.0
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

__version__ = "1.0.0"
__author__ = "AI Engineering Team"


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class ViewConfig:
    """
    Configuration for a single orthographic view.
    
    Defines how a view is projected and positioned on the drawing page.
    
    Attributes:
        name (str): View identifier ('front', 'top', 'right')
        direction (Tuple[float, float, float]): View direction vector (x, y, z)
        x_direction (Tuple[float, float, float]): X-axis direction for the view
        position (Tuple[float, float]): Position on page (x, y) in mm
        include_circles (bool): Whether to detect circular features in this view
        
    Example:
        >>> front = ViewConfig(
        ...     name='front',
        ...     direction=(0, -1, 0),
        ...     x_direction=(1, 0, 0),
        ...     position=(105, 170)
        ... )
    """
    name: str
    direction: Tuple[float, float, float]
    x_direction: Tuple[float, float, float]
    position: Tuple[float, float]
    include_circles: bool = False


@dataclass
class Config:
    """
    Master configuration for the AI Drafting Agent.
    
    Centralizes all configurable parameters for easy customization.
    Uses sensible defaults that work for most mechanical parts.
    
    Attributes:
        VIEWS: Dictionary of view configurations
        TEMPLATE_NAMES: List of template filenames to search for
        TEMPLATE_PATHS: List of template directory paths
        DEFAULT_MATERIAL: Default material specification string
        MATERIAL_DENSITY: Density for weight calculation (kg/mm³)
        DIMENSION_TOLERANCE: Tolerance for matching expected dimensions
        MAX_SEARCH_INDICES: Maximum edge/vertex indices to search
        SCALE_THRESHOLDS: Automatic scale selection thresholds
        RECOMPUTE_ITERATIONS: Number of recompute cycles for TechDraw
        COMPUTE_DELAY: Delay after recompute (seconds)
        EDGE_DISCRETIZATION: Deflection for edge discretization
        DIMENSION_FORMAT: Format string for linear dimensions
        DIAMETER_FORMAT: Format string for diameter dimensions
        OFFSET_HORIZONTAL: Default horizontal dimension offset
        OFFSET_VERTICAL: Default vertical dimension offset
        PDF_DPI: Resolution for PDF output
        PNG_DPI: Resolution for PNG output
        
    Example:
        >>> config = Config()
        >>> config.DIMENSION_TOLERANCE = 0.5  # Tighter tolerance
        >>> config.DEFAULT_MATERIAL = "STEEL 1018"
    """
    
    # View configurations - standard orthographic views
    VIEWS: Dict[str, ViewConfig] = field(default_factory=lambda: {
        'front': ViewConfig(
            name='front',
            direction=(0, -1, 0),      # Looking from -Y direction
            x_direction=(1, 0, 0),     # X axis points right
            position=(105, 170),       # Position on A4 landscape
            include_circles=False
        ),
        'top': ViewConfig(
            name='top',
            direction=(0, 0, 1),       # Looking from +Z direction
            x_direction=(1, 0, 0),     # X axis points right
            position=(105, 75),        # Below front view
            include_circles=False
        ),
        'right': ViewConfig(
            name='right',
            direction=(1, 0, 0),       # Looking from +X direction
            x_direction=(0, 1, 0),     # Y axis points right
            position=(230, 170),       # Right of front view
            include_circles=True       # Detect holes in this view
        ),
    })
    
    # Template settings
    TEMPLATE_NAMES: List[str] = field(default_factory=lambda: [
        "A4_LandscapeTD.svg",
        "A3_LandscapeTD.svg",
        "A4_Landscape_blank.svg",
    ])
    
    TEMPLATE_PATHS: List[str] = field(default_factory=lambda: [
        "Mod/TechDraw/Templates",
        "share/Mod/TechDraw/Templates",
        "data/Mod/TechDraw/Templates",
    ])
    
    # Material defaults (Aluminum 6061-T6)
    DEFAULT_MATERIAL: str = "ALUMINUM 6061-T6"
    MATERIAL_DENSITY: float = 2.7e-6  # kg/mm³
    
    # Dimension search settings
    DIMENSION_TOLERANCE: float = 1.0      # mm tolerance for matching
    MAX_SEARCH_INDICES: int = 50          # Max edges/vertices to search
    
    # Dimension formatting
    DIMENSION_FORMAT: str = "%.1f"        # One decimal place
    DIAMETER_FORMAT: str = "%%c%.1f"      # Diameter symbol + value
    OFFSET_HORIZONTAL: float = 15.0       # Horizontal dim offset
    OFFSET_VERTICAL: float = 20.0         # Vertical dim offset
    
    # Scale thresholds: (max_dimension_mm, scale_string, scale_factor)
    SCALE_THRESHOLDS: List[Tuple[float, str, float]] = field(default_factory=lambda: [
        (100, "1:1", 1.0),
        (200, "1:2", 0.5),
        (500, "1:5", 0.2),
        (1000, "1:10", 0.1),
    ])
    
    # Processing settings
    RECOMPUTE_ITERATIONS: int = 10        # Recompute cycles
    COMPUTE_DELAY: float = 0.5            # Seconds to wait
    EDGE_DISCRETIZATION: float = 0.3      # Deflection for discretize()
    
    # Output settings
    PDF_DPI: int = 300
    PNG_DPI: int = 150
    
    def get_scale(self, max_dimension: float) -> Tuple[str, float]:
        """
        Determine appropriate scale for given dimension.
        
        Selects scale to fit part comfortably on drawing page.
        
        Args:
            max_dimension: Maximum bounding box dimension in mm
            
        Returns:
            Tuple of (scale_string, scale_factor)
            
        Example:
            >>> config = Config()
            >>> config.get_scale(150)
            ('1:2', 0.5)
        """
        for threshold, scale_str, factor in self.SCALE_THRESHOLDS:
            if max_dimension <= threshold:
                return scale_str, factor
        return "1:10", 0.1


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EdgeInfo:
    """
    Information about a 3D edge.
    
    Stores geometric properties of an edge for analysis
    and dimension placement.
    
    Attributes:
        index (int): Edge index in shape.Edges list
        length (float): Edge length in mm
        direction (str): Primary direction ('X', 'Y', 'Z', 'OTHER')
        start_point (Tuple): Start vertex position (x, y, z)
        end_point (Tuple): End vertex position (x, y, z)
        is_circular (bool): Whether edge is circular/arc
    """
    index: int
    length: float
    direction: str
    start_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    end_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_circular: bool = False


@dataclass
class AnalysisResult:
    """
    Complete geometry analysis results.
    
    Contains all extracted information about the 3D part
    needed for technical drawing generation.
    
    Attributes:
        bbox (Dict[str, float]): Bounding box with dx, dy, dz, max_dim, center
        volume (float): Part volume in mm³
        surface_area (float): Total surface area in mm²
        face_count (int): Number of faces
        edge_count (int): Number of edges
        hole_count (int): Number of detected cylindrical holes
        hole_diameters (List[float]): List of unique hole diameters
        hole_positions (Dict): Mapping of diameter to list of center positions
        edges (List[EdgeInfo]): Classified edge information
        
    Properties:
        has_holes: True if part has detected holes
        is_complex: True if part has many features
    """
    bbox: Dict[str, float] = field(default_factory=dict)
    volume: float = 0.0
    surface_area: float = 0.0
    face_count: int = 0
    edge_count: int = 0
    hole_count: int = 0
    hole_diameters: List[float] = field(default_factory=list)
    hole_positions: Dict[float, List[Tuple]] = field(default_factory=dict)
    edges: List[EdgeInfo] = field(default_factory=list)
    
    @property
    def has_holes(self) -> bool:
        """Check if part has detected holes."""
        return self.hole_count > 0
    
    @property
    def is_complex(self) -> bool:
        """Heuristic check for part complexity."""
        return self.face_count > 20 or self.hole_count > 3
    
    @property
    def dimensions_tuple(self) -> Tuple[float, float, float]:
        """Get dimensions as tuple (dx, dy, dz)."""
        return (
            self.bbox.get('dx', 0),
            self.bbox.get('dy', 0),
            self.bbox.get('dz', 0)
        )


@dataclass
class TitleBlockData:
    """
    Data for populating technical drawing title block.
    
    Contains all metadata fields for the drawing title block.
    
    Attributes:
        part_name (str): Display name of the part
        part_number (str): Part number/ID
        material (str): Material specification
        weight (str): Calculated weight as formatted string
        scale (str): Scale string (e.g., "1:2")
        scale_factor (float): Numeric scale factor (e.g., 0.5)
        date (str): Drawing date (YYYY-MM-DD)
        drawn_by (str): Author/creator
        dimensions_str (str): Formatted overall dimensions
        revision (str): Revision letter/number
    """
    part_name: str = "UNKNOWN"
    part_number: str = "XXX-001"
    material: str = "ALUMINUM 6061-T6"
    weight: str = "0.000 kg"
    scale: str = "1:1"
    scale_factor: float = 1.0
    date: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d'))
    drawn_by: str = "AI Agent"
    dimensions_str: str = "0 x 0 x 0"
    revision: str = "A"


@dataclass
class ProjectedView:
    """
    A 2D projected view of the 3D part.
    
    Contains geometry data for rendering orthographic projections
    in the PDF output.
    
    Attributes:
        name (str): View name ('front', 'top', 'right')
        edges (List[np.ndarray]): List of edge polylines as Nx2 arrays
        circles (List): List of ((center_x, center_y), radius) tuples
        bounds (Tuple[float, float]): View bounds as (width, height)
        
    Properties:
        width: View width
        height: View height
    """
    name: str
    edges: List[np.ndarray] = field(default_factory=list)
    circles: List[Tuple[Tuple[float, float], float]] = field(default_factory=list)
    bounds: Tuple[float, float] = (0.0, 0.0)
    
    @property
    def width(self) -> float:
        """View width."""
        return self.bounds[0]
    
    @property
    def height(self) -> float:
        """View height."""
        return self.bounds[1]


# =============================================================================
# GEOMETRY ANALYZER
# =============================================================================

class GeometryAnalyzer:
    """
    Analyzes 3D shape geometry for technical drawing generation.
    
    Extracts dimensional information, detects features like holes,
    and classifies edges by their primary direction.
    
    Attributes:
        shape: FreeCAD Part.Shape object
        config (Config): Configuration object
        
    Example:
        >>> analyzer = GeometryAnalyzer(part.Shape)
        >>> result = analyzer.analyze()
        >>> print(f"Size: {result.bbox['dx']} x {result.bbox['dy']}")
        >>> print(f"Holes: {result.hole_diameters}")
    """
    
    def __init__(self, shape: Any, config: Optional[Config] = None):
        """
        Initialize analyzer.
        
        Args:
            shape: FreeCAD Part.Shape object
            config: Optional configuration object
        """
        self.shape = shape
        self.config = config or Config()
    
    def analyze(self) -> AnalysisResult:
        """
        Perform complete geometry analysis.
        
        Extracts all geometric information needed for drawing generation.
        
        Returns:
            AnalysisResult with bounding box, holes, edges, etc.
        """
        print("\n[STEP 2: GEOMETRY ANALYSIS]")
        
        result = AnalysisResult()
        
        # Extract bounding box
        result.bbox = self._analyze_bounding_box()
        
        # Basic properties
        result.volume = self.shape.Volume
        result.surface_area = self.shape.Area
        result.face_count = len(self.shape.Faces)
        result.edge_count = len(self.shape.Edges)
        
        # Print basic info
        bbox = result.bbox
        print(f"  ✓ Bounding Box: {bbox['dx']:.1f} x {bbox['dy']:.1f} x {bbox['dz']:.1f} mm")
        print(f"  ✓ Volume: {result.volume:.1f} mm³")
        print(f"  ✓ Surface Area: {result.surface_area:.1f} mm²")
        print(f"  ✓ Topology: {result.face_count} faces, {result.edge_count} edges")
        
        # Detect holes (cylindrical features)
        self._detect_holes(result)
        
        # Classify edges by direction
        result.edges = self._classify_edges()
        
        return result
    
    def _analyze_bounding_box(self) -> Dict[str, float]:
        """
        Extract bounding box information.
        
        Returns:
            Dictionary with dx, dy, dz, max_dim, and center
        """
        bbox = self.shape.BoundBox
        
        return {
            'dx': bbox.XLength,
            'dy': bbox.YLength,
            'dz': bbox.ZLength,
            'max_dim': max(bbox.XLength, bbox.YLength, bbox.ZLength),
            'min_dim': min(bbox.XLength, bbox.YLength, bbox.ZLength),
            'center': (
                (bbox.XMin + bbox.XMax) / 2,
                (bbox.YMin + bbox.YMax) / 2,
                (bbox.ZMin + bbox.ZMax) / 2
            ),
            'min': (bbox.XMin, bbox.YMin, bbox.ZMin),
            'max': (bbox.XMax, bbox.YMax, bbox.ZMax),
        }
    
    def _detect_holes(self, result: AnalysisResult) -> None:
        """
        Detect cylindrical holes in the part.
        
        Analyzes faces for cylindrical surfaces and extracts
        diameter and position information.
        
        Args:
            result: AnalysisResult to populate with hole data
        """
        detected_diameters = set()
        
        for face in self.shape.Faces:
            try:
                surface = face.Surface
                surface_type = getattr(surface, 'TypeId', str(type(surface)))
                
                if 'Cylinder' in surface_type:
                    # Get diameter and center
                    diameter = round(surface.Radius * 2, 1)
                    center = (
                        surface.Center.x,
                        surface.Center.y,
                        surface.Center.z
                    )
                    
                    # Track unique diameters
                    if diameter not in detected_diameters:
                        detected_diameters.add(diameter)
                        result.hole_diameters.append(diameter)
                        result.hole_positions[diameter] = []
                    
                    # Track all positions
                    result.hole_positions[diameter].append(center)
                    result.hole_count += 1
                    
            except Exception:
                # Skip faces that can't be analyzed
                pass
        
        # Sort diameters
        result.hole_diameters.sort()
        
        # Report findings
        if result.hole_count > 0:
            print(f"  ✓ Holes Detected: {result.hole_count}")
            print(f"  ✓ Unique Diameters: {result.hole_diameters}")
    
    def _classify_edges(self) -> List[EdgeInfo]:
        """
        Classify all edges by their primary direction.
        
        Determines if each edge is primarily aligned with
        X, Y, Z axis or is diagonal/curved.
        
        Returns:
            List of EdgeInfo objects
        """
        edges = []
        
        for i, edge in enumerate(self.shape.Edges):
            try:
                # Need at least 2 vertices
                if len(edge.Vertexes) < 2:
                    continue
                
                # Get endpoint coordinates
                p1 = edge.Vertexes[0].Point
                p2 = edge.Vertexes[1].Point
                
                # Calculate deltas
                dx = abs(p2.x - p1.x)
                dy = abs(p2.y - p1.y)
                dz = abs(p2.z - p1.z)
                
                # Determine primary direction
                if dx > dy and dx > dz:
                    direction = 'X'
                elif dy > dx and dy > dz:
                    direction = 'Y'
                elif dz > dx and dz > dy:
                    direction = 'Z'
                else:
                    direction = 'OTHER'
                
                # Check if circular
                is_circular = False
                try:
                    curve_type = type(edge.Curve).__name__
                    is_circular = 'Circle' in curve_type or 'Arc' in curve_type
                except:
                    pass
                
                edges.append(EdgeInfo(
                    index=i,
                    length=edge.Length,
                    direction=direction,
                    start_point=(p1.x, p1.y, p1.z),
                    end_point=(p2.x, p2.y, p2.z),
                    is_circular=is_circular
                ))
                
            except Exception:
                pass
        
        return edges
    
    def find_edges_by_length(
        self,
        target_length: float,
        direction: Optional[str] = None,
        tolerance: float = 1.0
    ) -> List[EdgeInfo]:
        """
        Find edges matching specified criteria.
        
        Args:
            target_length: Length to match in mm
            direction: Optional direction filter ('X', 'Y', 'Z')
            tolerance: Matching tolerance in mm
            
        Returns:
            List of matching EdgeInfo objects
        """
        # Ensure edges are classified
        edges = self._classify_edges()
        
        matches = []
        for edge in edges:
            # Check length
            if abs(edge.length - target_length) > tolerance:
                continue
            
            # Check direction if specified
            if direction and edge.direction != direction:
                continue
            
            matches.append(edge)
        
        return matches


# =============================================================================
# TITLE BLOCK GENERATOR
# =============================================================================

class TitleBlockGenerator:
    """
    Generates title block data from part analysis.
    
    Creates metadata for technical drawing title block including
    part name, number, material, weight, scale, and dimensions.
    
    Attributes:
        config (Config): Configuration object
        
    Example:
        >>> generator = TitleBlockGenerator()
        >>> title_data = generator.generate("bracket.step", analysis)
        >>> print(title_data.part_name)  # "BRACKET"
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize generator.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
    
    def generate(self, filepath: str, analysis: AnalysisResult) -> TitleBlockData:
        """
        Generate title block data from file and analysis.
        
        Args:
            filepath: Input file path (used for naming)
            analysis: Geometry analysis results
            
        Returns:
            TitleBlockData with all fields populated
        """
        data = TitleBlockData()
        
        # Part name from filename
        basename = os.path.splitext(os.path.basename(filepath))[0]
        data.part_name = self._clean_name(basename)
        
        # Part number
        data.part_number = self._generate_part_number(basename)
        
        # Material
        data.material = self.config.DEFAULT_MATERIAL
        
        # Weight calculation (volume * density)
        weight_kg = analysis.volume * self.config.MATERIAL_DENSITY
        data.weight = f"{weight_kg:.3f} kg"
        
        # Scale selection based on max dimension
        max_dim = analysis.bbox.get('max_dim', 100)
        data.scale, data.scale_factor = self.config.get_scale(max_dim)
        
        # Dimensions string
        bbox = analysis.bbox
        data.dimensions_str = (
            f"{bbox.get('dx', 0):.1f} x "
            f"{bbox.get('dy', 0):.1f} x "
            f"{bbox.get('dz', 0):.1f}"
        )
        
        return data
    
    def _clean_name(self, name: str) -> str:
        """
        Clean filename for display as part name.
        
        Args:
            name: Raw filename without extension
            
        Returns:
            Cleaned display name
        """
        # Replace separators with spaces
        clean = name.replace('_', ' ').replace('-', ' ')
        
        # Convert to uppercase
        clean = clean.upper()
        
        # Remove multiple spaces
        while '  ' in clean:
            clean = clean.replace('  ', ' ')
        
        return clean.strip()
    
    def _generate_part_number(self, name: str) -> str:
        """
        Generate part number from filename.
        
        Args:
            name: Raw filename without extension
            
        Returns:
            Part number string
        """
        # Clean and truncate
        clean = name.upper().replace(' ', '').replace('_', '').replace('-', '')
        prefix = clean[:6] if len(clean) >= 6 else clean
        
        return f"{prefix}-001"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def to_float(val: Any) -> float:
    """
    Convert various value types to float.
    
    Handles FreeCAD Quantity objects, None, and numeric types.
    
    Args:
        val: Value to convert (Quantity, number, or None)
        
    Returns:
        Float value (0.0 if conversion fails)
        
    Example:
        >>> to_float(None)
        0.0
        >>> to_float(3.14)
        3.14
        >>> to_float(quantity_with_units)  # FreeCAD Quantity
        42.0
    """
    if val is None:
        return 0.0
    
    # Handle FreeCAD Quantity objects
    if hasattr(val, 'Value'):
        return float(val.Value)
    
    # Try direct conversion
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def find_template(freecad_module: Any, config: Config) -> Optional[str]:
    """
    Find a TechDraw template file.
    
    Searches FreeCAD resource directories for template SVG files.
    
    Args:
        freecad_module: FreeCAD module reference
        config: Configuration object
        
    Returns:
        Full path to template file, or None if not found
        
    Example:
        >>> path = find_template(FreeCAD, config)
        >>> print(path)
        '/usr/share/freecad/Mod/TechDraw/Templates/A4_LandscapeTD.svg'
    """
    res_dir = freecad_module.getResourceDir()
    
    for template_name in config.TEMPLATE_NAMES:
        for template_path in config.TEMPLATE_PATHS:
            full_path = os.path.join(res_dir, template_path, template_name)
            if os.path.exists(full_path):
                return full_path
    
    return None


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'Config',
    'ViewConfig',
    'EdgeInfo',
    'AnalysisResult',
    'TitleBlockData',
    'ProjectedView',
    'GeometryAnalyzer',
    'TitleBlockGenerator',
    'to_float',
    'find_template',
]