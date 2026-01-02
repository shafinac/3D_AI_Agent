"""
AI Drafting Agent - Generators Module
======================================

This module contains:
- TechDraw generators (view and dimension creation)
- View projector (3D to 2D projection)
- PDF renderer (matplotlib-based output)
- Dimension drawer utilities

All output generation functionality for both FreeCAD TechDraw
and matplotlib PDF rendering.

Author: AI Engineering Team
Version: 1.0.0
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np

# Matplotlib with non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from core import (
    Config, AnalysisResult, TitleBlockData, ProjectedView,
    to_float, find_template
)


# =============================================================================
# BASE TECHDRAW GENERATOR
# =============================================================================

class BaseTechDrawGenerator(ABC):
    """
    Abstract base class for TechDraw generation.
    
    Provides common infrastructure for page, view, and dimension
    creation. Subclasses implement specific dimensioning strategies.
    
    Attributes:
        doc: FreeCAD document
        part: Part object to draw
        analysis: Geometry analysis results
        title_data: Title block data
        config: Configuration object
        page: TechDraw page object
        views: Dictionary mapping view names to view objects
        dim_count: Number of dimensions created
        scale: Drawing scale factor
    """
    
    def __init__(
        self,
        doc,
        part_obj,
        analysis: AnalysisResult,
        title_data: TitleBlockData,
        config: Optional[Config],
        freecad_module,
        techdraw_module
    ):
        """
        Initialize generator.
        
        Args:
            doc: FreeCAD document
            part_obj: Part object to draw
            analysis: Analysis results
            title_data: Title block data
            config: Configuration object
            freecad_module: FreeCAD module reference
            techdraw_module: TechDraw module reference
        """
        self.doc = doc
        self.part = part_obj
        self.analysis = analysis
        self.title_data = title_data
        self.config = config or Config()
        self.FreeCAD = freecad_module
        self.TechDraw = techdraw_module
        
        self.page = None
        self.views: Dict[str, object] = {}
        self.scale = title_data.scale_factor
        self.dim_count = 0
    
    def generate(self):
        """
        Generate complete TechDraw page.
        
        Template method that orchestrates:
        1. Page creation with template
        2. Orthographic view creation
        3. Wait for view computation
        4. Dimension addition (implemented by subclass)
        
        Returns:
            TechDraw page object
        """
        print("\n[STEP 4: TECHDRAW]")
        self.FreeCAD.setActiveDocument(self.doc.Name)
        
        self._create_page()
        self._create_views()
        self._wait_for_computation()
        self._add_dimensions()
        
        self.doc.recompute()
        print(f"  ✓ Total dimensions: {self.dim_count}")
        
        return self.page
    
    def _create_page(self) -> None:
        """Create TechDraw page with SVG template."""
        self.page = self.doc.addObject('TechDraw::DrawPage', 'Page')
        template = self.doc.addObject('TechDraw::DrawSVGTemplate', 'Template')
        
        # Find and load template
        template_path = find_template(self.FreeCAD, self.config)
        if template_path:
            template.Template = template_path
            print(f"  ✓ Template: {os.path.basename(template_path)}")
        else:
            print("  ⚠ No template found, using blank page")
        
        self.page.Template = template
        self.doc.recompute()
    
    def _create_views(self) -> None:
        """Create orthographic views (front, top, right)."""
        for name, view_config in self.config.VIEWS.items():
            view = self.doc.addObject('TechDraw::DrawViewPart', f'View_{name}')
            self.page.addView(view)
            
            # Configure view
            view.Source = [self.part]
            view.Direction = self.FreeCAD.Vector(*view_config.direction)
            view.XDirection = self.FreeCAD.Vector(*view_config.x_direction)
            view.X, view.Y = view_config.position
            view.Scale = self.scale
            view.CoarseView = False
            
            self.views[name] = view
            self.doc.recompute()
            print(f"  ✓ View: {name}")
    
    def _wait_for_computation(self) -> None:
        """Wait for TechDraw views to fully compute."""
        print("  Computing views...")
        for _ in range(self.config.RECOMPUTE_ITERATIONS):
            self.doc.recompute()
        time.sleep(self.config.COMPUTE_DELAY)
    
    @abstractmethod
    def _add_dimensions(self) -> None:
        """
        Add dimensions to views.
        
        Abstract method implemented by subclasses with specific
        dimensioning strategies.
        """
        pass
    
    def _create_dimension(
        self,
        view,
        dim_type: str,
        references: List[str],
        name: str,
        offset_x: float = 0,
        offset_y: float = 0
    ) -> Optional[object]:
        """
        Create a TechDraw dimension object.
        
        Args:
            view: TechDraw view to attach dimension to
            dim_type: Dimension type - 'Distance', 'DistanceX', 
                     'DistanceY', or 'Diameter'
            references: List of reference strings (e.g., ['Edge0', 'Vertex1'])
            name: Unique name for the dimension object
            offset_x: X offset for dimension text position
            offset_y: Y offset for dimension text position
            
        Returns:
            Dimension object if successful, None otherwise
        """
        try:
            dim = self.doc.addObject('TechDraw::DrawViewDimension', name)
            dim.Type = dim_type
            dim.References2D = [(view, ref) for ref in references]
            
            # Set format based on type
            if dim_type == 'Diameter':
                dim.FormatSpec = self.config.DIAMETER_FORMAT
            else:
                dim.FormatSpec = self.config.DIMENSION_FORMAT
            
            dim.X = offset_x
            dim.Y = offset_y
            
            # Disable arbitrary value
            if hasattr(dim, 'Arbitrary'):
                dim.Arbitrary = False
            
            self.page.addView(dim)
            self.doc.recompute()
            
            return dim
            
        except Exception as e:
            # Clean up on failure
            try:
                self.doc.removeObject(name)
            except:
                pass
            return None
    
    def _get_dimension_value(self, dim) -> float:
        """
        Get measured value from dimension object.
        
        Args:
            dim: Dimension object
            
        Returns:
            Measured value as float, or 0.0 if unavailable
        """
        if hasattr(dim, 'Measurement'):
            return to_float(dim.Measurement)
        return 0.0
    
    def _remove_dimension(self, dim, name: str) -> None:
        """
        Remove dimension from page and document.
        
        Args:
            dim: Dimension object to remove
            name: Object name for removal
        """
        try:
            self.page.removeView(dim)
            self.doc.removeObject(name)
        except:
            pass
    
    def save(self, output_path: str) -> str:
        """
        Save FreeCAD document.
        
        Args:
            output_path: Base path (without .FCStd extension)
            
        Returns:
            Full path to saved file
        """
        self.doc.recompute()
        path = f"{output_path}.FCStd"
        self.doc.saveAs(path)
        print(f"\n  ✓ Saved: {path}")
        return path


# =============================================================================
# VERTEX-BASED DIMENSION GENERATOR
# =============================================================================

class TechDrawGeneratorVertex(BaseTechDrawGenerator):
    """
    Dimension generator using vertex-based approach.
    
    Strategy:
    1. Search for vertex pairs that produce dimensions matching
       the expected bounding box values
    2. Use DistanceX/DistanceY for more reliable results
    3. Fall back to edge-based dimensions if vertex method fails
    4. Add diameter dimensions for circular features
    
    This is the primary dimensioning strategy as it produces
    more reliable results across different part geometries.
    """
    
    def _add_dimensions(self) -> None:
        """Add dimensions using vertex-pair search method."""
        print("  Adding dimensions (vertex method)...")
        
        bbox = self.analysis.bbox
        
        for view_name, view in self.views.items():
            # Determine expected dimensions for this view
            if view_name == 'front':
                width_expected = bbox['dx']
                height_expected = bbox['dz']
            elif view_name == 'top':
                width_expected = bbox['dx']
                height_expected = bbox['dy']
            else:  # right
                width_expected = bbox['dy']
                height_expected = bbox['dz']
            
            # Try vertex-based dimensions first
            width_found = self._find_vertex_dimension(
                view, view_name, 'DistanceX', width_expected, 'W',
                0, self.config.OFFSET_HORIZONTAL
            )
            
            height_found = self._find_vertex_dimension(
                view, view_name, 'DistanceY', height_expected, 'H',
                self.config.OFFSET_VERTICAL, 0
            )
            
            # Fallback to edge-based if vertex method failed
            if not width_found:
                self._try_edge_dimension(
                    view, view_name, 'W', width_expected,
                    0, self.config.OFFSET_HORIZONTAL
                )
            
            if not height_found:
                self._try_edge_dimension(
                    view, view_name, 'H', height_expected,
                    self.config.OFFSET_VERTICAL, 0
                )
            
            # Add diameter dimensions for right view (shows holes)
            if view_name == 'right':
                self._add_diameter_dimensions(view)
    
    def _find_vertex_dimension(
        self,
        view,
        view_name: str,
        dim_type: str,
        expected: float,
        suffix: str,
        offset_x: float,
        offset_y: float
    ) -> bool:
        """
        Search for vertex pair producing expected dimension.
        
        Iterates through vertex pairs, creates test dimensions,
        and keeps the one matching the expected value.
        
        Args:
            view: TechDraw view object
            view_name: View identifier for naming
            dim_type: 'DistanceX' or 'DistanceY'
            expected: Expected dimension value
            suffix: Dimension name suffix ('W' or 'H')
            offset_x: X position offset
            offset_y: Y position offset
            
        Returns:
            True if dimension was successfully created
        """
        tolerance = self.config.DIMENSION_TOLERANCE
        max_idx = min(30, self.config.MAX_SEARCH_INDICES)
        
        for i in range(max_idx):
            for j in range(i + 1, max_idx):
                test_name = f'_vtest_{view_name}_{i}_{j}'
                
                try:
                    # Create test dimension
                    test_dim = self._create_dimension(
                        view, dim_type,
                        [f'Vertex{i}', f'Vertex{j}'],
                        test_name
                    )
                    
                    if test_dim is None:
                        continue
                    
                    # Check measured value
                    measured = self._get_dimension_value(test_dim)
                    self._remove_dimension(test_dim, test_name)
                    
                    # If matches expected, create permanent dimension
                    if abs(measured - expected) < tolerance:
                        dim_name = f'Dim_{view_name}_{suffix}'
                        dim = self._create_dimension(
                            view, dim_type,
                            [f'Vertex{i}', f'Vertex{j}'],
                            dim_name, offset_x, offset_y
                        )
                        
                        if dim:
                            self.dim_count += 1
                            print(f"    ✓ {view_name}/{suffix}: {measured:.1f}mm (V{i}-V{j})")
                            return True
                            
                except Exception:
                    try:
                        self.doc.removeObject(test_name)
                    except:
                        pass
        
        return False
    
    def _try_edge_dimension(
        self,
        view,
        view_name: str,
        suffix: str,
        expected: float,
        offset_x: float,
        offset_y: float
    ) -> bool:
        """
        Fallback: try edge-based dimension.
        
        Searches through edges for one matching the expected length.
        
        Args:
            view: TechDraw view
            view_name: View identifier
            suffix: Dimension suffix
            expected: Expected value
            offset_x: X offset
            offset_y: Y offset
            
        Returns:
            True if dimension was created
        """
        tolerance = self.config.DIMENSION_TOLERANCE
        
        for i in range(self.config.MAX_SEARCH_INDICES):
            test_name = f'_etest_{view_name}_{i}'
            
            try:
                test_dim = self._create_dimension(
                    view, 'Distance', [f'Edge{i}'], test_name
                )
                
                if test_dim is None:
                    continue
                
                measured = self._get_dimension_value(test_dim)
                self._remove_dimension(test_dim, test_name)
                
                if abs(measured - expected) < tolerance:
                    dim_name = f'Dim_{view_name}_{suffix}'
                    dim = self._create_dimension(
                        view, 'Distance', [f'Edge{i}'],
                        dim_name, offset_x, offset_y
                    )
                    
                    if dim:
                        self.dim_count += 1
                        print(f"    ✓ {view_name}/{suffix}: {measured:.1f}mm (Edge{i})")
                        return True
                        
            except Exception:
                try:
                    self.doc.removeObject(test_name)
                except:
                    pass
        
        print(f"    ✗ {view_name}/{suffix}: not found")
        return False
    
    def _add_diameter_dimensions(self, view) -> None:
        """
        Add diameter dimensions for circular features.
        
        Searches for circular edges and creates diameter dimensions.
        Avoids duplicating dimensions for same diameter.
        
        Args:
            view: TechDraw view (typically right view)
        """
        added_diameters = set()
        position_offset = 0
        
        for i in range(self.config.MAX_SEARCH_INDICES):
            test_name = f'_dtest_{i}'
            
            try:
                test_dim = self._create_dimension(
                    view, 'Diameter', [f'Edge{i}'], test_name
                )
                
                if test_dim is None:
                    continue
                
                measured = self._get_dimension_value(test_dim)
                self._remove_dimension(test_dim, test_name)
                
                if measured > 0:
                    diameter = round(measured, 1)
                    
                    # Avoid duplicate diameters
                    if diameter not in added_diameters:
                        dim_name = f'Dim_Dia_{int(diameter * 10)}'
                        dim = self._create_dimension(
                            view, 'Diameter', [f'Edge{i}'],
                            dim_name,
                            25 + position_offset * 12,
                            10
                        )
                        
                        if dim:
                            added_diameters.add(diameter)
                            position_offset += 1
                            self.dim_count += 1
                            print(f"    ✓ Ø{diameter}mm")
                            
            except Exception:
                try:
                    self.doc.removeObject(test_name)
                except:
                    pass


# =============================================================================
# SIMPLE DIRECT GENERATOR (FALLBACK)
# =============================================================================

class TechDrawSimpleDirect(BaseTechDrawGenerator):
    """
    Simple fallback generator using direct edge indexing.
    
    Creates dimensions directly on edges 0, 1, 2, etc. without
    validation or searching. Used when vertex-based method fails
    completely.
    
    This ensures at least some dimensions are created even for
    problematic geometries.
    """
    
    def _add_dimensions(self) -> None:
        """Create dimensions directly on first available edges."""
        print("  Adding dimensions (simple direct method)...")
        
        for view_name, view in self.views.items():
            # Create width and height dimensions
            for i, suffix in enumerate(['W', 'H']):
                dim_name = f'Dim_{view_name}_{suffix}'
                
                try:
                    dim = self._create_dimension(
                        view, 'Distance', [f'Edge{i}'],
                        dim_name,
                        0 if i == 0 else 20,  # X offset
                        15 if i == 0 else 0   # Y offset
                    )
                    
                    if dim:
                        measured = self._get_dimension_value(dim)
                        if measured > 0:
                            self.dim_count += 1
                            print(f"    ✓ {view_name}/{suffix}: {measured:.1f}mm")
                        else:
                            self._remove_dimension(dim, dim_name)
                            print(f"    ✗ {view_name}/{suffix}: no measurement")
                            
                except Exception as e:
                    print(f"    ✗ {view_name}/{suffix}: {e}")
            
            # Try diameter dimensions for right view
            if view_name == 'right':
                for i in range(2, 15):
                    dim_name = f'Dim_dia_{i}'
                    
                    try:
                        dim = self._create_dimension(
                            view, 'Diameter', [f'Edge{i}'],
                            dim_name, 25, 10
                        )
                        
                        if dim:
                            measured = self._get_dimension_value(dim)
                            if measured > 0:
                                self.dim_count += 1
                                print(f"    ✓ Ø{measured:.1f}mm")
                                break  # Only need one diameter
                            else:
                                self._remove_dimension(dim, dim_name)
                                
                    except Exception:
                        pass


# =============================================================================
# VIEW PROJECTOR
# =============================================================================

class ViewProjector:
    """
    Projects 3D geometry to 2D orthographic views.
    
    Used for PDF rendering. Creates ProjectedView objects containing
    discretized edge polylines and detected circles.
    
    Attributes:
        shape: FreeCAD Part.Shape object
        config: Configuration object
        
    Example:
        >>> projector = ViewProjector(part.Shape)
        >>> front_view = projector.project('front')
        >>> print(f"Front view has {len(front_view.edges)} edges")
    """
    
    # Projection functions mapping 3D (x,y,z) to 2D (u,v)
    PROJECTIONS = {
        'front': lambda x, y, z: (x, z),      # X-Z plane (looking from -Y)
        'top': lambda x, y, z: (x, y),        # X-Y plane (looking from +Z)
        'right': lambda x, y, z: (-y, z),     # Y-Z plane (looking from +X)
    }
    
    def __init__(self, shape, config: Optional[Config] = None):
        """
        Initialize projector.
        
        Args:
            shape: FreeCAD Part.Shape object
            config: Configuration object
        """
        self.shape = shape
        self.config = config or Config()
    
    def project(self, view_name: str, include_circles: bool = False) -> ProjectedView:
        """
        Project shape to 2D view.
        
        Args:
            view_name: View identifier ('front', 'top', 'right')
            include_circles: Whether to detect and include circular features
            
        Returns:
            ProjectedView containing edges, circles, and bounds
        """
        if view_name not in self.PROJECTIONS:
            raise ValueError(f"Unknown view: {view_name}. Use: {list(self.PROJECTIONS.keys())}")
        
        proj_func = self.PROJECTIONS[view_name]
        view = ProjectedView(name=view_name)
        
        # Project all edges
        for edge in self.shape.Edges:
            try:
                # Discretize edge into points
                points = edge.discretize(Deflection=self.config.EDGE_DISCRETIZATION)
                
                if len(points) > 1:
                    # Project each point to 2D
                    projected_points = np.array([
                        proj_func(p.x, p.y, p.z) for p in points
                    ])
                    view.edges.append(projected_points)
                    
            except Exception:
                pass
        
        # Detect circles (for hole visualization)
        if include_circles:
            view.circles = self._detect_circles(proj_func)
        
        # Normalize coordinates (move to origin)
        self._normalize_view(view)
        
        return view
    
    def _detect_circles(self, proj_func) -> List[Tuple[Tuple[float, float], float]]:
        """
        Detect cylindrical faces and project their centers.
        
        Args:
            proj_func: Projection function
            
        Returns:
            List of ((center_x, center_y), radius) tuples
        """
        circles = []
        
        for face in self.shape.Faces:
            try:
                surface_type = getattr(face.Surface, 'TypeId', '')
                
                if 'Cylinder' in surface_type:
                    center = face.Surface.Center
                    projected_center = proj_func(center.x, center.y, center.z)
                    radius = face.Surface.Radius
                    circles.append((projected_center, radius))
                    
            except Exception:
                pass
        
        return circles
    
    def _normalize_view(self, view: ProjectedView) -> None:
        """
        Normalize view coordinates to start at origin.
        
        Shifts all geometry so minimum coordinates are at (0, 0).
        Updates bounds to reflect actual size.
        
        Args:
            view: ProjectedView to normalize (modified in place)
        """
        if not view.edges:
            return
        
        # Find bounding box of all edges
        all_points = np.vstack(view.edges)
        min_point = all_points.min(axis=0)
        max_point = all_points.max(axis=0)
        
        # Shift edges to origin
        view.edges = [edge - min_point for edge in view.edges]
        
        # Shift circles
        view.circles = [
            ((c[0] - min_point[0], c[1] - min_point[1]), r)
            for c, r in view.circles
        ]
        
        # Update bounds
        view.bounds = (max_point[0] - min_point[0], max_point[1] - min_point[1])


# =============================================================================
# DIMENSION DRAWER (FOR PDF)
# =============================================================================

class DimensionDrawer:
    """
    Draws dimension annotations on matplotlib axes.
    
    Provides methods for creating professional-looking dimensions
    including extension lines, dimension lines, arrows, and text.
    
    Attributes:
        ax: Matplotlib axes object
        color: Dimension line color
        line_width: Line width for dimension lines
        font_size: Font size for dimension text
    """
    
    def __init__(
        self,
        ax,
        color: str = '#0066CC',
        line_width: float = 0.6,
        font_size: int = 8
    ):
        """
        Initialize dimension drawer.
        
        Args:
            ax: Matplotlib axes to draw on
            color: Color for dimension lines and text
            line_width: Width of dimension lines
            font_size: Size of dimension text
        """
        self.ax = ax
        self.color = color
        self.line_width = line_width
        self.font_size = font_size
    
    def horizontal(
        self,
        x1: float,
        x2: float,
        y_ref: float,
        value: float,
        offset: float = 12
    ) -> None:
        """
        Draw horizontal dimension.
        
        Args:
            x1: Left X coordinate
            x2: Right X coordinate
            y_ref: Y reference (top of geometry)
            value: Dimension value to display
            offset: Distance above reference
        """
        y = y_ref + offset
        
        # Extension lines
        self.ax.plot([x1, x1], [y_ref + 2, y + 3],
                     color=self.color, lw=self.line_width)
        self.ax.plot([x2, x2], [y_ref + 2, y + 3],
                     color=self.color, lw=self.line_width)
        
        # Dimension line
        self.ax.plot([x1, x2], [y, y],
                     color=self.color, lw=self.line_width)
        
        # Arrows
        arrow_props = dict(arrowstyle='->', color=self.color, lw=self.line_width)
        self.ax.annotate('', xy=(x1, y), xytext=(x1 + 3, y), arrowprops=arrow_props)
        self.ax.annotate('', xy=(x2, y), xytext=(x2 - 3, y), arrowprops=arrow_props)
        
        # Dimension text
        self.ax.text(
            (x1 + x2) / 2, y,
            f'{value:.1f}',
            ha='center', va='center',
            fontsize=self.font_size,
            color=self.color,
            fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', pad=1)
        )
    
    def vertical(
        self,
        y1: float,
        y2: float,
        x_ref: float,
        value: float,
        offset: float = 12
    ) -> None:
        """
        Draw vertical dimension.
        
        Args:
            y1: Bottom Y coordinate
            y2: Top Y coordinate
            x_ref: X reference (right side of geometry)
            value: Dimension value to display
            offset: Distance to right of reference
        """
        x = x_ref + offset
        
        # Extension lines
        self.ax.plot([x_ref + 2, x + 3], [y1, y1],
                     color=self.color, lw=self.line_width)
        self.ax.plot([x_ref + 2, x + 3], [y2, y2],
                     color=self.color, lw=self.line_width)
        
        # Dimension line
        self.ax.plot([x, x], [y1, y2],
                     color=self.color, lw=self.line_width)
        
        # Arrows
        arrow_props = dict(arrowstyle='->', color=self.color, lw=self.line_width)
        self.ax.annotate('', xy=(x, y1), xytext=(x, y1 + 3), arrowprops=arrow_props)
        self.ax.annotate('', xy=(x, y2), xytext=(x, y2 - 3), arrowprops=arrow_props)
        
        # Dimension text (rotated)
        self.ax.text(
            x, (y1 + y2) / 2,
            f'{value:.1f}',
            ha='center', va='center',
            fontsize=self.font_size,
            color=self.color,
            fontweight='bold',
            rotation=90,
            bbox=dict(facecolor='white', edgecolor='none', pad=1)
        )
    
    def diameter(
        self,
        center: Tuple[float, float],
        radius: float,
        value: float,
        count: int = 1,
        angle: float = 45
    ) -> None:
        """
        Draw diameter dimension with leader line.
        
        Args:
            center: Circle center (x, y)
            radius: Circle radius
            value: Diameter value to display
            count: Number of identical holes (for "2x" notation)
            angle: Leader line angle in degrees
        """
        cx, cy = center
        angle_rad = np.radians(angle)
        
        # Point on circle
        px = cx + radius * np.cos(angle_rad)
        py = cy + radius * np.sin(angle_rad)
        
        # Leader line end point
        ex = px + 20 * np.cos(angle_rad)
        ey = py + 20 * np.sin(angle_rad)
        
        # Leader line
        self.ax.plot([px, ex], [py, ey],
                     color=self.color, lw=self.line_width)
        
        # Horizontal extension
        self.ax.plot([ex, ex + 8], [ey, ey],
                     color=self.color, lw=self.line_width)
        
        # Dimension text
        text = f'⌀{value:.1f}'
        if count > 1:
            text += f' ({count}x)'
        
        self.ax.text(
            ex + 10, ey,
            text,
            ha='left', va='center',
            fontsize=self.font_size - 1,
            color=self.color,
            fontweight='bold'
        )


# =============================================================================
# PDF RENDERER
# =============================================================================

class PDFRenderer:
    """
    Renders technical drawing to PDF using matplotlib.
    
    Creates a multi-view technical drawing with:
    - Title block with part information
    - Front, top, and right orthographic views
    - Automatic dimensions
    - Hole callouts
    
    Attributes:
        analysis: Geometry analysis results
        title_data: Title block data
        config: Configuration object
        views: Dictionary of ProjectedView objects
        dim_count: Number of dimensions rendered
    """
    
    def __init__(
        self,
        analysis: AnalysisResult,
        title_data: TitleBlockData,
        config: Optional[Config] = None
    ):
        """
        Initialize PDF renderer.
        
        Args:
            analysis: Geometry analysis results
            title_data: Title block data
            config: Configuration object
        """
        self.analysis = analysis
        self.title_data = title_data
        self.config = config or Config()
        self.views: Dict[str, ProjectedView] = {}
        self.dim_count = 0
    
    def add_view(self, view: ProjectedView) -> None:
        """
        Add a projected view to be rendered.
        
        Args:
            view: ProjectedView object
        """
        self.views[view.name] = view
    
    def render(self, output_path: str) -> str:
        """
        Render views to PDF and PNG files.
        
        Args:
            output_path: Base path for output files (without extension)
            
        Returns:
            Path to PDF file
        """
        print("\n  Rendering PDF...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 11))
        fig.suptitle('TECHNICAL DRAWING', fontsize=16, fontweight='bold')
        
        # Grid layout: title on top, 3 views below
        gs = fig.add_gridspec(2, 3, height_ratios=[0.25, 1], hspace=0.3, wspace=0.3)
        
        ax_title = fig.add_subplot(gs[0, :])
        ax_front = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[1, 1])
        ax_right = fig.add_subplot(gs[1, 2])
        
        # Render components
        self._render_title_block(ax_title)
        self._render_view(ax_front, 'front', 'FRONT VIEW')
        self._render_view(ax_top, 'top', 'TOP VIEW')
        self._render_view(ax_right, 'right', 'RIGHT VIEW')
        
        # Save PDF
        pdf_path = f"{output_path}.pdf"
        plt.savefig(pdf_path, format='pdf', dpi=self.config.PDF_DPI, bbox_inches='tight')
        print(f"  ✓ {pdf_path}")
        
        # Save PNG preview
        png_path = f"{output_path}.png"
        plt.savefig(png_path, format='png', dpi=self.config.PNG_DPI, bbox_inches='tight')
        print(f"  ✓ {png_path}")
        
        plt.close(fig)
        
        return pdf_path
    
    def _render_title_block(self, ax) -> None:
        """
        Render title block with part information.
        
        Args:
            ax: Matplotlib axes for title block
        """
        ax.axis('off')
        
        td = self.title_data
        
        # Build info text
        info_lines = [
            f"Part: {td.part_name}",
            f"Part No: {td.part_number}",
            f"Size: {td.dimensions_str} mm",
            f"Material: {td.material}",
            f"Weight: {td.weight}",
            f"Scale: {td.scale}",
            f"Date: {td.date}",
            f"Drawn: {td.drawn_by}",
        ]
        
        info_text = "  |  ".join(info_lines)
        
        ax.text(
            0.5, 0.5,
            info_text,
            ha='center', va='center',
            fontsize=9,
            transform=ax.transAxes,
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='#f5f5f5',
                edgecolor='#333333',
                linewidth=1
            )
        )
    
    def _render_view(self, ax, view_name: str, title: str) -> None:
        """
        Render a single orthographic view with dimensions.
        
        Args:
            ax: Matplotlib axes
            view_name: View identifier
            title: Display title for the view
        """
        ax.set_title(title, fontweight='bold', fontsize=11)
        
        if view_name not in self.views:
            ax.text(0.5, 0.5, 'View not available',
                    ha='center', va='center', transform=ax.transAxes)
            return
        
        view = self.views[view_name]
        width, height = view.bounds
        bbox = self.analysis.bbox
        
        # Draw edges
        for edge in view.edges:
            if len(edge) > 1:
                ax.plot(edge[:, 0], edge[:, 1], 'k-', lw=0.8)
        
        # Draw circles (holes)
        for (cx, cy), radius in view.circles:
            circle = Circle((cx, cy), radius, fill=False, color='k', lw=0.6)
            ax.add_patch(circle)
            # Draw center mark
            mark_size = radius * 0.3
            ax.plot([cx - mark_size, cx + mark_size], [cy, cy], 'k-', lw=0.3)
            ax.plot([cx, cx], [cy - mark_size, cy + mark_size], 'k-', lw=0.3)
        
        # Add dimensions
        dim_drawer = DimensionDrawer(ax)
        
        if view_name == 'front':
            dim_drawer.horizontal(0, width, height, bbox['dx'], 10)
            dim_drawer.vertical(0, height, width, bbox['dz'], 12)
            self.dim_count += 2
            
        elif view_name == 'top':
            dim_drawer.horizontal(0, width, height, bbox['dx'], 10)
            dim_drawer.vertical(0, height, width, bbox['dy'], 12)
            self.dim_count += 2
            
        elif view_name == 'right':
            dim_drawer.horizontal(0, width, height, bbox['dy'], 10)
            dim_drawer.vertical(0, height, width, bbox['dz'], 12)
            self.dim_count += 2
            
            # Add diameter dimensions for holes
            if view.circles:
                self._add_hole_dimensions(ax, dim_drawer, view.circles)
        
        # Configure axes
        padding = max(width, height) * 0.25
        ax.set_xlim(-padding, width + padding * 1.5)
        ax.set_ylim(-padding, height + padding * 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2, linestyle='--')
        
        # Remove axis labels
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _add_hole_dimensions(
        self,
        ax,
        dim_drawer: DimensionDrawer,
        circles: List[Tuple[Tuple[float, float], float]]
    ) -> None:
        """
        Add diameter dimensions for holes.
        
        Groups holes by diameter and adds callouts with count.
        
        Args:
            ax: Matplotlib axes
            dim_drawer: DimensionDrawer instance
            circles: List of ((center_x, center_y), radius) tuples
        """
        # Group circles by diameter
        diameter_groups = defaultdict(list)
        for center, radius in circles:
            diameter = round(radius * 2, 1)
            diameter_groups[diameter].append((center, radius))
        
        # Draw dimension for each diameter group
        angle = 30
        for diameter, items in sorted(diameter_groups.items(), reverse=True):
            # Use first circle in group as reference
            center, radius = items[0]
            count = len(items)
            
            dim_drawer.diameter(center, radius, diameter, count, angle)
            self.dim_count += 1
            
            # Offset angle for next diameter
            angle += 50