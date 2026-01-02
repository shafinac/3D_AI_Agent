

#!/usr/bin/env python3
"""
================================================================================
AI DRAFTING AGENT
================================================================================
Automated 2D technical drawing generation from 3D CAD models.

Creates TechDraw dimensions by directly referencing edges and vertices,
letting FreeCAD compute the measurements.

Usage:
    freecadcmd ai_drafting_agent.py <input.step> <output_name>

Outputs:
    - <output_name>.FCStd  : FreeCAD document with TechDraw
    - <output_name>.pdf    : Technical drawing PDF
    - <output_name>.png    : Preview image
================================================================================
"""

import sys
import os
import math
import numpy as np
from datetime import datetime
from collections import defaultdict
import time

# =============================================================================
# MODULE IMPORTS
# =============================================================================

try:
    import FreeCAD
except ImportError as e:
    print(f"Error: FreeCAD not found - {e}")
    sys.exit(1)

try:
    import Part
except ImportError:
    print("Error: Part module not available")
    sys.exit(1)

try:
    import TechDraw
except ImportError:
    print("Error: TechDraw module not available")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for headless rendering
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
except ImportError:
    print("Error: Matplotlib not available")
    sys.exit(1)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def to_float(val):
    """
    Safely convert FreeCAD quantity or value to Python float.
    Handles FreeCAD.Units.Quantity objects and None values.
    """
    if val is None:
        return 0.0
    if hasattr(val, 'Value'):
        return float(val.Value)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class AnalysisResult:
    """Container for geometry analysis results."""
    
    def __init__(self):
        self.bbox = {}           # Bounding box dimensions
        self.volume = 0.0        # Part volume in mm³
        self.surface_area = 0.0  # Total surface area in mm²
        self.hole_count = 0      # Number of cylindrical holes detected
        self.hole_diameters = [] # List of unique hole diameters
        self.hole_positions = {} # Hole center positions by diameter
        self.face_count = 0      # Total face count
        self.edge_count = 0      # Total edge count


class TitleBlockData:
    """Container for drawing title block information."""
    
    def __init__(self):
        self.part_name = "UNKNOWN"
        self.part_number = "XXX-001"
        self.material = "ALUMINUM 6061-T6"
        self.weight = "0.000 kg"
        self.scale = "1:1"
        self.scale_factor = 1.0
        self.date = datetime.now().strftime('%Y-%m-%d')
        self.drawn_by = "AI Agent"
        self.dimensions_str = "0 x 0 x 0"


class ProjectedView:
    """Container for 2D projected view data."""
    
    def __init__(self, name):
        self.name = name
        self.edges = []    # List of edge point arrays
        self.circles = []  # List of (center, radius) tuples
        self.bounds = (0, 0)  # View width and height


# =============================================================================
# GEOMETRY ANALYZER
# =============================================================================

class GeometryAnalyzer:
    """
    Analyzes 3D geometry to extract dimensional and feature information.
    Identifies bounding box, holes, and other manufacturing-relevant features.
    """
    
    def __init__(self, shape):
        self.shape = shape
    
    def analyze(self):
        """
        Perform complete geometry analysis.
        Returns AnalysisResult with all extracted information.
        """
        result = AnalysisResult()
        
        # Extract bounding box dimensions
        bbox = self.shape.BoundBox
        result.bbox = {
            'dx': bbox.XLength,
            'dy': bbox.YLength,
            'dz': bbox.ZLength,
            'max_dim': max(bbox.XLength, bbox.YLength, bbox.ZLength),
        }
        
        # Basic geometry metrics
        result.volume = self.shape.Volume
        result.surface_area = self.shape.Area
        result.face_count = len(self.shape.Faces)
        result.edge_count = len(self.shape.Edges)
        
        # Detect cylindrical holes by analyzing face surfaces
        for face in self.shape.Faces:
            try:
                if 'Cylinder' in face.Surface.TypeId:
                    diameter = round(face.Surface.Radius * 2, 1)
                    if diameter not in result.hole_diameters:
                        result.hole_diameters.append(diameter)
                    result.hole_count += 1
            except (AttributeError, RuntimeError):
                pass
        
        result.hole_diameters.sort()
        return result


# =============================================================================
# TITLE BLOCK GENERATOR
# =============================================================================

class TitleBlockGenerator:
    """Generates title block metadata from file path and analysis results."""
    
    def generate(self, filepath, analysis):
        """
        Create title block data from input file and geometry analysis.
        Automatically determines scale based on part size.
        """
        data = TitleBlockData()
        
        # Extract part name from filename
        name = os.path.splitext(os.path.basename(filepath))[0]
        data.part_name = name.upper().replace('_', ' ')
        data.part_number = f"{name.upper()[:6]}-001"
        
        # Calculate weight assuming aluminum (density 2.7 g/cm³)
        data.weight = f"{analysis.volume * 2.7e-6:.3f} kg"
        
        # Auto-select scale based on maximum dimension
        max_dim = analysis.bbox['max_dim']
        if max_dim > 200:
            data.scale, data.scale_factor = "1:5", 0.2
        elif max_dim > 100:
            data.scale, data.scale_factor = "1:2", 0.5
        else:
            data.scale, data.scale_factor = "1:1", 1.0
        
        data.dimensions_str = (
            f"{analysis.bbox['dx']:.1f} x "
            f"{analysis.bbox['dy']:.1f} x "
            f"{analysis.bbox['dz']:.1f}"
        )
        return data


# =============================================================================
# VIEW PROJECTOR (for PDF generation)
# =============================================================================

class ViewProjector:
    """
    Projects 3D geometry onto 2D planes for PDF rendering.
    Supports standard orthographic views: front, top, right.
    """
    
    # Projection functions for each view direction
    PROJ = {
        'front': lambda x, y, z: (x, z),   # XZ plane
        'top': lambda x, y, z: (x, y),     # XY plane
        'right': lambda x, y, z: (-y, z),  # YZ plane (mirrored)
    }
    
    def __init__(self, shape):
        self.shape = shape
    
    def project(self, view_name, include_circles=False):
        """
        Project shape onto specified view plane.
        Optionally includes circle detection for hole visualization.
        """
        projection_func = self.PROJ[view_name]
        view = ProjectedView(view_name)
        
        # Project all edges
        for edge in self.shape.Edges:
            try:
                # Discretize edge into points for rendering
                pts = edge.discretize(Deflection=0.3)
                if len(pts) > 1:
                    projected = np.array([
                        projection_func(p.x, p.y, p.z) for p in pts
                    ])
                    view.edges.append(projected)
            except (RuntimeError, AttributeError):
                pass
        
        # Extract circles from cylindrical faces
        if include_circles:
            for face in self.shape.Faces:
                try:
                    if 'Cylinder' in face.Surface.TypeId:
                        center = face.Surface.Center
                        projected_center = projection_func(
                            center.x, center.y, center.z
                        )
                        view.circles.append(
                            (projected_center, face.Surface.Radius)
                        )
                except (AttributeError, RuntimeError):
                    pass
        
        # Normalize view coordinates to origin
        if view.edges:
            all_pts = np.vstack(view.edges)
            min_pt, max_pt = all_pts.min(axis=0), all_pts.max(axis=0)
            
            # Shift edges to origin
            view.edges = [e - min_pt for e in view.edges]
            
            # Shift circles to match
            if view.circles:
                view.circles = [
                    ((c[0] - min_pt[0], c[1] - min_pt[1]), r) 
                    for c, r in view.circles
                ]
            
            view.bounds = (max_pt[0] - min_pt[0], max_pt[1] - min_pt[1])
        
        return view


# =============================================================================
# TECHDRAW GENERATOR - VERTEX METHOD
# =============================================================================

class TechDrawGeneratorVertex:
    """
    Creates FreeCAD TechDraw document with dimensioned orthographic views.
    Uses vertex-based dimensions for reliable measurement extraction.
    """
    
    # Standard view positions on A4 landscape sheet
    VIEW_POS = {
        'front': (105, 170),
        'top': (105, 75),
        'right': (230, 170)
    }
    
    def __init__(self, doc, part_obj, analysis, title_data):
        self.doc = doc
        self.part = part_obj
        self.analysis = analysis
        self.title_data = title_data
        self.page = None
        self.views = {}
        self.scale = title_data.scale_factor
        self.dim_count = 0
    
    def generate(self):
        """Generate complete TechDraw page with views and dimensions."""
        FreeCAD.setActiveDocument(self.doc.Name)
        
        self._create_page()
        self._create_views()
        
        # Allow views to compute geometry
        for _ in range(10):
            self.doc.recompute()
        time.sleep(0.5)
        
        self._add_vertex_dimensions()
        self.doc.recompute()
        
        return self.page
    
    def _create_page(self):
        """Create TechDraw page with template."""
        self.page = self.doc.addObject('TechDraw::DrawPage', 'Page')
        tpl = self.doc.addObject('TechDraw::DrawSVGTemplate', 'Template')
        
        # Search for template in standard FreeCAD locations
        res = FreeCAD.getResourceDir()
        template_paths = [
            "Mod/TechDraw/Templates",
            "share/Mod/TechDraw/Templates"
        ]
        template_files = ["A4_LandscapeTD.svg", "A3_LandscapeTD.svg"]
        
        for template_file in template_files:
            for subdir in template_paths:
                full_path = os.path.join(res, subdir, template_file)
                if os.path.exists(full_path):
                    tpl.Template = full_path
                    break
            else:
                continue
            break
        
        self.page.Template = tpl
        self.doc.recompute()
    
    def _create_views(self):
        """Create orthographic views: front, top, and right."""
        view_configs = [
            ('front', FreeCAD.Vector(0, -1, 0), FreeCAD.Vector(1, 0, 0)),
            ('top', FreeCAD.Vector(0, 0, 1), FreeCAD.Vector(1, 0, 0)),
            ('right', FreeCAD.Vector(1, 0, 0), FreeCAD.Vector(0, 1, 0)),
        ]
        
        for name, direction, x_direction in view_configs:
            view = self.doc.addObject('TechDraw::DrawViewPart', f'View_{name}')
            self.page.addView(view)
            
            view.Source = [self.part]
            view.Direction = direction
            view.XDirection = x_direction
            view.X, view.Y = self.VIEW_POS[name]
            view.Scale = self.scale
            view.CoarseView = False
            
            self.views[name] = view
            self.doc.recompute()
    
    def _add_vertex_dimensions(self):
        """Add dimensions to all views using vertex pairs."""
        bbox = self.analysis.bbox
        
        for view_name, view in self.views.items():
            # Determine expected dimensions for each view
            if view_name == 'front':
                width_expected, height_expected = bbox['dx'], bbox['dz']
            elif view_name == 'top':
                width_expected, height_expected = bbox['dx'], bbox['dy']
            else:  # right
                width_expected, height_expected = bbox['dy'], bbox['dz']
            
            # Try vertex-based dimensions first
            w_found = self._find_and_create_vertex_dim(
                view, view_name, 'DistanceX', width_expected, 'W', 0, 18
            )
            h_found = self._find_and_create_vertex_dim(
                view, view_name, 'DistanceY', height_expected, 'H', 22, 0
            )
            
            # Fallback to edge-based dimensions if vertex method fails
            if not w_found:
                self._try_edge_dimension(view, view_name, 'W', width_expected, 0, 18)
            if not h_found:
                self._try_edge_dimension(view, view_name, 'H', height_expected, 22, 0)
            
            # Add diameter dimensions for right view (shows holes)
            if view_name == 'right':
                self._add_diameter_dims(view)
    
    def _find_and_create_vertex_dim(self, view, view_name, dim_type, expected, 
                                     suffix, off_x, off_y):
        """
        Find vertex pair with correct distance and create dimension.
        Tests vertex combinations until finding one matching expected value.
        """
        dim_name = f'Dim_{view_name}_{suffix}'
        tolerance = 1.0
        
        # Search through vertex pairs
        for i in range(30):
            for j in range(i + 1, 30):
                try:
                    # Create test dimension
                    test_name = f'_vtest_{i}_{j}'
                    test = self.doc.addObject('TechDraw::DrawViewDimension', test_name)
                    test.Type = dim_type
                    test.References2D = [(view, f'Vertex{i}'), (view, f'Vertex{j}')]
                    self.page.addView(test)
                    self.doc.recompute()
                    
                    meas = to_float(test.Measurement) if hasattr(test, 'Measurement') else 0
                    
                    # Clean up test dimension
                    self.page.removeView(test)
                    self.doc.removeObject(test_name)
                    
                    # Check if measurement matches expected value
                    if abs(meas - expected) < tolerance:
                        # Create actual dimension
                        dim = self.doc.addObject('TechDraw::DrawViewDimension', dim_name)
                        dim.Type = dim_type
                        dim.References2D = [(view, f'Vertex{i}'), (view, f'Vertex{j}')]
                        dim.FormatSpec = "%.1f"
                        dim.X = off_x
                        dim.Y = off_y
                        
                        if hasattr(dim, 'Arbitrary'):
                            dim.Arbitrary = False
                        
                        self.page.addView(dim)
                        self.doc.recompute()
                        self.dim_count += 1
                        return True
                        
                except (RuntimeError, AttributeError):
                    try:
                        self.doc.removeObject(f'_vtest_{i}_{j}')
                    except:
                        pass
        
        return False
    
    def _try_edge_dimension(self, view, view_name, suffix, expected, off_x, off_y):
        """Fallback: try edge-based dimension when vertex method fails."""
        dim_name = f'Dim_{view_name}_{suffix}'
        tolerance = 1.0
        
        for i in range(50):
            try:
                # Create test dimension
                test_name = f'_etest_{i}'
                test = self.doc.addObject('TechDraw::DrawViewDimension', test_name)
                test.Type = 'Distance'
                test.References2D = [(view, f'Edge{i}')]
                self.page.addView(test)
                self.doc.recompute()
                
                meas = to_float(test.Measurement) if hasattr(test, 'Measurement') else 0
                
                # Clean up test
                self.page.removeView(test)
                self.doc.removeObject(test_name)
                
                # Check match
                if abs(meas - expected) < tolerance:
                    dim = self.doc.addObject('TechDraw::DrawViewDimension', dim_name)
                    dim.Type = 'Distance'
                    dim.References2D = [(view, f'Edge{i}')]
                    dim.FormatSpec = "%.1f"
                    dim.X = off_x
                    dim.Y = off_y
                    
                    if hasattr(dim, 'Arbitrary'):
                        dim.Arbitrary = False
                    
                    self.page.addView(dim)
                    self.doc.recompute()
                    self.dim_count += 1
                    return True
                    
            except (RuntimeError, AttributeError):
                try:
                    self.doc.removeObject(f'_etest_{i}')
                except:
                    pass
        
        return False
    
    def _add_diameter_dims(self, view):
        """Add diameter dimensions for circular features (holes)."""
        added = set()
        
        for i in range(50):
            try:
                test_name = f'_dtest_{i}'
                test = self.doc.addObject('TechDraw::DrawViewDimension', test_name)
                test.Type = 'Diameter'
                test.References2D = [(view, f'Edge{i}')]
                self.page.addView(test)
                self.doc.recompute()
                
                meas = to_float(test.Measurement) if hasattr(test, 'Measurement') else 0
                
                self.page.removeView(test)
                self.doc.removeObject(test_name)
                
                if meas > 0:
                    diameter = round(meas, 1)
                    if diameter not in added:
                        dim_name = f'Dim_D{int(diameter * 10)}'
                        dim = self.doc.addObject('TechDraw::DrawViewDimension', dim_name)
                        dim.Type = 'Diameter'
                        dim.References2D = [(view, f'Edge{i}')]
                        dim.FormatSpec = "%%c%.1f"
                        dim.X = 25 + len(added) * 12
                        dim.Y = 10
                        
                        if hasattr(dim, 'Arbitrary'):
                            dim.Arbitrary = False
                        
                        self.page.addView(dim)
                        self.doc.recompute()
                        
                        added.add(diameter)
                        self.dim_count += 1
                        
            except (RuntimeError, AttributeError):
                try:
                    self.doc.removeObject(f'_dtest_{i}')
                except:
                    pass
    
    def save(self, output_path):
        """Save FreeCAD document."""
        self.doc.recompute()
        path = f"{output_path}.FCStd"
        self.doc.saveAs(path)
        return path


# =============================================================================
# SIMPLE DIRECT METHOD (Fallback)
# =============================================================================

class TechDrawSimpleDirect:
    """
    Simplified dimension creation as fallback.
    Creates dimensions directly on first available edges without testing.
    """
    
    VIEW_POS = {'front': (105, 170), 'top': (105, 75), 'right': (230, 170)}
    
    def __init__(self, doc, part_obj, analysis, title_data):
        self.doc = doc
        self.part = part_obj
        self.analysis = analysis
        self.title_data = title_data
        self.page = None
        self.views = {}
        self.scale = title_data.scale_factor
        self.dim_count = 0
    
    def generate(self):
        """Generate TechDraw with direct dimension placement."""
        FreeCAD.setActiveDocument(self.doc.Name)
        
        self._create_page()
        self._create_views()
        
        for _ in range(10):
            self.doc.recompute()
        
        self._create_direct_dimensions()
        self.doc.recompute()
        
        return self.page
    
    def _create_page(self):
        """Create drawing page with template."""
        self.page = self.doc.addObject('TechDraw::DrawPage', 'Page')
        tpl = self.doc.addObject('TechDraw::DrawSVGTemplate', 'Template')
        
        res = FreeCAD.getResourceDir()
        for subdir in ["Mod/TechDraw/Templates", "share/Mod/TechDraw/Templates"]:
            path = os.path.join(res, subdir, "A4_LandscapeTD.svg")
            if os.path.exists(path):
                tpl.Template = path
                break
        
        self.page.Template = tpl
        self.doc.recompute()
    
    def _create_views(self):
        """Create orthographic views."""
        configs = [
            ('front', FreeCAD.Vector(0, -1, 0), FreeCAD.Vector(1, 0, 0)),
            ('top', FreeCAD.Vector(0, 0, 1), FreeCAD.Vector(1, 0, 0)),
            ('right', FreeCAD.Vector(1, 0, 0), FreeCAD.Vector(0, 1, 0)),
        ]
        
        for name, direction, x_dir in configs:
            v = self.doc.addObject('TechDraw::DrawViewPart', f'View_{name}')
            self.page.addView(v)
            v.Source = [self.part]
            v.Direction = direction
            v.XDirection = x_dir
            v.X, v.Y = self.VIEW_POS[name]
            v.Scale = self.scale
            v.CoarseView = False
            self.views[name] = v
            self.doc.recompute()
    
    def _create_direct_dimensions(self):
        """Create dimensions on first available edges."""
        for view_name, view in self.views.items():
            # Create width and height dimensions
            for i, suffix in enumerate(['W', 'H']):
                dim_name = f'D_{view_name}_{suffix}'
                try:
                    dim = self.doc.addObject('TechDraw::DrawViewDimension', dim_name)
                    dim.Type = 'Distance'
                    dim.References2D = [(view, f'Edge{i}')]
                    dim.FormatSpec = "%.1f"
                    dim.X = 0 if i == 0 else 20
                    dim.Y = 15 if i == 0 else 0
                    self.page.addView(dim)
                    self.doc.recompute()
                    
                    meas = to_float(dim.Measurement) if hasattr(dim, 'Measurement') else 0
                    if meas > 0:
                        self.dim_count += 1
                except (RuntimeError, AttributeError):
                    pass
            
            # Try diameter dimensions on right view
            if view_name == 'right':
                for i in range(2, 10):
                    dim_name = f'D_dia_{i}'
                    try:
                        dim = self.doc.addObject('TechDraw::DrawViewDimension', dim_name)
                        dim.Type = 'Diameter'
                        dim.References2D = [(view, f'Edge{i}')]
                        dim.FormatSpec = "%%c%.1f"
                        dim.X = 25
                        dim.Y = 10
                        self.page.addView(dim)
                        self.doc.recompute()
                        
                        meas = to_float(dim.Measurement) if hasattr(dim, 'Measurement') else 0
                        if meas > 0:
                            self.dim_count += 1
                            break
                    except (RuntimeError, AttributeError):
                        pass
    
    def save(self, output_path):
        """Save document."""
        self.doc.recompute()
        path = f"{output_path}.FCStd"
        self.doc.saveAs(path)
        return path


# =============================================================================
# DIMENSION DRAWER (for PDF)
# =============================================================================

class DimensionDrawer:
    """Draws dimension annotations on matplotlib axes."""
    
    def __init__(self, ax, color='#0066CC'):
        self.ax = ax
        self.color = color
    
    def horizontal(self, x1, x2, y_ref, value, offset=12):
        """Draw horizontal dimension with extension lines."""
        y = y_ref + offset
        
        # Extension lines
        self.ax.plot([x1, x1], [y_ref + 2, y + 3], color=self.color, lw=0.6)
        self.ax.plot([x2, x2], [y_ref + 2, y + 3], color=self.color, lw=0.6)
        
        # Dimension line
        self.ax.plot([x1, x2], [y, y], color=self.color, lw=0.6)
        
        # Arrows
        self.ax.annotate('', xy=(x1, y), xytext=(x1 + 3, y),
                        arrowprops=dict(arrowstyle='->', color=self.color, lw=0.6))
        self.ax.annotate('', xy=(x2, y), xytext=(x2 - 3, y),
                        arrowprops=dict(arrowstyle='->', color=self.color, lw=0.6))
        
        # Value text
        self.ax.text((x1 + x2) / 2, y, f'{value:.1f}',
                    ha='center', va='center', fontsize=8,
                    color=self.color, fontweight='bold',
                    bbox=dict(fc='white', ec='none', pad=1))
    
    def vertical(self, y1, y2, x_ref, value, offset=12):
        """Draw vertical dimension with extension lines."""
        x = x_ref + offset
        
        # Extension lines
        self.ax.plot([x_ref + 2, x + 3], [y1, y1], color=self.color, lw=0.6)
        self.ax.plot([x_ref + 2, x + 3], [y2, y2], color=self.color, lw=0.6)
        
        # Dimension line
        self.ax.plot([x, x], [y1, y2], color=self.color, lw=0.6)
        
        # Arrows
        self.ax.annotate('', xy=(x, y1), xytext=(x, y1 + 3),
                        arrowprops=dict(arrowstyle='->', color=self.color, lw=0.6))
        self.ax.annotate('', xy=(x, y2), xytext=(x, y2 - 3),
                        arrowprops=dict(arrowstyle='->', color=self.color, lw=0.6))
        
        # Value text
        self.ax.text(x, (y1 + y2) / 2, f'{value:.1f}',
                    ha='center', va='center', fontsize=8,
                    color=self.color, fontweight='bold', rotation=90,
                    bbox=dict(fc='white', ec='none', pad=1))
    
    def diameter(self, center, radius, value, count=1, angle=45):
        """Draw diameter dimension with leader line."""
        cx, cy = center
        ang = np.radians(angle)
        
        # Point on circle
        px = cx + radius * np.cos(ang)
        py = cy + radius * np.sin(ang)
        
        # Leader line end
        ex = px + 20 * np.cos(ang)
        ey = py + 20 * np.sin(ang)
        
        # Draw leader
        self.ax.plot([px, ex], [py, ey], color=self.color, lw=0.6)
        self.ax.plot([ex, ex + 8], [ey, ey], color=self.color, lw=0.6)
        
        # Dimension text
        txt = f'⌀{value:.1f}' + (f' ({count}x)' if count > 1 else '')
        self.ax.text(ex + 10, ey, txt, ha='left', va='center',
                    fontsize=7, color=self.color, fontweight='bold')


# =============================================================================
# PDF RENDERER
# =============================================================================

class PDFRenderer:
    """Renders technical drawing to PDF using matplotlib."""
    
    def __init__(self, analysis, title_data):
        self.analysis = analysis
        self.title_data = title_data
        self.views = {}
        self.dim_count = 0
    
    def add_view(self, view):
        """Add projected view for rendering."""
        self.views[view.name] = view
    
    def render(self, output_path):
        """Render complete technical drawing to PDF and PNG."""
        fig = plt.figure(figsize=(16, 11))
        fig.suptitle('TECHNICAL DRAWING', fontsize=16, fontweight='bold')
        
        # Create grid layout
        gs = fig.add_gridspec(2, 3, height_ratios=[0.3, 1], 
                             hspace=0.25, wspace=0.25)
        
        ax_title = fig.add_subplot(gs[0, :])
        ax_front = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[1, 1])
        ax_right = fig.add_subplot(gs[1, 2])
        
        # Render each section
        self._render_title(ax_title)
        self._render_view(ax_front, 'front', 'FRONT')
        self._render_view(ax_top, 'top', 'TOP')
        self._render_view(ax_right, 'right', 'RIGHT')
        
        # Save outputs
        pdf_path = f"{output_path}.pdf"
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        
        png_path = f"{output_path}.png"
        plt.savefig(png_path, format='png', dpi=150, bbox_inches='tight')
        
        plt.close()
        return pdf_path
    
    def _render_title(self, ax):
        """Render title block with part information."""
        ax.axis('off')
        td = self.title_data
        
        txt = (f"Part: {td.part_name}  |  "
               f"{td.dimensions_str}  |  "
               f"Material: {td.material}  |  "
               f"Scale: {td.scale}  |  "
               f"Date: {td.date}")
        
        ax.text(0.5, 0.5, txt, ha='center', va='center', fontsize=10,
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='#333'))
    
    def _render_view(self, ax, view_name, title):
        """Render single orthographic view with dimensions."""
        if view_name not in self.views:
            ax.set_title(title)
            return
        
        view = self.views[view_name]
        w, h = view.bounds
        bbox = self.analysis.bbox
        
        # Draw edges
        for edge in view.edges:
            if len(edge) > 1:
                ax.plot(edge[:, 0], edge[:, 1], 'k-', lw=0.8)
        
        # Draw circles (holes)
        if view.circles:
            for (cx, cy), r in view.circles:
                ax.add_patch(plt.Circle((cx, cy), r, fill=False, color='k', lw=0.6))
        
        # Add dimensions
        dim = DimensionDrawer(ax)
        
        if view_name == 'front':
            dim.horizontal(0, w, h, bbox['dx'], 10)
            dim.vertical(0, h, w, bbox['dz'], 12)
            self.dim_count += 2
        elif view_name == 'top':
            dim.horizontal(0, w, h, bbox['dx'], 10)
            dim.vertical(0, h, w, bbox['dy'], 12)
            self.dim_count += 2
        elif view_name == 'right':
            dim.horizontal(0, w, h, bbox['dy'], 10)
            dim.vertical(0, h, w, bbox['dz'], 12)
            self.dim_count += 2
            
            # Add diameter dimensions for holes
            if view.circles:
                grouped = defaultdict(list)
                for c, r in view.circles:
                    grouped[round(r * 2, 1)].append((c, r))
                
                angle = 30
                for diameter, items in sorted(grouped.items(), reverse=True):
                    dim.diameter(items[0][0], items[0][1], diameter, len(items), angle)
                    self.dim_count += 1
                    angle += 50
        
        # Set view bounds with padding
        pad = max(w, h) * 0.2
        ax.set_xlim(-pad, w + pad * 1.5)
        ax.set_ylim(-pad, h + pad * 1.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.2)


# =============================================================================
# MAIN AGENT CLASS
# =============================================================================

class AIDraftingAgent:
    """
    Main orchestrator for the AI Drafting Agent.
    Coordinates geometry analysis, TechDraw generation, and PDF rendering.
    """
    
    def run(self, input_file, output_base):
        """
        Execute complete drafting workflow.
        
        Args:
            input_file: Path to input CAD file (STEP, IGES, FCStd)
            output_base: Base path for output files (without extension)
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Import CAD file
        doc = FreeCAD.newDocument("Drawing")
        Part.insert(input_file, doc.Name)
        
        # Find part with valid shape
        part = None
        for obj in doc.Objects:
            if hasattr(obj, 'Shape') and obj.Shape.Volume > 0:
                part = obj
                break
        
        if not part:
            raise ValueError("No valid shape found in input file")
        
        # Analyze geometry
        analyzer = GeometryAnalyzer(part.Shape)
        analysis = analyzer.analyze()
        
        # Generate title block data
        title_data = TitleBlockGenerator().generate(input_file, analysis)
        
        # Create TechDraw document
        techdraw = TechDrawGeneratorVertex(doc, part, analysis, title_data)
        techdraw.generate()
        
        # Fallback to simple method if no dimensions created
        if techdraw.dim_count == 0:
            techdraw2 = TechDrawSimpleDirect(doc, part, analysis, title_data)
            techdraw2.generate()
            techdraw.dim_count = techdraw2.dim_count
        
        techdraw.save(output_base)
        
        # Generate PDF
        projector = ViewProjector(part.Shape)
        renderer = PDFRenderer(analysis, title_data)
        
        # Project standard views
        for view_name in ['front', 'top']:
            view = projector.project(view_name)
            renderer.add_view(view)
        
        # Right view with circle detection for holes
        right_view = projector.project('right', include_circles=True)
        renderer.add_view(right_view)
        
        renderer.render(output_base)
        
        # Return summary
        return {
            'part_name': title_data.part_name,
            'dimensions': f"{analysis.bbox['dx']:.1f} x {analysis.bbox['dy']:.1f} x {analysis.bbox['dz']:.1f}",
            'techdraw_dims': techdraw.dim_count,
            'pdf_dims': renderer.dim_count,
            'files': [f"{output_base}.FCStd", f"{output_base}.pdf", f"{output_base}.png"]
        }


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Parse command line arguments and run agent."""
    args = sys.argv
    
    # Extract script arguments (handle FreeCAD's argument passing)
    script_args = []
    for i, arg in enumerate(args):
        if arg.endswith('.py'):
            script_args = args[i + 1:]
            break
    
    if len(script_args) < 2:
        print("Usage: freecadcmd ai_drafting_agent.py <input.step> <output_name>")
        sys.exit(1)
    
    input_file = os.path.abspath(script_args[0])
    output_base = os.path.abspath(script_args[1])
    
    agent = AIDraftingAgent()
    result = agent.run(input_file, output_base)
    
    print(f"\nCompleted: {result['part_name']}")
    print(f"Size: {result['dimensions']} mm")
    print("PDF PNG TechDraw files generated!")


if __name__ == '__main__':
    main()
else:
    main()