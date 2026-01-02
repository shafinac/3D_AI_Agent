#!/usr/bin/env python3
"""
================================================================================
AI DRAFTING AGENT - FINAL WORKING VERSION
================================================================================

Creates TechDraw dimensions by directly referencing edges without validation.
The key is to NOT test/remove - just create and let FreeCAD compute.

Usage:
  freecadcmd ai_drafting_agent.py <input.step> <output_name>

================================================================================
"""

import sys
import os
import math
import numpy as np
from datetime import datetime
from collections import defaultdict
import time

print("=" * 70)
print("  AI DRAFTING AGENT - Final Working Version")
print("=" * 70)

try:
    import FreeCAD
    print(f"✓ FreeCAD {FreeCAD.Version()[0]}.{FreeCAD.Version()[1]}")
except ImportError as e:
    print(f"✗ FreeCAD not found: {e}")
    sys.exit(1)

try:
    import Part
    print("✓ Part module")
except ImportError:
    sys.exit(1)

try:
    import TechDraw
    print("✓ TechDraw module")
except ImportError:
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    print("✓ Matplotlib")
except ImportError:
    sys.exit(1)


def to_float(val):
    if val is None:
        return 0.0
    if hasattr(val, 'Value'):
        return float(val.Value)
    try:
        return float(val)
    except:
        return 0.0


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class AnalysisResult:
    def __init__(self):
        self.bbox = {}
        self.volume = 0.0
        self.surface_area = 0.0
        self.hole_count = 0
        self.hole_diameters = []
        self.hole_positions = {}
        self.face_count = 0
        self.edge_count = 0


class TitleBlockData:
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
    def __init__(self, name):
        self.name = name
        self.edges = []
        self.circles = []
        self.bounds = (0, 0)


# ============================================================================
# GEOMETRY ANALYZER
# ============================================================================

class GeometryAnalyzer:
    def __init__(self, shape):
        self.shape = shape
    
    def analyze(self):
        print("\n[ANALYSIS]")
        result = AnalysisResult()
        
        bbox = self.shape.BoundBox
        result.bbox = {
            'dx': bbox.XLength, 'dy': bbox.YLength, 'dz': bbox.ZLength,
            'max_dim': max(bbox.XLength, bbox.YLength, bbox.ZLength),
        }
        print(f"  Size: {result.bbox['dx']:.1f} x {result.bbox['dy']:.1f} x {result.bbox['dz']:.1f} mm")
        
        result.volume = self.shape.Volume
        result.surface_area = self.shape.Area
        result.face_count = len(self.shape.Faces)
        result.edge_count = len(self.shape.Edges)
        
        # Detect holes
        for face in self.shape.Faces:
            try:
                if 'Cylinder' in face.Surface.TypeId:
                    d = round(face.Surface.Radius * 2, 1)
                    if d not in result.hole_diameters:
                        result.hole_diameters.append(d)
                    result.hole_count += 1
            except:
                pass
        
        result.hole_diameters.sort()
        print(f"  Holes: {result.hole_count}, Diameters: {result.hole_diameters}")
        
        return result


# ============================================================================
# TITLE BLOCK
# ============================================================================

class TitleBlockGenerator:
    def generate(self, filepath, analysis):
        data = TitleBlockData()
        name = os.path.splitext(os.path.basename(filepath))[0]
        data.part_name = name.upper().replace('_', ' ')
        data.part_number = f"{name.upper()[:6]}-001"
        data.weight = f"{analysis.volume * 2.7e-6:.3f} kg"
        
        max_dim = analysis.bbox['max_dim']
        if max_dim > 200:
            data.scale, data.scale_factor = "1:5", 0.2
        elif max_dim > 100:
            data.scale, data.scale_factor = "1:2", 0.5
        else:
            data.scale, data.scale_factor = "1:1", 1.0
        
        data.dimensions_str = f"{analysis.bbox['dx']:.1f} x {analysis.bbox['dy']:.1f} x {analysis.bbox['dz']:.1f}"
        return data


# ============================================================================
# VIEW PROJECTOR (for PDF)
# ============================================================================

class ViewProjector:
    PROJ = {
        'front': lambda x, y, z: (x, z),
        'top': lambda x, y, z: (x, y),
        'right': lambda x, y, z: (-y, z),
    }
    
    def __init__(self, shape):
        self.shape = shape
    
    def project(self, view_name, include_circles=False):
        pf = self.PROJ[view_name]
        view = ProjectedView(view_name)
        
        for edge in self.shape.Edges:
            try:
                pts = edge.discretize(Deflection=0.3)
                if len(pts) > 1:
                    view.edges.append(np.array([pf(p.x, p.y, p.z) for p in pts]))
            except:
                pass
        
        if include_circles:
            for face in self.shape.Faces:
                try:
                    if 'Cylinder' in face.Surface.TypeId:
                        c = face.Surface.Center
                        view.circles.append((pf(c.x, c.y, c.z), face.Surface.Radius))
                except:
                    pass
        
        if view.edges:
            all_pts = np.vstack(view.edges)
            min_pt, max_pt = all_pts.min(axis=0), all_pts.max(axis=0)
            view.edges = [e - min_pt for e in view.edges]
            if view.circles:
                view.circles = [((c[0]-min_pt[0], c[1]-min_pt[1]), r) for c, r in view.circles]
            view.bounds = (max_pt[0]-min_pt[0], max_pt[1]-min_pt[1])
        
        return view


# ============================================================================
# TECHDRAW GENERATOR - FINAL WORKING VERSION
# ============================================================================

class TechDrawGeneratorFinal:
    """
    Creates dimensions by directly adding them without validation.
    Key insight: Just create the dimensions and let FreeCAD compute them.
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
        print("\n[TECHDRAW]")
        FreeCAD.setActiveDocument(self.doc.Name)
        
        self._create_page()
        self._create_views()
        
        # Wait for views to compute
        print("  Computing views...")
        for _ in range(8):
            self.doc.recompute()
        time.sleep(0.5)
        
        # Get edge info from source geometry
        print("  Analyzing source geometry for dimensions...")
        self._add_dimensions_from_geometry()
        
        self.doc.recompute()
        print(f"  ✓ Created {self.dim_count} dimensions")
        return self.page
    
    def _create_page(self):
        self.page = self.doc.addObject('TechDraw::DrawPage', 'Page')
        tpl = self.doc.addObject('TechDraw::DrawSVGTemplate', 'Template')
        
        res = FreeCAD.getResourceDir()
        for t in ["A4_LandscapeTD.svg", "A3_LandscapeTD.svg"]:
            for s in ["Mod/TechDraw/Templates", "share/Mod/TechDraw/Templates"]:
                p = os.path.join(res, s, t)
                if os.path.exists(p):
                    tpl.Template = p
                    print(f"  ✓ Template: {t}")
                    break
            else:
                continue
            break
        
        self.page.Template = tpl
        self.doc.recompute()
    
    def _create_views(self):
        cfgs = [
            ('front', FreeCAD.Vector(0, -1, 0), FreeCAD.Vector(1, 0, 0)),
            ('top', FreeCAD.Vector(0, 0, 1), FreeCAD.Vector(1, 0, 0)),
            ('right', FreeCAD.Vector(1, 0, 0), FreeCAD.Vector(0, 1, 0)),
        ]
        
        for name, dir, xdir in cfgs:
            v = self.doc.addObject('TechDraw::DrawViewPart', f'View_{name}')
            self.page.addView(v)
            v.Source = [self.part]
            v.Direction = dir
            v.XDirection = xdir
            v.X, v.Y = self.VIEW_POS[name]
            v.Scale = self.scale
            v.CoarseView = False
            self.views[name] = v
            self.doc.recompute()
            print(f"  ✓ {name} view")
    
    def _add_dimensions_from_geometry(self):
        """
        Analyze the 3D geometry to find edges, then map them to TechDraw edge indices.
        """
        shape = self.part.Shape
        bbox = self.analysis.bbox
        
        # Analyze 3D edges
        edge_info = []
        for i, edge in enumerate(shape.Edges):
            try:
                if len(edge.Vertexes) >= 2:
                    p1 = edge.Vertexes[0].Point
                    p2 = edge.Vertexes[1].Point
                    
                    dx = abs(p2.x - p1.x)
                    dy = abs(p2.y - p1.y)
                    dz = abs(p2.z - p1.z)
                    length = edge.Length
                    
                    # Classify edge direction
                    if dx > dy and dx > dz:
                        direction = 'X'
                    elif dy > dx and dy > dz:
                        direction = 'Y'
                    else:
                        direction = 'Z'
                    
                    edge_info.append({
                        'index': i,
                        'length': length,
                        'direction': direction,
                        'dx': dx, 'dy': dy, 'dz': dz
                    })
            except:
                pass
        
        print(f"    Found {len(edge_info)} 3D edges")
        
        # Find edges matching bounding box dimensions
        tolerance = 1.0
        
        x_edges = [e for e in edge_info if e['direction'] == 'X' and abs(e['length'] - bbox['dx']) < tolerance]
        y_edges = [e for e in edge_info if e['direction'] == 'Y' and abs(e['length'] - bbox['dy']) < tolerance]
        z_edges = [e for e in edge_info if e['direction'] == 'Z' and abs(e['length'] - bbox['dz']) < tolerance]
        
        print(f"    Matching edges - X:{len(x_edges)}, Y:{len(y_edges)}, Z:{len(z_edges)}")
        
        # Create dimensions for each view
        for view_name, view in self.views.items():
            print(f"\n    [{view_name.upper()}]")
            
            if view_name == 'front':
                # Front view shows X (width) and Z (height)
                self._create_dim_for_edges(view, x_edges, 'W', 0, 15)
                self._create_dim_for_edges(view, z_edges, 'H', 20, 0)
            elif view_name == 'top':
                # Top view shows X (width) and Y (depth)
                self._create_dim_for_edges(view, x_edges, 'W', 0, 15)
                self._create_dim_for_edges(view, y_edges, 'H', 20, 0)
            elif view_name == 'right':
                # Right view shows Y (width) and Z (height)
                self._create_dim_for_edges(view, y_edges, 'W', 0, 15)
                self._create_dim_for_edges(view, z_edges, 'H', 20, 0)
                # Add diameter dimensions
                self._create_diameter_dims(view)
    
    def _create_dim_for_edges(self, view, edges, suffix, offset_x, offset_y):
        """Create dimension using first matching edge"""
        
        if not edges:
            print(f"      No edges for {suffix}")
            return
        
        view_name = view.Name.replace('View_', '')
        
        # Try multiple edge indices
        for edge in edges[:5]:  # Try first 5 matching edges
            edge_idx = edge['index']
            dim_name = f'Dim_{view_name}_{suffix}'
            
            if self.doc.getObject(dim_name):
                continue
            
            try:
                dim = self.doc.addObject('TechDraw::DrawViewDimension', dim_name)
                dim.Type = 'Distance'
                dim.References2D = [(view, f'Edge{edge_idx}')]
                dim.FormatSpec = "%.1f"
                dim.X = offset_x
                dim.Y = offset_y
                
                if hasattr(dim, 'Arbitrary'):
                    dim.Arbitrary = False
                
                self.page.addView(dim)
                self.doc.recompute()
                
                # Check if dimension was created successfully
                if hasattr(dim, 'Measurement'):
                    meas = to_float(dim.Measurement)
                    if meas > 0:
                        self.dim_count += 1
                        print(f"      ✓ {suffix}: {meas:.1f}mm (Edge{edge_idx})")
                        return
                    else:
                        # Remove failed dimension
                        self.page.removeView(dim)
                        self.doc.removeObject(dim_name)
                
            except Exception as e:
                try:
                    self.doc.removeObject(dim_name)
                except:
                    pass
        
        print(f"      ✗ Could not create {suffix} dimension")
    
    def _create_diameter_dims(self, view):
        """Create diameter dimensions for holes"""
        
        shape = self.part.Shape
        added = set()
        
        # Find circular edges
        for i, edge in enumerate(shape.Edges):
            try:
                curve = edge.Curve
                if 'Circle' in type(curve).__name__:
                    dia = round(curve.Radius * 2, 1)
                    
                    if dia in added:
                        continue
                    
                    dim_name = f'Dim_Dia_{int(dia*10)}'
                    
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
                    
                    if hasattr(dim, 'Measurement') and to_float(dim.Measurement) > 0:
                        added.add(dia)
                        self.dim_count += 1
                        print(f"      ✓ Ø{dia}mm (Edge{i})")
                    else:
                        self.page.removeView(dim)
                        self.doc.removeObject(dim_name)
                        
            except:
                pass
    
    def save(self, output_path):
        self.doc.recompute()
        path = f"{output_path}.FCStd"
        self.doc.saveAs(path)
        print(f"\n  ✓ Saved: {path}")
        return path


# ============================================================================
# ALTERNATIVE: USE VERTEX-BASED DIMENSIONS
# ============================================================================

class TechDrawGeneratorVertex:
    """
    Alternative approach: Use vertices instead of edges for dimensions.
    DistanceX/DistanceY with two vertices is more reliable.
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
        print("\n[TECHDRAW - Vertex Method]")
        FreeCAD.setActiveDocument(self.doc.Name)
        
        self._create_page()
        self._create_views()
        
        print("  Computing views...")
        for _ in range(10):
            self.doc.recompute()
        time.sleep(0.5)
        
        # Add dimensions using vertices
        self._add_vertex_dimensions()
        
        self.doc.recompute()
        print(f"\n  ✓ Created {self.dim_count} dimensions")
        return self.page
    
    def _create_page(self):
        self.page = self.doc.addObject('TechDraw::DrawPage', 'Page')
        tpl = self.doc.addObject('TechDraw::DrawSVGTemplate', 'Template')
        
        res = FreeCAD.getResourceDir()
        for t in ["A4_LandscapeTD.svg", "A3_LandscapeTD.svg"]:
            for s in ["Mod/TechDraw/Templates", "share/Mod/TechDraw/Templates"]:
                p = os.path.join(res, s, t)
                if os.path.exists(p):
                    tpl.Template = p
                    break
            else:
                continue
            break
        
        self.page.Template = tpl
        self.doc.recompute()
    
    def _create_views(self):
        cfgs = [
            ('front', FreeCAD.Vector(0, -1, 0), FreeCAD.Vector(1, 0, 0)),
            ('top', FreeCAD.Vector(0, 0, 1), FreeCAD.Vector(1, 0, 0)),
            ('right', FreeCAD.Vector(1, 0, 0), FreeCAD.Vector(0, 1, 0)),
        ]
        
        for name, dir, xdir in cfgs:
            v = self.doc.addObject('TechDraw::DrawViewPart', f'View_{name}')
            self.page.addView(v)
            v.Source = [self.part]
            v.Direction = dir
            v.XDirection = xdir
            v.X, v.Y = self.VIEW_POS[name]
            v.Scale = self.scale
            v.CoarseView = False
            self.views[name] = v
            self.doc.recompute()
            print(f"  ✓ {name} view")
    
    def _add_vertex_dimensions(self):
        """Add dimensions using vertex pairs"""
        
        bbox = self.analysis.bbox
        
        for view_name, view in self.views.items():
            print(f"\n  [{view_name.upper()}]")
            
            # Get expected dimensions for this view
            if view_name == 'front':
                w_exp, h_exp = bbox['dx'], bbox['dz']
            elif view_name == 'top':
                w_exp, h_exp = bbox['dx'], bbox['dy']
            else:
                w_exp, h_exp = bbox['dy'], bbox['dz']
            
            # Try to find vertex pairs that give correct distances
            w_found = self._find_and_create_vertex_dim(view, view_name, 'DistanceX', w_exp, 'W', 0, 18)
            h_found = self._find_and_create_vertex_dim(view, view_name, 'DistanceY', h_exp, 'H', 22, 0)
            
            # Fallback: try edge-based dimensions
            if not w_found:
                self._try_edge_dimension(view, view_name, 'W', w_exp, 0, 18)
            if not h_found:
                self._try_edge_dimension(view, view_name, 'H', h_exp, 22, 0)
            
            # Diameter dimensions for right view
            if view_name == 'right':
                self._add_diameter_dims(view)
    
    def _find_and_create_vertex_dim(self, view, view_name, dim_type, expected, suffix, off_x, off_y):
        """Find vertex pair with correct distance and create dimension"""
        
        dim_name = f'Dim_{view_name}_{suffix}'
        tolerance = 1.0
        
        # Try vertex pairs
        for i in range(30):
            for j in range(i+1, 30):
                try:
                    test_name = f'_vtest_{i}_{j}'
                    test = self.doc.addObject('TechDraw::DrawViewDimension', test_name)
                    test.Type = dim_type
                    test.References2D = [(view, f'Vertex{i}'), (view, f'Vertex{j}')]
                    self.page.addView(test)
                    self.doc.recompute()
                    
                    meas = to_float(test.Measurement) if hasattr(test, 'Measurement') else 0
                    
                    self.page.removeView(test)
                    self.doc.removeObject(test_name)
                    
                    if abs(meas - expected) < tolerance:
                        # Found it! Create the real dimension
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
                        print(f"    ✓ {suffix}: {meas:.1f}mm (V{i}-V{j})")
                        return True
                        
                except:
                    try:
                        self.doc.removeObject(f'_vtest_{i}_{j}')
                    except:
                        pass
        
        return False
    
    def _try_edge_dimension(self, view, view_name, suffix, expected, off_x, off_y):
        """Fallback: try edge-based dimension"""
        
        dim_name = f'Dim_{view_name}_{suffix}'
        tolerance = 1.0
        
        for i in range(50):
            try:
                test_name = f'_etest_{i}'
                test = self.doc.addObject('TechDraw::DrawViewDimension', test_name)
                test.Type = 'Distance'
                test.References2D = [(view, f'Edge{i}')]
                self.page.addView(test)
                self.doc.recompute()
                
                meas = to_float(test.Measurement) if hasattr(test, 'Measurement') else 0
                
                self.page.removeView(test)
                self.doc.removeObject(test_name)
                
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
                    print(f"    ✓ {suffix}: {meas:.1f}mm (Edge{i})")
                    return True
                    
            except:
                try:
                    self.doc.removeObject(f'_etest_{i}')
                except:
                    pass
        
        print(f"    ✗ {suffix} not found")
        return False
    
    def _add_diameter_dims(self, view):
        """Add diameter dimensions"""
        
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
                    dia = round(meas, 1)
                    if dia not in added:
                        dim_name = f'Dim_D{int(dia*10)}'
                        dim = self.doc.addObject('TechDraw::DrawViewDimension', dim_name)
                        dim.Type = 'Diameter'
                        dim.References2D = [(view, f'Edge{i}')]
                        dim.FormatSpec = "%%c%.1f"
                        dim.X = 25 + len(added)*12
                        dim.Y = 10
                        
                        if hasattr(dim, 'Arbitrary'):
                            dim.Arbitrary = False
                        
                        self.page.addView(dim)
                        self.doc.recompute()
                        
                        added.add(dia)
                        self.dim_count += 1
                        print(f"    ✓ Ø{dia}mm")
                        
            except:
                try:
                    self.doc.removeObject(f'_dtest_{i}')
                except:
                    pass
    
    def save(self, output_path):
        self.doc.recompute()
        path = f"{output_path}.FCStd"
        self.doc.saveAs(path)
        print(f"\n  ✓ Saved: {path}")
        return path


# ============================================================================
# SIMPLE DIRECT METHOD - NO TESTING
# ============================================================================

class TechDrawSimpleDirect:
    """
    Simplest possible approach: Create dimensions directly on first few edges.
    No testing, no validation - just create them.
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
        print("\n[TECHDRAW - Simple Direct]")
        FreeCAD.setActiveDocument(self.doc.Name)
        
        self._create_page()
        self._create_views()
        
        print("  Computing...")
        for _ in range(10):
            self.doc.recompute()
        
        # Just create dimensions on edges 0, 1, 2, etc.
        self._create_direct_dimensions()
        
        self.doc.recompute()
        print(f"\n  ✓ Created {self.dim_count} dimensions")
        return self.page
    
    def _create_page(self):
        self.page = self.doc.addObject('TechDraw::DrawPage', 'Page')
        tpl = self.doc.addObject('TechDraw::DrawSVGTemplate', 'Template')
        
        res = FreeCAD.getResourceDir()
        for t in ["A4_LandscapeTD.svg"]:
            for s in ["Mod/TechDraw/Templates", "share/Mod/TechDraw/Templates"]:
                p = os.path.join(res, s, t)
                if os.path.exists(p):
                    tpl.Template = p
                    break
        
        self.page.Template = tpl
        self.doc.recompute()
    
    def _create_views(self):
        cfgs = [
            ('front', FreeCAD.Vector(0, -1, 0), FreeCAD.Vector(1, 0, 0)),
            ('top', FreeCAD.Vector(0, 0, 1), FreeCAD.Vector(1, 0, 0)),
            ('right', FreeCAD.Vector(1, 0, 0), FreeCAD.Vector(0, 1, 0)),
        ]
        
        for name, dir, xdir in cfgs:
            v = self.doc.addObject('TechDraw::DrawViewPart', f'View_{name}')
            self.page.addView(v)
            v.Source = [self.part]
            v.Direction = dir
            v.XDirection = xdir
            v.X, v.Y = self.VIEW_POS[name]
            v.Scale = self.scale
            v.CoarseView = False
            self.views[name] = v
            self.doc.recompute()
            print(f"  ✓ {name}")
    
    def _create_direct_dimensions(self):
        """Create dimensions directly without testing"""
        
        for view_name, view in self.views.items():
            print(f"\n  [{view_name}]")
            
            # Create 2 Distance dimensions on edges 0 and 1
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
                        print(f"    ✓ {suffix}: {meas:.1f}mm")
                    else:
                        print(f"    ✗ {suffix}: no measurement")
                except Exception as e:
                    print(f"    ✗ {suffix}: {e}")
            
            # Try diameter on edge 2 for right view
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
                            print(f"    ✓ Ø{meas:.1f}mm")
                            break
                    except:
                        pass
    
    def save(self, output_path):
        self.doc.recompute()
        path = f"{output_path}.FCStd"
        self.doc.saveAs(path)
        print(f"  ✓ Saved: {path}")
        return path


# ============================================================================
# PDF RENDERER
# ============================================================================

class DimensionDrawer:
    def __init__(self, ax, color='#0066CC'):
        self.ax = ax
        self.color = color
    
    def horizontal(self, x1, x2, y_ref, value, offset=12):
        y = y_ref + offset
        self.ax.plot([x1, x1], [y_ref+2, y+3], color=self.color, lw=0.6)
        self.ax.plot([x2, x2], [y_ref+2, y+3], color=self.color, lw=0.6)
        self.ax.plot([x1, x2], [y, y], color=self.color, lw=0.6)
        self.ax.annotate('', xy=(x1, y), xytext=(x1+3, y), arrowprops=dict(arrowstyle='->', color=self.color, lw=0.6))
        self.ax.annotate('', xy=(x2, y), xytext=(x2-3, y), arrowprops=dict(arrowstyle='->', color=self.color, lw=0.6))
        self.ax.text((x1+x2)/2, y, f'{value:.1f}', ha='center', va='center', fontsize=8, color=self.color, fontweight='bold', bbox=dict(fc='white', ec='none', pad=1))
    
    def vertical(self, y1, y2, x_ref, value, offset=12):
        x = x_ref + offset
        self.ax.plot([x_ref+2, x+3], [y1, y1], color=self.color, lw=0.6)
        self.ax.plot([x_ref+2, x+3], [y2, y2], color=self.color, lw=0.6)
        self.ax.plot([x, x], [y1, y2], color=self.color, lw=0.6)
        self.ax.annotate('', xy=(x, y1), xytext=(x, y1+3), arrowprops=dict(arrowstyle='->', color=self.color, lw=0.6))
        self.ax.annotate('', xy=(x, y2), xytext=(x, y2-3), arrowprops=dict(arrowstyle='->', color=self.color, lw=0.6))
        self.ax.text(x, (y1+y2)/2, f'{value:.1f}', ha='center', va='center', fontsize=8, color=self.color, fontweight='bold', rotation=90, bbox=dict(fc='white', ec='none', pad=1))
    
    def diameter(self, center, radius, value, count=1, angle=45):
        cx, cy = center
        ang = np.radians(angle)
        px, py = cx + radius*np.cos(ang), cy + radius*np.sin(ang)
        ex, ey = px + 20*np.cos(ang), py + 20*np.sin(ang)
        self.ax.plot([px, ex], [py, ey], color=self.color, lw=0.6)
        self.ax.plot([ex, ex+8], [ey, ey], color=self.color, lw=0.6)
        txt = f'⌀{value:.1f}' + (f' ({count}x)' if count > 1 else '')
        self.ax.text(ex+10, ey, txt, ha='left', va='center', fontsize=7, color=self.color, fontweight='bold')


class PDFRenderer:
    def __init__(self, analysis, title_data):
        self.analysis = analysis
        self.title_data = title_data
        self.views = {}
        self.dim_count = 0
    
    def add_view(self, view):
        self.views[view.name] = view
    
    def render(self, output_path):
        print("\n[PDF]")
        
        fig = plt.figure(figsize=(16, 11))
        fig.suptitle('TECHNICAL DRAWING', fontsize=16, fontweight='bold')
        
        gs = fig.add_gridspec(2, 3, height_ratios=[0.3, 1], hspace=0.25, wspace=0.25)
        
        ax_title = fig.add_subplot(gs[0, :])
        ax_front = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[1, 1])
        ax_right = fig.add_subplot(gs[1, 2])
        
        self._render_title(ax_title)
        self._render_view(ax_front, 'front', 'FRONT')
        self._render_view(ax_top, 'top', 'TOP')
        self._render_view(ax_right, 'right', 'RIGHT')
        
        pdf_path = f"{output_path}.pdf"
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"  ✓ {pdf_path}")
        
        png_path = f"{output_path}.png"
        plt.savefig(png_path, format='png', dpi=150, bbox_inches='tight')
        print(f"  ✓ {png_path}")
        
        plt.close()
        return pdf_path
    
    def _render_title(self, ax):
        ax.axis('off')
        td = self.title_data
        txt = f"Part: {td.part_name}  |  {td.dimensions_str}  |  Material: {td.material}  |  Scale: {td.scale}  |  Date: {td.date}"
        ax.text(0.5, 0.5, txt, ha='center', va='center', fontsize=10, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='#333'))
    
    def _render_view(self, ax, view_name, title):
        if view_name not in self.views:
            ax.set_title(title)
            return
        
        view = self.views[view_name]
        w, h = view.bounds
        bbox = self.analysis.bbox
        
        for edge in view.edges:
            if len(edge) > 1:
                ax.plot(edge[:,0], edge[:,1], 'k-', lw=0.8)
        
        if view.circles:
            for (cx, cy), r in view.circles:
                ax.add_patch(plt.Circle((cx, cy), r, fill=False, color='k', lw=0.6))
        
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
            
            if view.circles:
                grp = defaultdict(list)
                for c, r in view.circles:
                    grp[round(r*2, 1)].append((c, r))
                ang = 30
                for d, items in sorted(grp.items(), reverse=True):
                    dim.diameter(items[0][0], items[0][1], d, len(items), ang)
                    self.dim_count += 1
                    ang += 50
        
        pad = max(w, h) * 0.2
        ax.set_xlim(-pad, w + pad*1.5)
        ax.set_ylim(-pad, h + pad*1.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.2)


# ============================================================================
# MAIN
# ============================================================================

class AIDraftingAgent:
    def run(self, input_file, output_base):
        print("\n" + "="*60)
        print("  AI DRAFTING AGENT")
        print("="*60)
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(input_file)
        
        # Import
        print("\n[IMPORT]")
        doc = FreeCAD.newDocument("Drawing")
        Part.insert(input_file, doc.Name)
        
        part = None
        for obj in doc.Objects:
            if hasattr(obj, 'Shape') and obj.Shape.Volume > 0:
                part = obj
                break
        
        if not part:
            raise ValueError("No shape found")
        print(f"  ✓ {os.path.basename(input_file)}")
        
        # Analyze
        analyzer = GeometryAnalyzer(part.Shape)
        analysis = analyzer.analyze()
        
        # Metadata
        print("\n[METADATA]")
        title_data = TitleBlockGenerator().generate(input_file, analysis)
        print(f"  ✓ {title_data.part_name}")
        
        # TechDraw - Try vertex method first, then fallback
        techdraw = TechDrawGeneratorVertex(doc, part, analysis, title_data)
        techdraw.generate()
        
        # If no dimensions created, try simple direct method
        if techdraw.dim_count == 0:
            print("\n  Trying alternative method...")
            techdraw2 = TechDrawSimpleDirect(doc, part, analysis, title_data)
            techdraw2.generate()
            techdraw.dim_count = techdraw2.dim_count
        
        techdraw.save(output_base)
        
        # PDF
        print("\n[PROJECTING]")
        projector = ViewProjector(part.Shape)
        renderer = PDFRenderer(analysis, title_data)
        
        for vn in ['front', 'top']:
            v = projector.project(vn)
            renderer.add_view(v)
            print(f"  ✓ {vn}: {len(v.edges)} edges")
        
        v = projector.project('right', True)
        renderer.add_view(v)
        print(f"  ✓ right: {len(v.edges)} edges, {len(v.circles)} circles")
        
        renderer.render(output_base)
        
        # Summary
        print("\n" + "="*60)
        print("  DONE")
        print("="*60)
        print(f"""
  Part: {title_data.part_name}
  Size: {analysis.bbox['dx']:.1f} x {analysis.bbox['dy']:.1f} x {analysis.bbox['dz']:.1f} mm
  
  TechDraw dims: {techdraw.dim_count}
  PDF dims: {renderer.dim_count}
  
  Files: {output_base}.FCStd, .pdf, .png
""")


def main():
    args = sys.argv
    script_args = [a for i, a in enumerate(args) if i > 0 and args[i-1].endswith('.py') or (i > 0 and not args[i-1].endswith('.py') and any(x.endswith('.py') for x in args[:i]))]
    
    if not script_args:
        for i, a in enumerate(args):
            if a.endswith('.py'):
                script_args = args[i+1:]
                break
    
    if len(script_args) < 2:
        print("Usage: freecadcmd script.py <input.step> <output>")
        sys.exit(1)
    
    agent = AIDraftingAgent()
    agent.run(os.path.abspath(script_args[0]), os.path.abspath(script_args[1]))


if __name__ == '__main__':
    main()
else:
    main()