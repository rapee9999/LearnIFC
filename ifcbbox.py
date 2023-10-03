import ifcopenshell
import numpy as np
from typing import List, Tuple


def createTransformationMat(relPlacement: ifcopenshell.entity_instance) -> np.ndarray:
    """
    Create transformation matrix related to given placement.
    
    Args:
        relPlacement: Relative placement object.

    Returns:
        Transformation Numpy matrix.
    """
    # Validate.
    if not relPlacement.is_entity(): raise ValueError({'msg': f"#{relPlacement.id()} is not an entity", 'id': relPlacement.id()})
    if relPlacement.is_a() != "IfcAxis2Placement3D": raise ValueError({'msg': f"#{relPlacement.id()} is not IfcAxis2Placement3D", 'id': relPlacement.id()})
    
    # Get placement attributes.
    relLoc = relPlacement.Location[0] # relative location of placement.
    relZAx = relPlacement.Axis[0] # Z axis to relative axises.
    relXAx = relPlacement.RefDirection[0] # X axis to relative axises.
    relYAx = np.cross(relZAx, relXAx) # Y axis to relative axises.
    
    # Translation matrix.
    translateMat = np.identity(4)
    translateMat[:3, 3] = relLoc
    
    # Rotation matrix.
    rotateMat = np.identity(4)
    rotateMat[:3, :3] = np.array([relXAx, relYAx, relZAx]).T
    # rotateMat[:3, :3] = np.array([relZAx, relXAx, relYAx]).T

    # Combine translation and rotation matrix.
    transformMat = translateMat @ rotateMat
    return transformMat


def initBBox() -> np.ndarray:
    """
    Initialize bounding box (min/max).
    
    Returns:
        Numpy array of mininum and maximum coordinates to a bounding box, 
        where [min X, min Y, min Z],
              [max X, max Y, max Z]
    """
    return np.array([[float("inf"), float("inf"), float("inf")], 
                     [float("-inf"), float("-inf"), float("-inf")]])


def fitBBox(bbox: np.ndarray, point: Tuple[float, float, float]) -> np.ndarray:
    """
    Fit bounding box by given coordinates.
    
    Args:
        bbox: Dict containing min/max coordinates.
        point: Tuple of X, Y and Z coordinates.
        
    Returns:
        Updated Numpy array of mininum and maximum coordinates to a bounding box.
    """
    bbox[0][0] = min(bbox[0][0], point[0])
    bbox[0][1] = min(bbox[0][1], point[1])
    bbox[0][2] = min(bbox[0][2], point[2])
    bbox[1][0] = max(bbox[1][0], point[0])
    bbox[1][1] = max(bbox[1][1], point[1])
    bbox[1][2] = max(bbox[1][2], point[2])
    return bbox
    

def findAssemblyBBox(ifcModel: ifcopenshell.file, assemblyId: int, verbose: bool = True) -> any:
    """ 
    Find bounding box of the given assembly object.
    
    Args:
        ifcModel: IFC model.
        assemblyId: IFC's entity ID of an assembly.
        verbose: Set to print out details.
    
    Returns:
        Dict of following keys: 
            RefObjName: element name as reference placement, 
            RefObjId: element ID as reference placement,
            BBox: boudning box coordinates in min/max points,
            Dim: dimenstion in width, length and height,
            Vol: bounding box volume
    """
    elementAssembly = ifcModel.by_id(assemblyId)
    
    # Validate.
    if not elementAssembly.is_entity(): raise ValueError({'msg': f"#{assemblyId} is not an entity", 'id': assemblyId})
    if elementAssembly.is_a() != "IfcElementAssembly": raise ValueError({'msg': f"#{assemblyId} is not an element assembly", 'id': assemblyId})
    if verbose: print("ASSEMBLY:", elementAssembly)
    
    # Get parts under the given assembly.
    parts = elementAssembly.IsDecomposedBy[0].RelatedObjects

    # Result store.
    res = {
        'RefObjName': None,
        'RefObjId': None,
        'BBox': None,
        'Dim': None,
        'Vol': float("inf")
    }

    # Loop over each part as local placement.  
    for refPart in parts:
        if verbose: print("REFERENCE:", refPart)
        
        # Get transformation matrix from local reference to global placement.
        refTransformationMat = createTransformationMat(refPart.ObjectPlacement.RelativePlacement)

        # Initialize bounding box as min/max coordinates.
        bbox = initBBox()

        # Loop over each part to get min/max coordinates of bounding box.
        for part in parts:
            try:
                # Get face of object.
                faces = part.Representation.Representations[0].Items[0].Outer[0]
            except AttributeError as e:
                # No surface vertices then next part.
                continue

            # Get transformation matrix to reference placement.
            transformMat = createTransformationMat(part.ObjectPlacement.RelativePlacement)
            transformMat = np.linalg.inv(refTransformationMat) @ transformMat

            # Loop over each face.
            for face in faces:
                # Get vertices.
                vertices = face.Bounds[0].Bound[0]

                # Loop over each vertice.
                for point in vertices:
                    # Get local coordinates.
                    point = point.Coordinates
                    point = np.array(point + (1,))

                    # local coordinates on reference placement.
                    pointOnRef = transformMat @ point

                    # Fit bounding box.
                    bbox = fitBBox(bbox, pointOnRef)

        # Find minimum boudning box volume.
        dim = bbox[1] - bbox[0]
        vol = dim[0] * dim[1] * dim[2]
        if vol < res['Vol']:
            # Store smallest bouding box of whole assembly on reference placement.
            res = {
                'RefObjName': refPart.to_string(),
                'RefObjId': refPart.id(),
                'BBox': bbox,
                'Dim': dim,
                'Vol': vol
            }

    if verbose: print(f"*** Mininum bbox:\n   Reference to {res['RefObjName']}\n   Dimension: {res['Dim']}\n   Volume: {res['Vol']}\n   Boudning box:\n{res['BBox']}")
    return res
    