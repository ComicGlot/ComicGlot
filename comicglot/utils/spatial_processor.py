import numpy as np
import cv2
from typing import List, Tuple, Optional, Union, Dict, Any
from comicglot.utils.data_types import TextBlock

def boxes_overlap(box1: np.ndarray, box2: np.ndarray) -> bool:
    """
    Check if two boxes overlap at all.
    
    Args:
        box1: First box coordinates [x1, y1, x2, y2]
        box2: Second box coordinates [x1, y1, x2, y2]
        
    Returns:
        True if boxes overlap, False otherwise
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    if x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1:
        return False
    
    return True

def line_segments_intersection(
    p1: Tuple[float, float], p2: Tuple[float, float], 
    p3: Tuple[float, float], p4: Tuple[float, float]
) -> Optional[Tuple[float, float]]:
    """
    Find the intersection point of two line segments if it exists.
    
    Args:
        p1, p2: Start and end points of the first line segment
        p3, p4: Start and end points of the second line segment
        
    Returns:
        Intersection point (x, y) or None if there's no intersection
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Calculate denominators
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if abs(denom) < 1e-6:
        return None  # Parallel lines
    
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    
    # If the intersection is within both line segments
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (x, y)
    
    return None

def find_box_intersection_points(box1: np.ndarray, box2: np.ndarray) -> List[np.ndarray]:
    """
    Find all intersection points between two boxes.
    
    Args:
        box1: First box coordinates [x1, y1, x2, y2]
        box2: Second box coordinates [x1, y1, x2, y2]
        
    Returns:
        List of intersection points, each as [x, y]
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Create edges of the first box as line segments
    box1_edges = [
        ((x1_1, y1_1), (x2_1, y1_1)),  # Top edge
        ((x2_1, y1_1), (x2_1, y2_1)),  # Right edge
        ((x2_1, y2_1), (x1_1, y2_1)),  # Bottom edge
        ((x1_1, y2_1), (x1_1, y1_1))   # Left edge
    ]
    
    # Create edges of the second box as line segments
    box2_edges = [
        ((x1_2, y1_2), (x2_2, y1_2)),  # Top edge
        ((x2_2, y1_2), (x2_2, y2_2)),  # Right edge
        ((x2_2, y2_2), (x1_2, y2_2)),  # Bottom edge
        ((x1_2, y2_2), (x1_2, y1_2))   # Left edge
    ]
    
    intersection_points = []
    
    # Check for intersection between each pair of edges
    for edge1 in box1_edges:
        for edge2 in box2_edges:
            intersection = line_segments_intersection(
                edge1[0], edge1[1], edge2[0], edge2[1]
            )
            if intersection is not None:
                # Round to integers as we're dealing with pixel coordinates
                intersection_point = np.array([round(intersection[0]), round(intersection[1])], dtype=np.int32)
                
                # Check if this point is already in our list (within 1 pixel)
                is_duplicate = False
                for existing_point in intersection_points:
                    if np.all(np.abs(existing_point - intersection_point) <= 1):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    intersection_points.append(intersection_point)
    
    return intersection_points



def create_separator_line(box1: np.ndarray, box2: np.ndarray) -> Optional[np.ndarray]:
    """
    Create a separator line between two overlapping boxes by finding exact intersection points.
    
    Args:
        box1: First box coordinates [x1, y1, x2, y2]
        box2: Second box coordinates [x1, y1, x2, y2]
        
    Returns:
        Array of two points defining the separator line, or None if no valid separator can be created
    """

    if not boxes_overlap(box1, box2):
        return None
    
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    box1_contains_box2_x = x1_1 <= x1_2 and x2_2 <= x2_1
    box1_contains_box2_y = y1_1 <= y1_2 and y2_2 <= y2_1
    box2_contains_box1_x = x1_2 <= x1_1 and x2_1 <= x2_2
    box2_contains_box1_y = y1_2 <= y1_1 and y2_1 <= y2_2

    if not (box1_contains_box2_x or box1_contains_box2_y or box2_contains_box1_x or box2_contains_box1_y):
        intersection_points = find_box_intersection_points(box1, box2)
        if len(intersection_points) != 2:
            return None

        return np.array(intersection_points, dtype=np.int32)

    # Determine which box is "bigger" (contains the other along one coordinate)
    if box1_contains_box2_x or box1_contains_box2_y:
        bigger_box = box1
        smaller_box = box2
        bigger_center_x = (x1_1 + x2_1) / 2
        bigger_center_y = (y1_1 + y2_1) / 2
    else:  # box2 contains box1
        bigger_box = box2
        smaller_box = box1
        bigger_center_x = (x1_2 + x2_2) / 2
        bigger_center_y = (y1_2 + y2_2) / 2
    
    sx1, sy1, sx2, sy2 = smaller_box
    bx1, by1, bx2, by2 = bigger_box
    
     # Find the closest point on the smaller box to the center of the bigger box
    # Find distances to each edge
    left_edge_y = np.clip(bigger_center_y, sy1, sy2)
    right_edge_y = np.clip(bigger_center_y, sy1, sy2)
    top_edge_x = np.clip(bigger_center_x, sx1, sx2)
    bottom_edge_x = np.clip(bigger_center_x, sx1, sx2)
    
    # Calculate distances
    dist_left = (sx1 - bigger_center_x)**2 + (left_edge_y - bigger_center_y)**2
    dist_right = (sx2 - bigger_center_x)**2 + (right_edge_y - bigger_center_y)**2
    dist_top = (top_edge_x - bigger_center_x)**2 + (sy1 - bigger_center_y)**2
    dist_bottom = (bottom_edge_x - bigger_center_x)**2 + (sy2 - bigger_center_y)**2
    
    # Find minimum distance and corresponding point
    min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
    
    if min_dist == dist_left:
        closest_point = (sx1, left_edge_y)
    elif min_dist == dist_right:
        closest_point = (sx2, right_edge_y)
    elif min_dist == dist_top:
        closest_point = (top_edge_x, sy1)
    else:  # min_dist == dist_bottom
        closest_point = (bottom_edge_x, sy2)
    
    # Calculate vector from center to closest point
    vec_x = closest_point[0] - bigger_center_x
    vec_y = closest_point[1] - bigger_center_y
    
    # Normalize the vector
    length = np.sqrt(vec_x**2 + vec_y**2)
    if length < 1e-6:  # Avoid division by zero
        # Use default direction if too close
        vec_x, vec_y = 1.0, 0.0
    else:
        vec_x /= length
        vec_y /= length
    
    # Create perpendicular vector (rotate 90 degrees)
    perp_x = -vec_y
    perp_y = vec_x
    
    # Calculate the intersections of this perpendicular line with the borders of the bigger bubble
    # Line equation: (x - x0) * perp_x + (y - y0) * perp_y = 0
    # where (x0, y0) is closest_point
    
    # Find intersection with each edge of the bigger bubble
    intersections = []
    
    # Top edge (y = by1)
    if abs(perp_y) > 1e-6:  # Avoid division by zero
        x = closest_point[0] + (by1 - closest_point[1]) * perp_x / perp_y
        if bx1 <= x <= bx2:
            intersections.append((x, by1))
    
    # Bottom edge (y = by2)
    if abs(perp_y) > 1e-6:
        x = closest_point[0] + (by2 - closest_point[1]) * perp_x / perp_y
        if bx1 <= x <= bx2:
            intersections.append((x, by2))
    
    # Left edge (x = bx1)
    if abs(perp_x) > 1e-6:
        y = closest_point[1] + (bx1 - closest_point[0]) * perp_y / perp_x
        if by1 <= y <= by2:
            intersections.append((bx1, y))
    
    # Right edge (x = bx2)
    if abs(perp_x) > 1e-6:
        y = closest_point[1] + (bx2 - closest_point[0]) * perp_y / perp_x
        if by1 <= y <= by2:
            intersections.append((bx2, y))
    
    # We should have exactly 2 intersections with the bigger bubble
    if len(intersections) != 2:
        # Fallback to using the closest point and the center
        # and extending in that direction to the bubble boundaries
        
        # Direction from center to closest point
        dir_x = closest_point[0] - bigger_center_x
        dir_y = closest_point[1] - bigger_center_y
        
        # Find intersection of this line with bubble boundaries
        t_values = []
        
        # Top edge
        if dir_y < 0:  # Moving upward
            t = (by1 - bigger_center_y) / dir_y
            x = bigger_center_x + t * dir_x
            if bx1 <= x <= bx2:
                t_values.append(t)
        
        # Bottom edge
        if dir_y > 0:  # Moving downward
            t = (by2 - bigger_center_y) / dir_y
            x = bigger_center_x + t * dir_x
            if bx1 <= x <= bx2:
                t_values.append(t)
        
        # Left edge
        if dir_x < 0:  # Moving left
            t = (bx1 - bigger_center_x) / dir_x
            y = bigger_center_y + t * dir_y
            if by1 <= y <= by2:
                t_values.append(t)
        
        # Right edge
        if dir_x > 0:  # Moving right
            t = (bx2 - bigger_center_x) / dir_x
            x = bigger_center_x + t * dir_x
            if by1 <= y <= by2:
                t_values.append(t)
        
        if t_values:
            t = min(t_values)
            boundary_point = (
                bigger_center_x + t * dir_x,
                bigger_center_y + t * dir_y
            )
            
            # Now create perpendicular at this boundary point
            intersections = [(boundary_point[0] + 100*perp_x, boundary_point[1] + 100*perp_y),
                            (boundary_point[0] - 100*perp_x, boundary_point[1] - 100*perp_y)]
    
    # Convert to int and return
    p1 = (int(intersections[0][0]), int(intersections[0][1]))
    p2 = (int(intersections[1][0]), int(intersections[1][1]))
    
    return np.array([p1, p2], dtype=np.int32)

def group_by_vertical_overlap(
    predictions: List[Tuple[np.ndarray, str, float]], 
    y_tolerance_ratio: float = 0.5
) -> List[List[Tuple[np.ndarray, str, float]]]:
    """
    Group boxes that overlap vertically.
    
    Args:
        predictions: List of tuples (box, class_name, confidence)
        y_tolerance_ratio: Ratio of box height used to determine vertical overlap
            
    Returns:
        List of groups, where each group is a list of predictions
    """
    if not predictions:
        return []
    
    groups = []
    remaining = predictions.copy()
    
    while remaining:
        current = remaining.pop(0)
        current_box = current[0]
        _, y1, _, y2 = current_box
        current_height = y2 - y1
        
        # Start a new group with the current box
        group = [current]
        group_y1 = y1
        group_y2 = y2
        
        # Find all boxes that overlap with the current group in the Y direction
        i = 0
        while i < len(remaining):
            other = remaining[i]
            other_box = other[0]
            _, other_y1, _, other_y2 = other_box
            
            # Check if boxes overlap in Y-coordinate range
            if ((group_y1 <= other_y1 <= group_y2) or
                (group_y1 <= other_y2 <= group_y2) or
                (other_y1 <= group_y1 and other_y2 >= group_y2) or 
                (other_y1 >= group_y1 and other_y2 <= group_y2)):
                
                # Add to the group and expand the group's y-range
                group.append(other)
                group_y1 = min(group_y1, other_y1)
                group_y2 = max(group_y2, other_y2)
                
                # Remove from remaining predictions
                remaining.pop(i)
            else:
                i += 1
        
        groups.append(group)
    
    return groups

def create_vertical_subgroups(group: List[Tuple[np.ndarray, str, float]], height_ratio: float = 0.5) -> List[List[Tuple[np.ndarray, str, float]]]:
    """
    Create subgroups for boxes that are very close vertically.
    """
    if not group:
        return []
    
    # First, sort the group by y1 (top edge)
    sorted_by_y = sorted(group, key=lambda p: p[0][1])
    
    subgroups = []
    current_subgroup = [sorted_by_y[0]]
    current_y1 = sorted_by_y[0][0][1]
    current_height = sorted_by_y[0][0][3] - current_y1
    y_threshold = current_y1 + current_height * height_ratio
    
    for item in sorted_by_y[1:]:
        box = item[0]
        y1 = box[1]
        
        # If this box's top edge is within the threshold, add to current subgroup
        if y1 <= y_threshold:
            current_subgroup.append(item)
            # Recalculate threshold based on the lowest box in the subgroup
            new_y1 = min([b[0][1] for b in current_subgroup])
            new_height = max([b[0][3] - b[0][1] for b in current_subgroup])
            y_threshold = new_y1 + new_height * height_ratio
        else:
            # Start a new subgroup
            subgroups.append(current_subgroup)
            current_subgroup = [item]
            current_y1 = y1
            current_height = box[3] - y1
            y_threshold = current_y1 + current_height * height_ratio
    
    # Add the last subgroup
    if current_subgroup:
        subgroups.append(current_subgroup)
    
    return subgroups

def rank_boxes(
    predictions: List[Tuple[np.ndarray, str, float]], 
    y_tolerance_ratio: float = 0.5
) -> Tuple[List[Tuple[np.ndarray, str, float]], List[List[np.ndarray]]]:
    """
    Rank bounding boxes based on reading order and identify overlapping boxes.
    
    This function implements the algorithm described in section 2.4.1 of the thesis,
    which handles the spatial relationships between text blocks in comics.
    
    Args:
        predictions: List of tuples (box, class_name, confidence)
        y_tolerance_ratio: Ratio of box height used to determine vertical overlap
            
    Returns:
        Tuple containing:
        - Sorted list of (box, class_name, confidence) tuples in reading order
        - List of lists of separator lines, one list per box (empty list if no separators needed)
    """
    if not predictions:
        return [], []
    
    # Sort the boxes first by Y-coordinate (y1) as primary sort key
    predictions = sorted(predictions, key=lambda p: p[0][1])
    
    # Group boxes by vertical overlap
    groups = group_by_vertical_overlap(predictions, y_tolerance_ratio)
    
    # Process each group to handle horizontal ordering and identify separators
    ranked_preds = []
    separator_lines_lists = []
    
    for group in groups:
        # Create subgroups for boxes that are very close vertically
        subgroups = create_vertical_subgroups(group)
        
        for subgroup in subgroups:
            # Sort each subgroup left-to-right for reading order
            subgroup.sort(key=lambda p: p[0][0])  # Sort by x1
            
            for box, class_name, confidence in subgroup:
                ranked_preds.append((box, class_name, confidence))
    
    # Now that the predictions are ranked, identify separators for overlapping boxes
    for i, (box, _, _) in enumerate(ranked_preds):
        box_separator_lines = []
        
        # Check for overlaps with all other boxes
        for j, (other_box, _, _) in enumerate(ranked_preds):
            if i == j:  # Skip self-comparison
                continue
            
            # Check if boxes overlap
            if boxes_overlap(box, other_box):
                separator_line = create_separator_line(box, other_box)
                if separator_line is not None:
                    box_separator_lines.append(separator_line)
        
        separator_lines_lists.append(box_separator_lines)
    
    return ranked_preds, separator_lines_lists

def process_text_blocks(
    blocks: List[TextBlock], 
    y_tolerance_ratio: float = 0.5
) -> Tuple[List[TextBlock], List[List[np.ndarray]]]:
    """
    Process a list of text blocks, sorting them and computing separator masks.
    
    Args:
        blocks: List of TextBlock objects to process
        y_tolerance_ratio: Ratio of box height used to determine vertical overlap
            
    Returns:
        Tuple containing:
        - Sorted list of TextBlock objects
        - List of lists of separator lines, one list per block (empty list if no separators needed)
    """
    if not blocks:
        return [], []
    
    # Convert TextBlocks to a format suitable for processing
    predictions = [(block.box, block.class_name, block.confidence) for block in blocks]
    
    # Sort and generate separator masks
    ranked_preds, separator_lines_lists = rank_boxes(predictions, y_tolerance_ratio)
    
    # Create new TextBlock objects with updated order
    sorted_blocks = []
    for i, (box, class_name, confidence) in enumerate(ranked_preds):
        sorted_blocks.append(TextBlock(
            box=box,
            class_name=class_name,
            confidence=confidence,
            id=i
        ))
    
    return sorted_blocks, separator_lines_lists

def apply_separator_masks(image: np.ndarray, separator_lines: List[np.ndarray]) -> np.ndarray:
    """
    Apply multiple separator masks to an image to resolve overlapping text blocks.
    
    Args:
        image: Input image (cropped text block)
        separator_lines: List of separator lines, each defined by two points
            
    Returns:
        Image with overlapping regions masked out
    """
    if not separator_lines:
        return image
    
    result = image.copy()
    for separator_line in separator_lines:
        result = apply_separator_mask(result, separator_line)
    
    return result

def apply_separator_mask(image: np.ndarray, separator_line: np.ndarray) -> np.ndarray:
    """
    Apply a single separator mask to an image to resolve overlapping text blocks.
    
    Args:
        image: Input image (cropped text block)
        separator_line: Two points defining the separator line
            
    Returns:
        Image with one side of the separator masked out
    """
    if separator_line is None:
        return image
    
    # Get the separator line points
    (x1, y1), (x2, y2) = separator_line.reshape(2, 2)
    h, w = image.shape[:2]
    
    # Compute line equation coefficients: Ax + By + C = 0
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    
    # Find the center of the image
    cx, cy = w // 2, h // 2
    
    # Determine which side to keep based on where the center is relative to the line
    center_side = A * cx + B * cy + C
    
    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Create grid of points covering the image
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    # Calculate which side of the line each point is on
    point_sides = A * x_coords + B * y_coords + C
    
    # Set mask values based on which side of the line to keep
    if center_side > 0:
        mask[point_sides > 0] = 255
    else:
        mask[point_sides <= 0] = 255
    
    # Apply the mask to the image
    return cv2.bitwise_and(image, image, mask=mask)


if __name__ == "__main__":
    import os
    import logging
    import matplotlib.pyplot as plt

    from dotenv import load_dotenv
    load_dotenv()
    from file_utils import load_image
    from visualization import (
        draw_boxes_with_labels,
        visualize_separator_lines
    )

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    input_image = os.path.abspath(os.getenv("INPUT_IMAGE", "./data/frames/test.jpg"))   
    logger = logging.getLogger(__name__)

    logger.info(f"Loading image from {input_image}")
    image = load_image(input_image)
    if image is None:
        logger.error(f"Failed to load image from {input_image}")
    
    # Create sample boxes based on YOLO prediction results
    boxes = [
        np.array([55.687, 700.25, 296.83, 963.28]),
        np.array([360.8, 977.91, 606.4, 1279.2]), 
        np.array([434.81, 192.65, 626.01, 386.56]),
        np.array([133.28, 6.4681, 399.34, 259.26]),
        np.array([152.38, 900.05, 397.52, 1085.6])
    ]
    
    classes = ["speech_bubble"] * 5
    confidences = [0.9721983, 0.9720702, 0.97205555, 0.971609, 0.9682035]
    
    # Create TextBlock objects
    text_blocks = []
    for i, (box, class_name, confidence) in enumerate(zip(boxes, classes, confidences)):
        text_blocks.append(TextBlock(
            box=box,
            class_name=class_name,
            confidence=confidence,
            id=i
        ))
    
    # Sort the blocks and identify separators
    logger.info("Processing spatial relationships between text blocks")
    sorted_blocks, separator_lines_lists = process_text_blocks(text_blocks)
    
    # Calculate how many blocks have separator lines
    blocks_with_separators = sum(1 for lines in separator_lines_lists if lines)
    logger.info(f"Found {len(sorted_blocks)} text blocks, {blocks_with_separators} with separator lines")
    
    # Create a figure to visualize everything
    plt.figure(figsize=(15, 10))
    
    # Plot original image with boxes and separator lines
    plt.subplot(1, 2, 1)
    vis_image = image.copy()
    
    # Draw bounding boxes
    vis_boxes = [block.box for block in sorted_blocks]
    vis_classes = [block.class_name for block in sorted_blocks]
    vis_confidences = [block.confidence for block in sorted_blocks]
    vis_ids = [block.id for block in sorted_blocks]
    
    labels = [
        f"{vis_ids[i]}: {vis_classes[i]} ({vis_confidences[i]:.2f})"
        for i in range(len(vis_boxes))
    ]

    vis_image = draw_boxes_with_labels(vis_image, vis_boxes, labels)
    vis_image = visualize_separator_lines(vis_image, sorted_blocks, separator_lines_lists)
    
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Text Block Detection with Separator Lines')
    
    # Plot grid of cropped blocks with masks applied
    plt.subplot(1, 2, 2)
    
    # Create a grid to display cropped blocks
    num_blocks = len(sorted_blocks)
    grid_cols = 3
    grid_rows = (num_blocks + grid_cols - 1) // grid_cols  # Ceiling division
    
    # Create a blank canvas for the grid
    grid_height = image.shape[0] 
    grid_width = image.shape[1]
    grid_image = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    
    # Calculate cell dimensions
    cell_width = grid_width // grid_cols
    cell_height = grid_height // grid_rows
    
    # Place each block in the grid
    for i, block in enumerate(sorted_blocks):
        row = i // grid_cols
        col = i % grid_cols
        
        # Crop the original block
        x1, y1, x2, y2 = map(int, block.box)
        cropped_block = image[y1:y2, x1:x2].copy()
        
        # Apply masks if there are separator lines
        if separator_lines_lists[i]:
            # Create a copy of separator lines adjusted to block's local coordinates
            local_separator_lines = []
            for line in separator_lines_lists[i]:
                # Convert line points from global to local coordinates
                local_p1 = np.array([line[0][0] - x1, line[0][1] - y1], dtype=np.int32)
                local_p2 = np.array([line[1][0] - x1, line[1][1] - y1], dtype=np.int32)
                local_line = np.array([local_p1, local_p2])
                local_separator_lines.append(local_line)
            
            # Apply masks using the local coordinates
            masked_block = apply_separator_masks(cropped_block, local_separator_lines)
            display_block = masked_block
            title = f"Block {block.id} (masked)"
        else:
            display_block = cropped_block
            title = f"Block {block.id}"
        
        # Resize the block to fit in the cell (maintaining aspect ratio)
        block_h, block_w = display_block.shape[:2]
        scale = min(cell_height * 0.8 / block_h, cell_width * 0.8 / block_w)
        resized_block = cv2.resize(display_block, None, fx=scale, fy=scale)
        
        # Calculate position to center the block in the cell
        pos_y = row * cell_height + (cell_height - resized_block.shape[0]) // 2
        pos_x = col * cell_width + (cell_width - resized_block.shape[1]) // 2
        
        # Place the block in the grid
        grid_image[pos_y:pos_y+resized_block.shape[0], pos_x:pos_x+resized_block.shape[1]] = resized_block
        
        # Add a title for each block
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_pos = (pos_x, pos_y - 10)
        cv2.putText(grid_image, title, title_pos, font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    plt.imshow(cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Cropped Text Blocks (with Masks Applied)')
    
    plt.tight_layout()
    plt.show()