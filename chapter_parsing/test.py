# import MeCab
# import mecab_ko_dic
# tagger = MeCab.Tagger(mecab_ko_dic.MECAB_ARGS)
# print(tagger.parse("안녕하세요세계."))

import os
import cv2
import logging
import numpy as np
from typing import List, Tuple

from utils import crop_image, load_image, apply_separator_mask
#from prediction_model import rank_boxes

def rank_boxes(predictions: List[Tuple[np.ndarray, str, float]], 
               y_tolerance_ratio: float = 0.5) -> Tuple[List[Tuple[np.ndarray, str, float]], List[List[np.ndarray]]]:
    """
    Rank bounding boxes based on Y-coordinates and create separator lines for overlapping boxes.
    
    Args:
        predictions (List[Tuple[np.ndarray, str, float]]): List of predictions with boxes, class names, and confidence scores.
        y_tolerance_ratio (float): Tolerance as a ratio of the box height to consider boxes as overlapping.
        
    Returns:
        Tuple containing:
            - List[Tuple[np.ndarray, str, float]]: Ranked list of predictions.
            - List[List[np.ndarray]]: List of separator line lists for each box (empty list if no separators).
    """
    if not predictions:
        return [], []
    
    # Make a copy of predictions to avoid modifying the original
    preds = predictions.copy()
    
    # Sort the boxes first by Y-coordinate (y1) as a primary sort
    preds.sort(key=lambda p: p[0][1])
    
    # Group boxes that overlap vertically
    groups = []
    while preds:
        current = preds.pop(0)
        current_box = current[0]
        _, y1, _, y2 = current_box
        
        # Start a new group with the current box
        group = [current]
        group_y1 = y1
        group_y2 = y2
        
        # Find all boxes that overlap with the current group in the Y direction
        i = 0
        while i < len(preds):
            other = preds[i]
            other_box = other[0]
            _, other_y1, _, other_y2 = other_box
            
            # Check if boxes overlap in Y-coordinate range
            # A box overlaps if its top edge is within the group's y-range
            # or if its bottom edge is within the group's y-range
            # or if it completely contains the group's y-range
            if (group_y1 <= other_y1 <= group_y2 or 
                group_y1 <= other_y2 <= group_y2 or
                (other_y1 <= group_y1 and other_y2 >= group_y2)):
                
                # Add to the group and expand the group's y-range
                group.append(other)
                group_y1 = min(group_y1, other_y1)
                group_y2 = max(group_y2, other_y2)
                
                # Remove from remaining predictions
                preds.pop(i)
            else:
                i += 1
        
        # Sort the group by x1 (left edge)
        group.sort(key=lambda p: p[0][0])
        groups.append(group)
    
    # Process all groups to create ranked boxes and separator lines
    ranked_boxes = []
    separator_lines_list = []
    
    for group in groups:
        # For each box in the group
        for i, box_data in enumerate(group):
            box = box_data[0]
            x1, y1, x2, y2 = box
            
            # Find all separator lines for this box
            box_separator_lines = []
            
            # Check for overlaps with all other boxes in the same group
            for j, other_data in enumerate(group):
                if i == j:  # Skip self
                    continue
                    
                other_box = other_data[0]
                other_x1, other_y1, other_x2, other_y2 = other_box
                
                # Check for X-axis overlap
                if (
                        x1 <= other_x1 <= x2 or x1 <= other_x2 <= x2 or (other_x1 <= x1 and other_x2 >= x2)
                    ) and (
                        y1 <= other_y1 <= y2 or y1 <= other_y2 <= y2 or (other_y1 <= y1 and other_y2 >= y2)
                    ): #!!!!!!!!!!!!!!!!!!!!!! WORKS BUT NEED TO FIX THE INVARIANCE. UP AND DOWN DIAGONALS HAVE DIFFERENT FORMULAS
                    # WHAT ABOUT HORIZONTAL AND VERTICAL LINES? CHANGE ALGORYTHM


                    # Calculate the separator line points
                    # The separator should go from the max of the left edges to the min of the right edges
                    # And from the max of the top edges to the min of the bottom edges
                    sep_x1 = max(x1, other_x1)
                    sep_x2 = min(x2, other_x2)
                    sep_y1 = max(y1, other_y1)
                    sep_y2 = min(y2, other_y2)
                    
                    # Create separator line as two points: top-left to bottom-right of overlap area
                    separator_line = np.array([[sep_x1, sep_y2], [sep_x2, sep_y1]], dtype=np.int32)
                    box_separator_lines.append(separator_line)
            
            # Add box and its separator lines to results
            ranked_boxes.append(box_data)
            separator_lines_list.append(box_separator_lines)
    
    return ranked_boxes, separator_lines_list

image = load_image("E:/Projects/Python/Manhua_Scanner/OCR/chapter_parsing/test.png")

# 1000x1000
# x1, y1, x2, y2 = box1
boxes = [
    ([100, 100, 400, 400], 'speech_bubble', 0.9), # top left (big)
    ([600, 200, 800, 400], 'speech_bubble', 0.9), # top right (small)
    ([300, 300, 700, 700], 'speech_bubble', 0.9), # middle (huge)
    ([600, 600, 900, 900], 'speech_bubble', 0.9), # bottom right (big)
    ([200, 600, 400, 800], 'speech_bubble', 0.9) # bottom left (small)
]

print(rank_boxes(boxes))

# [
#     ([100, 100, 400, 400], 'speech_bubble', 0.9), 
#     ([600, 200, 800, 400], 'speech_bubble', 0.9), 
#     ([300, 300, 700, 700], 'speech_bubble', 0.9), 
#     ([600, 600, 900, 900], 'speech_bubble', 0.9), 
#     ([200, 600, 400, 800], 'speech_bubble', 0.9)]
# (
#     [
#         ([100, 100, 400, 400], 'speech_bubble', 0.9), 
#         ([200, 600, 400, 800], 'speech_bubble', 0.9), 
#         ([300, 300, 700, 700], 'speech_bubble', 0.9), 
#         ([600, 200, 800, 400], 'speech_bubble', 0.9), 
#         ([600, 600, 900, 900], 'speech_bubble', 0.9)], 

#         [None, None, None, None, None, 
#             array([[200, 100],[400, 800]]), 
#             array([[300, 200],[400, 700]]), 
#             array([[600, 200],[700, 700]]), 
#             array([[600, 600],[800, 900]]), 
#             array([[600, 200],[800, 900]])
#         ]
# )

# [   [
#         array([[200, 400], [400, 600]]), 
#         array([[300, 400],[400, 300]])
#     ], 
#     [
#         array([[300, 700],[400, 600]])
#     ], 
#     [
#         array([[600, 400],[700, 300]]),
#         array([[600, 700],[700, 600]])
#     ], 
#         [array([[600, 400],[800, 600]])
#     ], 
#         [array([[600, 400],[800, 600]])
#     ]
# ]
# 300 400, 400 300
