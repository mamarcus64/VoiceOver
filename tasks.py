"""
Annotation task definitions and base classes
"""
from pathlib import Path
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
import numpy as np
from abc import ABC, abstractmethod


# Label helper classes
class ChooseOne:
    """Radio button style selection - only one choice allowed"""
    def __init__(self, label, choices, required=False):
        self.label = label
        self.choices = choices
        self.required = required
        self.__class__.__name__ = 'ChooseOne'  # For template identification


class ChooseMany:
    """Checkbox style selection - multiple choices allowed"""
    def __init__(self, label, choices, required=False):
        self.label = label
        self.choices = choices
        self.required = required
        self.__class__.__name__ = 'ChooseMany'  # For template identification


class FreeText:
    """Free text input field"""
    def __init__(self, label, placeholder='', required=False):
        self.label = label
        self.placeholder = placeholder
        self.required = required
        self.__class__.__name__ = 'FreeText'  # For template identification


# Base annotation task class
class AnnotationTask(ABC):
    """
    Base class for all annotation tasks.
    Subclass this to create specific annotation tasks.
    """
    name: str = None  # Task name used in URLs
    stimuli: pd.DataFrame = None  # Must include 'stimulus_id' column
    options: list = None  # List of Label objects (ChooseOne, ChooseMany, FreeText)
    
    @abstractmethod
    def render_stimuli(self, row: pd.Series) -> list:
        """
        Return list of renderables (images, videos, or text) for the given stimulus.
        
        Returns:
            list containing:
            - image: np.ndarray (H, W, 3) in BGR format
            - video: np.ndarray (T, H, W, 3) in BGR format  
            - text: str
        """
        raise NotImplementedError


# Concrete implementation for count_subjects task
class CountSubjects(AnnotationTask):
    """Count the number of subjects visible in video frames"""
    name = "count_subjects"
    
    def __init__(self):
        # Use actual production data folder
        folder = Path("/data2/mjma/voices/test_data/videos/")
        
        # Find all mp4 files
        files = sorted(folder.glob("*.mp4"))
        
        # Create stimuli dataframe
        self.stimuli = pd.DataFrame({
            "stimulus_id": [f"{i:05d}" for i in range(len(files))],
            "filepath": [str(f) for f in files],
        })
        
        # Define task options
        self.options = [
            ChooseOne("Number of Subjects", ["1", "2", "3", "Do not use (at least one frame is different)"], required=True)
        ]
    
    def render_stimuli(self, row: pd.Series) -> list:
        """
        Extract 10 frames from video at evenly spaced intervals
        """
        filepath = row["filepath"]
        
        try:
            # Load video using moviepy
            clip = VideoFileClip(filepath)
            duration = clip.duration
            
            frames = []
            # Extract frames at 10 evenly spaced intervals
            for i in range(10):
                ratio = (i + 1) / 11.0  # Avoid the very end of video
                timestamp = duration * ratio
                
                # Get frame at timestamp (returns RGB)
                frame = clip.get_frame(timestamp)
                # Convert RGB to BGR for consistency
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames.append(frame_bgr)
            
            # Clean up
            clip.close()
            
            return frames
            
        except Exception as e:
            print(f"Error processing video {filepath}: {e}")
            # Return placeholder frames on error
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Error loading video", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return [placeholder] * 10