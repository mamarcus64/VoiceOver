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
    template: str = "annotate.html"  # Template to use for rendering
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
            ChooseOne("Number of Subjects", ["Use (same subject in all five frames)", "Do not use (at least one frame is different)"], required=True)
        ]
    
    def render_stimuli(self, row: pd.Series) -> list:
        """
        Extract 5 frames from video at 10%, 20%, 70%, 80%, 90%
        """
        filepath = row["filepath"]
        
        try:
            # Load video using moviepy
            clip = VideoFileClip(filepath)
            duration = clip.duration
            
            frames = []
            # Extract frames at specific percentages
            for ratio in [0.1, 0.2, 0.7, 0.8, 0.9]:
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
            return [placeholder] * 5


class EmotionComparison(AnnotationTask):
    """
    Emotion comparison task: show two 10-second clips side by side.
    Annotators judge which clip is happier and which is more energetic.
    """
    name = "emotion_comparison"
    template = "emotion_compare.html"
    
    def __init__(self):
        csv_path = Path(__file__).parent / "emotion_pairs.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(
                f"emotion_pairs.csv not found at {csv_path}. "
                "Run generate_emotion_pairs.py first."
            )
        
        self.stimuli = pd.read_csv(csv_path, dtype={'stimulus_id': str})
        
        # Ensure stimulus_id is zero-padded
        self.stimuli['stimulus_id'] = self.stimuli['stimulus_id'].apply(
            lambda x: f"{int(x):05d}"
        )
        
        # Video directory
        self.video_dir = "/data2/mjma/voices/test_data/videos/"
        
        # Define task options
        self.options = [
            ChooseOne("Which was happier?", ["Video A", "Video B"], required=True),
            ChooseOne("Which was more energetic?", ["Video A", "Video B"], required=True),
        ]
    
    def render_stimuli(self, row: pd.Series) -> list:
        """Return video metadata as a dict (not actual frames)."""
        return [{
            'type': 'video_comparison',
            'video_file': row['video_file'],
            'clip_a_start': float(row['clip_a_start']),
            'clip_a_end': float(row['clip_a_end']),
            'clip_b_start': float(row['clip_b_start']),
            'clip_b_end': float(row['clip_b_end']),
            'subject_id': row['subject_id'],
            'comparison_type': row['comparison_type'],
        }]