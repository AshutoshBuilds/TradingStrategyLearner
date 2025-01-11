import os
import numpy as np
import torch
import whisper
import moviepy.editor as mp
from transformers import pipeline
from typing import List, Dict, Any, Optional, Tuple

class AudioProcessor:
    def __init__(self, use_cuda: bool = True):
        """Initialize the audio processor.
        
        Args:
            use_cuda (bool): Whether to use CUDA for processing
        """
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.initialize_models()

    def initialize_models(self):
        """Initialize required models for audio processing."""
        # Initialize Whisper model
        self.whisper_model = whisper.load_model("base", device=self.device)
        
        # Initialize emotion classifier
        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=0 if self.device == "cuda" else -1
        )

    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """Extract audio from video file.
        
        Args:
            video_path (str): Path to video file
            output_path (Optional[str]): Path to save audio file
            
        Returns:
            str: Path to extracted audio file
        """
        if output_path is None:
            output_path = os.path.splitext(video_path)[0] + ".wav"
            
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(output_path, verbose=False, logger=None)
        video.close()
        
        return output_path

    def _process_audio_segment(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """Process an audio segment.
        
        Args:
            segment (Dict[str, Any]): Audio segment data
            
        Returns:
            Dict[str, Any]: Processed segment data
        """
        # Get emotion from text
        emotion_result = self.emotion_classifier(segment['text'])[0]
        emotion = emotion_result['label']
        confidence = emotion_result['score']
        
        return {
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text'],
            'emotion': emotion,
            'confidence': confidence
        }

    def process_audio(self, audio_path: str, start_time: Optional[float] = None,
                     end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """Process an audio file.
        
        Args:
            audio_path (str): Path to audio file
            start_time (Optional[float]): Start time in seconds
            end_time (Optional[float]): End time in seconds
            
        Returns:
            List[Dict[str, Any]]: Processed audio segments
        """
        # Transcribe audio
        result = self.whisper_model.transcribe(
            audio_path,
            start=start_time,
            end=end_time,
            language="en"
        )
        
        # Process each segment
        processed_segments = []
        for segment in result['segments']:
            processed_segment = self._process_audio_segment(segment)
            processed_segments.append(processed_segment)
            
        return processed_segments

    def process_video_audio(self, video_path: str, output_dir: str,
                          start_time: Optional[float] = None,
                          end_time: Optional[float] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """Process audio from a video file.
        
        Args:
            video_path (str): Path to video file
            output_dir (str): Directory to save extracted audio
            start_time (Optional[float]): Start time in seconds
            end_time (Optional[float]): End time in seconds
            
        Returns:
            Tuple[str, List[Dict[str, Any]]]: Path to audio file and processed segments
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output path
        audio_path = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0] + ".wav")
        
        # Extract audio if it doesn't exist
        if not os.path.exists(audio_path):
            self.extract_audio(video_path, audio_path)
            
        # Process audio
        results = self.process_audio(audio_path, start_time, end_time)
        
        return audio_path, results 