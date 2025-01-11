import unittest
import os
import numpy as np
from datetime import datetime
from src.main import TradingStrategyBot
from src.content_processor.video_processor import VideoProcessor
from src.content_processor.audio_processor import AudioProcessor
from src.content_processor.text_detector import TextDetector
from src.content_processor.knowledge_extractor import KnowledgeExtractor
from src.strategy.strategy_generator import StrategyGenerator
from src.visualization.visualizer import Visualizer

class TestTradingStrategyBot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_data_dir = os.path.join(cls.test_dir, 'test_data')
        os.makedirs(cls.test_data_dir, exist_ok=True)
        
        # Create test video file
        cls.test_video_path = os.path.join(cls.test_data_dir, 'test_video.mp4')
        cls._create_test_video()
        
        # Initialize bot
        cls.bot = TradingStrategyBot()

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove test data
        if os.path.exists(cls.test_data_dir):
            import shutil
            shutil.rmtree(cls.test_data_dir)

    @classmethod
    def _create_test_video(cls):
        """Create a test video file."""
        # TODO: Implement test video creation
        # For now, we'll just create a dummy file
        with open(cls.test_video_path, 'wb') as f:
            f.write(b'dummy video content')

    def test_video_processor(self):
        """Test video processing functionality."""
        processor = VideoProcessor()
        
        # Create test frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[300:400, 500:600] = 255  # Add white rectangle
        
        # Process frame
        processed = processor._process_frame(frame)
        
        self.assertEqual(processed.shape, (720, 1280, 3))
        self.assertTrue(np.any(processed > 0))

    def test_audio_processor(self):
        """Test audio processing functionality."""
        processor = AudioProcessor()
        
        # Create test audio data
        audio_data = np.random.rand(16000)  # 1 second of random audio
        
        # Process audio
        chunks = processor._process_audio_data(audio_data, 16000)
        
        self.assertTrue(len(chunks) > 0)
        self.assertTrue(isinstance(chunks[0], np.ndarray))

    def test_text_detector(self):
        """Test text detection functionality."""
        detector = TextDetector()
        
        # Create test image with text
        image = np.zeros((200, 400), dtype=np.uint8)
        # TODO: Add actual text to image for OCR testing
        
        # Detect text
        results = detector.detect_text(image)
        
        self.assertTrue(isinstance(results, list))

    def test_knowledge_extractor(self):
        """Test knowledge extraction functionality."""
        extractor = KnowledgeExtractor()
        
        # Test text
        text = "The market shows a bullish trend with increasing volume. AAPL stock rose 5% today."
        
        # Extract entities
        entities = extractor.extract_entities(text)
        self.assertTrue(len(entities) > 0)
        
        # Extract sentiment
        sentiment = extractor.analyze_sentiment(text)
        self.assertTrue('label' in sentiment)
        self.assertTrue('score' in sentiment)

    def test_strategy_generator(self):
        """Test strategy generation functionality."""
        generator = StrategyGenerator()
        
        # Test data
        market_data = {
            'dates': [datetime.now()],
            'prices': [100.0],
            'volumes': [1000000],
            'volatility': 0.02,
            'trend_strength': 0.01
        }
        
        extracted_knowledge = {
            'sentiment': {'label': 'POSITIVE', 'score': 0.8},
            'topics': [{'main_term': 'bullish', 'frequency': 5}]
        }
        
        # Generate strategy
        strategy = generator.generate_strategy(market_data, extracted_knowledge)
        
        self.assertTrue('template_name' in strategy)
        self.assertTrue('parameters' in strategy)
        self.assertTrue('conditions' in strategy)
        self.assertTrue('evaluation' in strategy)

    def test_visualizer(self):
        """Test visualization functionality."""
        visualizer = Visualizer()
        
        # Test data
        market_data = {
            'dates': [datetime.now()],
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [102.0],
            'volume': [1000000]
        }
        
        # Test price analysis plot
        try:
            visualizer.plot_price_analysis(market_data)
        except Exception as e:
            self.fail(f"Visualization failed: {str(e)}")

    def test_end_to_end(self):
        """Test end-to-end functionality."""
        try:
            # Process test video
            results = self.bot.process_video(self.test_video_path)
            
            # Check results
            self.assertTrue('extracted_knowledge' in results)
            self.assertTrue('market_data' in results)
            self.assertTrue('strategy' in results)
            self.assertTrue('output_dir' in results)
            
            # Check output files
            output_dir = results['output_dir']
            self.assertTrue(os.path.exists(output_dir))
            self.assertTrue(os.path.exists(os.path.join(output_dir, 'results.json')))
            
        except Exception as e:
            self.fail(f"End-to-end test failed: {str(e)}")

if __name__ == '__main__':
    unittest.main() 