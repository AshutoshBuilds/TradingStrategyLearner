import os
import sys
import json
from datetime import datetime

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import TradingStrategyBot

def main():
    """Demonstrate basic usage of the Trading Strategy Bot."""
    
    # Example configuration
    config = {
        "video_processor": {
            "frame_rate": 2,  # Process 2 frames per second
            "resize_width": 1920,
            "resize_height": 1080
        },
        "output": {
            "save_intermediates": True,
            "output_dir": "example_output"
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "example_bot.log"
        }
    }
    
    try:
        # Save configuration
        os.makedirs("example_output", exist_ok=True)
        with open("example_output/config.json", "w") as f:
            json.dump(config, f, indent=4)
            
        # Initialize bot with configuration
        bot = TradingStrategyBot(config_path="example_output/config.json")
        
        # Example video URL (replace with your video path)
        video_path = "path/to/your/trading_video.mp4"
        
        print(f"\nProcessing video: {video_path}")
        print("This may take a few minutes depending on the video length...\n")
        
        # Process video
        results = bot.process_video(video_path)
        
        # Print summary of results
        print("\nProcessing completed successfully!")
        print(f"Results saved to: {results['output_dir']}\n")
        
        print("Generated Strategy Summary:")
        print("-" * 50)
        
        strategy = results['strategy']
        print(f"Strategy Type: {strategy['template_name']}")
        print(f"Description: {strategy['description']}")
        print("\nParameters:")
        for param, value in strategy['parameters'].items():
            print(f"  {param}: {value}")
            
        print("\nTrading Conditions:")
        for condition in strategy['conditions']:
            print(f"  - {condition['type']}: {condition['indicator']} {condition['operator']} "
                  f"{condition.get('threshold', condition.get('value', ''))}")
            
        print("\nStrategy Evaluation:")
        for metric, value in strategy['evaluation'].items():
            print(f"  {metric}: {value:.2f}")
            
        print(f"\nOverall Confidence Score: {strategy['confidence_score']:.2%}")
        
        # Print locations of generated visualizations
        print("\nGenerated Visualizations:")
        print("-" * 50)
        for viz_file in os.listdir(results['output_dir']):
            if viz_file.endswith('.png'):
                print(f"- {viz_file}")
                
    except FileNotFoundError:
        print(f"Error: Video file not found: {video_path}")
        print("Please update the video_path variable with the path to your trading video.")
        return 1
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
        
    return 0

def create_example_config():
    """Create an example configuration file."""
    config = {
        "video_processor": {
            "frame_rate": 2,
            "resize_width": 1920,
            "resize_height": 1080
        },
        "audio_processor": {
            "sample_rate": 16000,
            "chunk_duration": 30
        },
        "text_detector": {
            "min_confidence": 0.5
        },
        "knowledge_extractor": {
            "model_path": "en_core_web_sm",
            "sentiment_model": "distilbert-base-uncased-finetuned-sst-2-english",
            "embedding_model": "all-MiniLM-L6-v2"
        },
        "strategy_generator": {
            "llm_model": "gpt2",
            "embedding_model": "all-MiniLM-L6-v2"
        },
        "visualizer": {
            "style": "darkgrid",
            "figsize": [12, 8]
        },
        "output": {
            "save_intermediates": True,
            "output_dir": "example_output"
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "example_bot.log"
        }
    }
    
    os.makedirs("example_output", exist_ok=True)
    with open("example_output/example_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    print("Example configuration file created: example_output/example_config.json")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--create-config":
        create_example_config()
    else:
        exit(main()) 