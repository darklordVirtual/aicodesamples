"""
Kapittel 15: Multimodal AI
Image and document analysis using AI vision capabilities.
"""
from typing import Dict, Any, Optional, List
import base64
from pathlib import Path

try:
    from utils import config, logger, LoggerMixin
    from anthropic import Anthropic
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import config, logger, LoggerMixin
    from anthropic import Anthropic


class ImageAnalyzer(LoggerMixin):
    """
    Analyze images using Claude's vision capabilities.
    """
    
    def __init__(self):
        self.client = Anthropic(api_key=config.ai.api_key)
        self.log_info("Initialized image analyzer")
    
    def analyze_image(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail."
    ) -> str:
        """
        Analyze image with AI.
        
        Args:
            image_path: Path to image file
            prompt: Analysis prompt
            
        Returns:
            Analysis result
        """
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = base64.standard_b64encode(f.read()).decode('utf-8')
        
        # Detect media type
        suffix = Path(image_path).suffix.lower()
        media_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }.get(suffix, 'image/jpeg')
        
        # Call API with vision
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )
        
        result = response.content[0].text
        self.log_info(f"Analyzed image: {image_path}")
        return result


class DocumentAnalyzer(LoggerMixin):
    """
    Analyze documents with text and images.
    """
    
    def __init__(self):
        self.client = Anthropic(api_key=config.ai.api_key)
        self.image_analyzer = ImageAnalyzer()
        self.log_info("Initialized document analyzer")
    
    def extract_document_data(
        self,
        document_text: str,
        document_images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract structured data from document with text and images.
        
        Args:
            document_text: Text content
            document_images: Optional list of image paths
            
        Returns:
            Extracted data
        """
        analysis_results = {"text_analysis": document_text}
        
        # Analyze images if provided
        if document_images:
            image_analyses = []
            for img_path in document_images:
                result = self.image_analyzer.analyze_image(
                    img_path,
                    prompt="Extract all text and describe visual elements."
                )
                image_analyses.append(result)
            
            analysis_results["image_analyses"] = image_analyses
        
        # Combine all information
        combined_prompt = f"Extract key information from this document:\n\n{document_text}"
        
        if document_images:
            combined_prompt += f"\n\nImage content:\n" + "\n".join(
                f"Image {i+1}: {img}" for i, img in enumerate(analysis_results.get("image_analyses", []))
            )
        
        response = self.client.messages.create(
            model=config.ai.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": combined_prompt}]
        )
        
        analysis_results["structured_data"] = response.content[0].text
        
        return analysis_results


# Example usage
def example_image_analysis():
    """Example: Image analysis (requires actual image file)"""
    print("Image analysis requires actual image files.")
    print("Usage:")
    print("  analyzer = ImageAnalyzer()")
    print("  result = analyzer.analyze_image('path/to/image.jpg', 'Describe this image')")


def example_document_analysis():
    """Example: Document analysis"""
    analyzer = DocumentAnalyzer()
    
    doc_text = """
    KONTRAKT
    
    Mellom: Leverandør AS og Kunde AS
    Dato: 2025-01-20
    
    1. Leveranse: Konsulentjenester
    2. Varighet: 6 måneder
    3. Pris: 100,000 NOK per måned
    4. Betalingstermin: 30 dager
    """
    
    result = analyzer.extract_document_data(doc_text)
    print("Document analysis:")
    print(result["structured_data"][:200] + "...")


if __name__ == "__main__":
    print("=== Multimodal AI ===")
    example_image_analysis()
    print()
    example_document_analysis()
