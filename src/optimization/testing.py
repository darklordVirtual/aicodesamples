"""
Kapittel 12: Testing AI Systems
Framework for testing AI prompts and systems.
"""
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from anthropic import Anthropic
import time

try:
    from utils import config, logger, LoggerMixin
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import config, logger, LoggerMixin


@dataclass
class TestResult:
    """Result from a test case"""
    test_name: str
    passed: bool
    expected: Optional[str]
    actual: str
    score: Optional[float] = None
    latency: Optional[float] = None
    error: Optional[str] = None


class PromptTestSuite(LoggerMixin):
    """
    Test suite for prompt engineering.
    """
    
    def __init__(self, model: Optional[str] = None):
        self.model = model or config.ai.model
        self.client = Anthropic(api_key=config.ai.api_key)
        self.test_results: List[TestResult] = []
        self.log_info("Initialized prompt test suite")
    
    def test_prompt(
        self,
        test_name: str,
        prompt: str,
        validator: Callable[[str], bool],
        expected: Optional[str] = None
    ) -> TestResult:
        """
        Test a prompt with validation function.
        
        Args:
            test_name: Test identifier
            prompt: Prompt to test
            validator: Function that returns True if response is valid
            expected: Optional expected response
            
        Returns:
            Test result
        """
        self.log_info(f"Running test: {test_name}")
        
        start_time = time.time()
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            actual = response.content[0].text.strip()
            latency = time.time() - start_time
            
            passed = validator(actual)
            
            result = TestResult(
                test_name=test_name,
                passed=passed,
                expected=expected,
                actual=actual,
                latency=latency
            )
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                passed=False,
                expected=expected,
                actual="",
                error=str(e)
            )
        
        self.test_results.append(result)
        return result
    
    def test_variations(
        self,
        test_name: str,
        prompts: List[str],
        validator: Callable[[str], bool]
    ) -> Dict[str, TestResult]:
        """
        Test multiple prompt variations.
        
        Args:
            test_name: Base test name
            prompts: List of prompt variations
            validator: Validation function
            
        Returns:
            Dict of results by variation name
        """
        results = {}
        
        for i, prompt in enumerate(prompts, 1):
            variation_name = f"{test_name}_v{i}"
            result = self.test_prompt(variation_name, prompt, validator)
            results[variation_name] = result
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary statistics."""
        if not self.test_results:
            return {"message": "No tests run"}
        
        passed = sum(1 for r in self.test_results if r.passed)
        total = len(self.test_results)
        avg_latency = sum(r.latency or 0 for r in self.test_results) / total
        
        return {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total,
            "avg_latency": avg_latency
        }


class AITestFramework(LoggerMixin):
    """
    Complete framework for AI system testing.
    """
    
    def __init__(self):
        self.suites: Dict[str, PromptTestSuite] = {}
        self.log_info("Initialized AI test framework")
    
    def create_suite(self, name: str, model: Optional[str] = None) -> PromptTestSuite:
        """Create new test suite."""
        suite = PromptTestSuite(model=model)
        self.suites[name] = suite
        return suite
    
    def run_all_suites(self) -> Dict[str, Dict[str, Any]]:
        """Run all test suites and get results."""
        results = {}
        
        for name, suite in self.suites.items():
            results[name] = suite.get_summary()
        
        return results


# Example usage
def example_prompt_testing():
    """Example: Prompt testing"""
    suite = PromptTestSuite()
    
    # Test 1: Classification
    result1 = suite.test_prompt(
        test_name="sentiment_classification",
        prompt="Classify the sentiment of: 'This product is amazing!' Return only: positive, negative, or neutral.",
        validator=lambda r: r.lower() in ["positive", "negative", "neutral"],
        expected="positive"
    )
    
    print(f"Test 1: {result1.test_name}")
    print(f"  Passed: {result1.passed}")
    print(f"  Actual: {result1.actual}")
    print(f"  Latency: {result1.latency:.2f}s")
    
    # Test 2: JSON output
    result2 = suite.test_prompt(
        test_name="json_extraction",
        prompt="Extract name and age from: 'John is 30 years old.' Return as JSON.",
        validator=lambda r: "name" in r.lower() and "age" in r.lower()
    )
    
    print(f"\nTest 2: {result2.test_name}")
    print(f"  Passed: {result2.passed}")
    
    # Summary
    summary = suite.get_summary()
    print(f"\nSummary:")
    print(f"  Total: {summary['total_tests']}")
    print(f"  Pass rate: {summary['pass_rate']:.0%}")
    print(f"  Avg latency: {summary['avg_latency']:.2f}s")


def example_prompt_variations():
    """Example: Testing prompt variations"""
    suite = PromptTestSuite()
    
    variations = [
        "Summarize this in one sentence: [text]",
        "Provide a brief one-sentence summary: [text]",
        "TL;DR: [text]"
    ]
    
    results = suite.test_variations(
        test_name="summarization",
        prompts=[v.replace("[text]", "AI is transforming many industries.") for v in variations],
        validator=lambda r: len(r.split()) <= 20
    )
    
    print("Prompt variation results:")
    for name, result in results.items():
        print(f"  {name}: {'✓' if result.passed else '✗'} ({result.latency:.2f}s)")


if __name__ == "__main__":
    print("=== Prompt Testing ===")
    example_prompt_testing()
    
    print("\n=== Prompt Variations ===")
    example_prompt_variations()
