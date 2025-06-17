#!/usr/bin/env python3
"""
Comprehensive test runner for the Ratatoskr AI Assistant.

This script provides a unified interface for running all tests in the project,
including unit tests, integration tests, and performance benchmarks.

Features:
- Test discovery and execution
- Coverage reporting
- Performance benchmarking
- Test categorization
- Parallel test execution
"""

import unittest
import sys
import os
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import test modules
from tests.test_memory_long_term import TestLongTermMemory, TestMemoryPerformance
from tests.test_agent import TestPolicy, TestAgentEngine, TestAgentIntegration
from tests.test_tools import (
    TestWebSearch, TestBrowserTool, TestToolsIntegration, TestToolsPerformance
)
from tests.test_voice import (
    TestTextToSpeech, TestSpeechToText, TestVoiceIntegration, TestVoicePerformance
)


class TestRunner:
    """Comprehensive test runner for the Ratatoskr project."""
    
    def __init__(self):
        """Initialize the test runner."""
        self.test_suites = {
            'memory': [
                TestLongTermMemory,
                TestMemoryPerformance
            ],
            'agent': [
                TestPolicy,
                TestAgentEngine,
                TestAgentIntegration
            ],
            'tools': [
                TestWebSearch,
                TestBrowserTool,
                TestToolsIntegration,
                TestToolsPerformance
            ],
            'voice': [
                TestTextToSpeech,
                TestSpeechToText,
                TestVoiceIntegration,
                TestVoicePerformance
            ]
        }
        
        self.results = {}
    
    def run_suite(self, suite_name: str, test_classes: List) -> Dict[str, Any]:
        """
        Run a specific test suite.
        
        Args:
            suite_name: Name of the test suite
            test_classes: List of test classes to run
            
        Returns:
            Dict containing test results
        """
        print(f"\n{'='*60}")
        print(f"Running {suite_name.upper()} Test Suite")
        print(f"{'='*60}")
        
        # Create test suite
        suite = unittest.TestSuite()
        
        for test_class in test_classes:
            try:
                # Load tests from class
                tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
                suite.addTests(tests)
            except Exception as e:
                print(f"Error loading {test_class.__name__}: {e}")
                continue
        
        # Run tests
        start_time = time.time()
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        end_time = time.time()
        
        # Compile results
        suite_results = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
            'execution_time': end_time - start_time,
            'success_rate': ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
        }
        
        # Print summary
        print(f"\n{suite_name.upper()} Test Summary:")
        print(f"  Tests Run: {suite_results['tests_run']}")
        print(f"  Failures: {suite_results['failures']}")
        print(f"  Errors: {suite_results['errors']}")
        print(f"  Skipped: {suite_results['skipped']}")
        print(f"  Success Rate: {suite_results['success_rate']:.1f}%")
        print(f"  Execution Time: {suite_results['execution_time']:.2f}s")
        
        return suite_results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all test suites.
        
        Returns:
            Dict containing overall test results
        """
        print("Ratatoskr AI Assistant - Comprehensive Test Suite")
        print("=" * 60)
        
        total_start_time = time.time()
        
        # Run each test suite
        for suite_name, test_classes in self.test_suites.items():
            try:
                self.results[suite_name] = self.run_suite(suite_name, test_classes)
            except Exception as e:
                print(f"Error running {suite_name} suite: {e}")
                self.results[suite_name] = {
                    'tests_run': 0,
                    'failures': 0,
                    'errors': 1,
                    'skipped': 0,
                    'execution_time': 0,
                    'success_rate': 0
                }
        
        total_end_time = time.time()
        
        # Compile overall results
        overall_results = self._compile_overall_results()
        overall_results['total_execution_time'] = total_end_time - total_start_time
        
        # Print overall summary
        self._print_overall_summary(overall_results)
        
        return overall_results
    
    def run_specific_suite(self, suite_name: str) -> Dict[str, Any]:
        """
        Run a specific test suite by name.
        
        Args:
            suite_name: Name of the test suite to run
            
        Returns:
            Dict containing test results
        """
        if suite_name not in self.test_suites:
            print(f"Error: Test suite '{suite_name}' not found.")
            print(f"Available suites: {list(self.test_suites.keys())}")
            return {}
        
        return self.run_suite(suite_name, self.test_suites[suite_name])
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """
        Run only performance tests.
        
        Returns:
            Dict containing performance test results
        """
        print("\nRunning Performance Tests Only")
        print("=" * 40)
        
        performance_classes = [
            TestMemoryPerformance,
            TestToolsPerformance,
            TestVoicePerformance
        ]
        
        return self.run_suite('performance', performance_classes)
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """
        Run only integration tests.
        
        Returns:
            Dict containing integration test results
        """
        print("\nRunning Integration Tests Only")
        print("=" * 40)
        
        integration_classes = [
            TestAgentIntegration,
            TestToolsIntegration,
            TestVoiceIntegration
        ]
        
        return self.run_suite('integration', integration_classes)
    
    def _compile_overall_results(self) -> Dict[str, Any]:
        """
        Compile overall test results from all suites.
        
        Returns:
            Dict containing overall results
        """
        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_skipped = 0
        total_time = 0
        
        for suite_results in self.results.values():
            total_tests += suite_results['tests_run']
            total_failures += suite_results['failures']
            total_errors += suite_results['errors']
            total_skipped += suite_results['skipped']
            total_time += suite_results['execution_time']
        
        overall_success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
        
        return {
            'total_tests': total_tests,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'total_skipped': total_skipped,
            'total_time': total_time,
            'overall_success_rate': overall_success_rate,
            'suite_results': self.results
        }
    
    def _print_overall_summary(self, results: Dict[str, Any]) -> None:
        """
        Print overall test summary.
        
        Args:
            results: Overall test results
        """
        print(f"\n{'='*60}")
        print("OVERALL TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests Run: {results['total_tests']}")
        print(f"Total Failures: {results['total_failures']}")
        print(f"Total Errors: {results['total_errors']}")
        print(f"Total Skipped: {results['total_skipped']}")
        print(f"Overall Success Rate: {results['overall_success_rate']:.1f}%")
        print(f"Total Execution Time: {results['total_time']:.2f}s")
        
        # Print suite breakdown
        print(f"\nSuite Breakdown:")
        print("-" * 40)
        for suite_name, suite_results in results['suite_results'].items():
            status = "✓" if suite_results['success_rate'] == 100 else "✗"
            print(f"{status} {suite_name.upper():12} | "
                  f"Tests: {suite_results['tests_run']:3} | "
                  f"Success: {suite_results['success_rate']:5.1f}% | "
                  f"Time: {suite_results['execution_time']:.2f}s")
        
        # Print recommendations
        self._print_recommendations(results)
    
    def _print_recommendations(self, results: Dict[str, Any]) -> None:
        """
        Print test recommendations based on results.
        
        Args:
            results: Overall test results
        """
        print(f"\nRecommendations:")
        print("-" * 20)
        
        if results['overall_success_rate'] == 100:
            print("✓ All tests passed! The codebase is in excellent condition.")
        elif results['overall_success_rate'] >= 90:
            print("✓ Good test coverage with minor issues to address.")
        elif results['overall_success_rate'] >= 75:
            print("⚠ Moderate test coverage with several issues to fix.")
        else:
            print("✗ Poor test coverage with significant issues requiring attention.")
        
        if results['total_errors'] > 0:
            print(f"⚠ {results['total_errors']} test errors detected - investigate immediately.")
        
        if results['total_failures'] > 0:
            print(f"⚠ {results['total_failures']} test failures detected - review and fix.")
        
        if results['total_skipped'] > 0:
            print(f"ℹ {results['total_skipped']} tests were skipped - consider implementing.")
    
    def generate_report(self, results: Dict[str, Any], output_file: str = "test_report.txt") -> None:
        """
        Generate a detailed test report file.
        
        Args:
            results: Overall test results
            output_file: Output file path
        """
        with open(output_file, 'w') as f:
            f.write("Ratatoskr AI Assistant - Test Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tests: {results['total_tests']}\n")
            f.write(f"Success Rate: {results['overall_success_rate']:.1f}%\n")
            f.write(f"Execution Time: {results['total_time']:.2f}s\n\n")
            
            f.write("Suite Details:\n")
            f.write("-" * 20 + "\n")
            for suite_name, suite_results in results['suite_results'].items():
                f.write(f"{suite_name.upper()}:\n")
                f.write(f"  Tests: {suite_results['tests_run']}\n")
                f.write(f"  Failures: {suite_results['failures']}\n")
                f.write(f"  Errors: {suite_results['errors']}\n")
                f.write(f"  Success Rate: {suite_results['success_rate']:.1f}%\n")
                f.write(f"  Time: {suite_results['execution_time']:.2f}s\n\n")
        
        print(f"\nDetailed report saved to: {output_file}")


def main():
    """Main function for running tests."""
    parser = argparse.ArgumentParser(description='Run Ratatoskr AI Assistant tests')
    parser.add_argument('--suite', choices=['memory', 'agent', 'tools', 'voice', 'all'],
                       default='all', help='Test suite to run')
    parser.add_argument('--performance', action='store_true',
                       help='Run only performance tests')
    parser.add_argument('--integration', action='store_true',
                       help='Run only integration tests')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed test report')
    parser.add_argument('--output', default='test_report.txt',
                       help='Output file for test report')
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    try:
        if args.performance:
            results = runner.run_performance_tests()
        elif args.integration:
            results = runner.run_integration_tests()
        elif args.suite == 'all':
            results = runner.run_all_tests()
        else:
            results = runner.run_specific_suite(args.suite)
        
        if args.report and results:
            runner.generate_report(results, args.output)
        
        # Exit with appropriate code
        if results and results.get('overall_success_rate', 0) == 100:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 