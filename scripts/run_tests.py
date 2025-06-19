#!/usr/bin/env python3
"""
Script to run the test suite with different configurations.
"""
import subprocess
import sys
from pathlib import Path

def run_tests(test_path="tests/", verbose=True, coverage=False):
    """Run pytest with specified options."""
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    cmd.append(test_path)
    
    result = subprocess.run(cmd, cwd=Path.cwd())
    return result.returncode

if __name__ == "__main__":
    # Run different test suites
    print("Running unit tests...")
    run_tests("tests/unit/")
    
    print("\nRunning integration tests...")
    run_tests("tests/integration/")
    
    print("\nRunning all tests with coverage...")
    run_tests(coverage=True)