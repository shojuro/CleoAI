#!/usr/bin/env python3
"""
Verify that the test suite structure is correct and would execute properly.
This script checks for common issues in the test files.
"""

import os
import re
import ast
from pathlib import Path

def check_imports(file_path):
    """Check if imports in test file are valid."""
    issues = []
    
    with open(file_path, 'r') as f:
        content = f.read()
        
    # Parse the AST
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]
    
    # Check imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                if module_name.startswith('src.'):
                    # Check if the module exists
                    module_path = module_name.replace('.', '/')
                    if not os.path.exists(f"{module_path}.py") and not os.path.exists(module_path):
                        issues.append(f"Import not found: {module_name}")
                        
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith('src.'):
                module_path = node.module.replace('.', '/')
                if not os.path.exists(f"{module_path}.py") and not os.path.exists(module_path):
                    issues.append(f"Import not found: {node.module}")
    
    return issues

def check_test_structure(file_path):
    """Check if test file follows proper structure."""
    issues = []
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for test classes and functions
    if not re.search(r'class Test\w+:', content):
        if not re.search(r'def test_\w+\(', content):
            issues.append("No test classes or functions found")
    
    # Check for proper pytest usage
    if 'pytest' not in content:
        issues.append("pytest not imported")
    
    # Check for assert statements
    if 'assert' not in content:
        issues.append("No assert statements found")
    
    return issues

def main():
    """Main verification function."""
    test_dirs = ['tests/security', 'tests/integration', 'tests/api']
    all_issues = {}
    
    print("CleoAI Test Suite Verification Report")
    print("=" * 50)
    
    for test_dir in test_dirs:
        if not os.path.exists(test_dir):
            print(f"\n❌ Directory not found: {test_dir}")
            continue
            
        print(f"\n📁 Checking {test_dir}/")
        print("-" * 40)
        
        test_files = list(Path(test_dir).glob("test_*.py"))
        
        if not test_files:
            print("  ⚠️  No test files found")
            continue
            
        for test_file in test_files:
            file_issues = []
            
            # Check imports
            import_issues = check_imports(test_file)
            file_issues.extend(import_issues)
            
            # Check structure
            structure_issues = check_test_structure(test_file)
            file_issues.extend(structure_issues)
            
            if file_issues:
                print(f"\n  ❌ {test_file.name}")
                for issue in file_issues:
                    print(f"     - {issue}")
                all_issues[str(test_file)] = file_issues
            else:
                print(f"  ✅ {test_file.name}")
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    
    if all_issues:
        print(f"\n⚠️  Found issues in {len(all_issues)} test files")
        print("\nTo fix these issues:")
        print("1. Ensure all imported modules exist in the src/ directory")
        print("2. Install required dependencies: pytest, pytest-asyncio, pytest-cov")
        print("3. Create any missing source files that tests depend on")
    else:
        print("\n✅ All test files appear to be properly structured!")
        print("\nTo run the tests:")
        print("1. Install pytest: pip install pytest pytest-asyncio pytest-cov")
        print("2. Run tests: python -m pytest tests/ -v")
    
    # Test statistics
    total_files = sum(len(list(Path(d).glob("test_*.py"))) for d in test_dirs if os.path.exists(d))
    print(f"\nTotal test files: {total_files}")
    
    # Count test cases (approximate)
    total_tests = 0
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for test_file in Path(test_dir).glob("test_*.py"):
                with open(test_file, 'r') as f:
                    content = f.read()
                    total_tests += len(re.findall(r'def test_\w+\(', content))
    
    print(f"Approximate test cases: {total_tests}")

if __name__ == "__main__":
    main()