#!/usr/bin/env python3
"""
Automated dependency update script for CleoAI.
Safely updates dependencies while maintaining compatibility.
"""

import os
import sys
import json
import subprocess
import logging
import shutil
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import tempfile
import re
from packaging import version


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DependencyUpdater:
    """Handles safe dependency updates."""
    
    def __init__(self, project_root: str = ".", dry_run: bool = False):
        """Initialize the updater."""
        self.project_root = Path(project_root)
        self.dry_run = dry_run
        self.backup_dir = self.project_root / ".dependency_backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Files to backup before updates
        self.backup_files = [
            "requirements.txt",
            "requirements-dev.txt", 
            "pyproject.toml",
            "Pipfile",
            "package.json",
            "package-lock.json",
            "yarn.lock"
        ]
        
        # Critical packages that should be updated cautiously
        self.critical_packages = {
            "django", "flask", "fastapi", "sqlalchemy", "psycopg2",
            "redis", "celery", "gunicorn", "uvicorn", "pytest"
        }
        
        # Packages to pin (never auto-update)
        self.pinned_packages = set()
    
    def update_all_dependencies(self, security_only: bool = False) -> Dict[str, any]:
        """Update all dependencies with safety checks."""
        logger.info(f"Starting dependency update (security_only={security_only})")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "updates": [],
            "errors": [],
            "backup_created": False
        }
        
        try:
            # Create backup
            backup_path = self._create_backup()
            results["backup_created"] = True
            results["backup_path"] = str(backup_path)
            
            # Update Python dependencies
            python_results = self._update_python_dependencies(security_only)
            results["updates"].extend(python_results["updates"])
            results["errors"].extend(python_results["errors"])
            
            # Update Node.js dependencies (if applicable)
            if (self.project_root / "package.json").exists():
                node_results = self._update_node_dependencies(security_only)
                results["updates"].extend(node_results["updates"])
                results["errors"].extend(node_results["errors"])
            
            # Run tests to verify updates
            if not self.dry_run and results["updates"]:
                test_results = self._run_tests()
                if not test_results["passed"]:
                    logger.error("Tests failed after updates, rolling back")
                    self._restore_backup(backup_path)
                    results["rollback"] = True
                    results["test_failures"] = test_results["errors"]
                else:
                    results["tests_passed"] = True
            
        except Exception as e:
            logger.error(f"Update process failed: {e}")
            results["errors"].append(f"Update process failed: {e}")
            
            if results["backup_created"]:
                self._restore_backup(backup_path)
                results["rollback"] = True
        
        return results
    
    def _create_backup(self) -> Path:
        """Create backup of dependency files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{timestamp}"
        backup_path.mkdir(exist_ok=True)
        
        logger.info(f"Creating backup at {backup_path}")
        
        for file_name in self.backup_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                shutil.copy2(file_path, backup_path / file_name)
                logger.debug(f"Backed up {file_name}")
        
        return backup_path
    
    def _restore_backup(self, backup_path: Path):
        """Restore from backup."""
        logger.info(f"Restoring from backup: {backup_path}")
        
        for file_name in self.backup_files:
            backup_file = backup_path / file_name
            if backup_file.exists():
                target_file = self.project_root / file_name
                shutil.copy2(backup_file, target_file)
                logger.debug(f"Restored {file_name}")
    
    def _update_python_dependencies(self, security_only: bool) -> Dict[str, List]:
        """Update Python dependencies."""
        logger.info("Updating Python dependencies")
        
        results = {"updates": [], "errors": []}
        
        try:
            # Get current packages
            current_packages = self._get_current_python_packages()
            
            # Get available updates
            available_updates = self._get_python_updates(current_packages, security_only)
            
            for package_name, update_info in available_updates.items():
                try:
                    if self._should_update_package(package_name, update_info):
                        success = self._update_python_package(package_name, update_info)
                        if success:
                            results["updates"].append({
                                "package": package_name,
                                "from": update_info["current"],
                                "to": update_info["latest"],
                                "type": "python",
                                "security": update_info.get("security", False)
                            })
                        else:
                            results["errors"].append(f"Failed to update {package_name}")
                
                except Exception as e:
                    logger.error(f"Error updating {package_name}: {e}")
                    results["errors"].append(f"Error updating {package_name}: {e}")
        
        except Exception as e:
            logger.error(f"Python dependency update failed: {e}")
            results["errors"].append(f"Python dependency update failed: {e}")
        
        return results
    
    def _get_current_python_packages(self) -> Dict[str, str]:
        """Get currently installed Python packages."""
        packages = {}
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                package_list = json.loads(result.stdout)
                for package in package_list:
                    packages[package["name"]] = package["version"]
        
        except Exception as e:
            logger.error(f"Error getting current packages: {e}")
        
        return packages
    
    def _get_python_updates(self, current_packages: Dict[str, str], security_only: bool) -> Dict[str, Dict]:
        """Get available Python package updates."""
        updates = {}
        
        # Use pip-outdated or similar tool
        try:
            for package_name, current_version in current_packages.items():
                # Skip certain packages
                if package_name.lower() in ["pip", "setuptools", "wheel"]:
                    continue
                
                # Get package info from PyPI
                latest_info = self._get_pypi_latest(package_name)
                if not latest_info:
                    continue
                
                latest_version = latest_info.get("version")
                if not latest_version:
                    continue
                
                # Check if update is available
                try:
                    if version.parse(current_version) < version.parse(latest_version):
                        update_info = {
                            "current": current_version,
                            "latest": latest_version,
                            "security": self._is_security_update(package_name, current_version, latest_version)
                        }
                        
                        # If security_only mode, only include security updates
                        if not security_only or update_info["security"]:
                            updates[package_name] = update_info
                
                except Exception as e:
                    logger.debug(f"Error comparing versions for {package_name}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error getting updates: {e}")
        
        return updates
    
    def _get_pypi_latest(self, package_name: str) -> Optional[Dict]:
        """Get latest package info from PyPI."""
        try:
            import requests
            
            response = requests.get(
                f"https://pypi.org/pypi/{package_name}/json",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("info", {})
        
        except Exception as e:
            logger.debug(f"Error fetching PyPI info for {package_name}: {e}")
        
        return None
    
    def _is_security_update(self, package_name: str, current_version: str, latest_version: str) -> bool:
        """Check if update contains security fixes."""
        # This is a simplified check - in practice, you'd want to check
        # security databases or release notes
        
        try:
            # Check with OSV database
            import requests
            
            query = {
                "package": {
                    "name": package_name,
                    "ecosystem": "PyPI"
                },
                "version": current_version
            }
            
            response = requests.post(
                "https://api.osv.dev/v1/query",
                json=query,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                vulns = data.get("vulns", [])
                
                # Check if any vulnerability is fixed in the latest version
                for vuln in vulns:
                    affected = vuln.get("affected", [])
                    for affected_item in affected:
                        ranges = affected_item.get("ranges", [])
                        for range_item in ranges:
                            events = range_item.get("events", [])
                            for event in events:
                                if "fixed" in event:
                                    fixed_version = event["fixed"]
                                    try:
                                        if (version.parse(current_version) < version.parse(fixed_version) <= 
                                            version.parse(latest_version)):
                                            return True
                                    except:
                                        pass
        
        except Exception:
            pass
        
        return False
    
    def _should_update_package(self, package_name: str, update_info: Dict) -> bool:
        """Determine if a package should be updated."""
        # Don't update pinned packages
        if package_name.lower() in self.pinned_packages:
            logger.info(f"Skipping pinned package: {package_name}")
            return False
        
        # Always update security fixes
        if update_info.get("security", False):
            logger.info(f"Security update available for {package_name}")
            return True
        
        # Be cautious with critical packages
        if package_name.lower() in self.critical_packages:
            current_ver = version.parse(update_info["current"])
            latest_ver = version.parse(update_info["latest"])
            
            # Only allow minor/patch updates for critical packages
            if latest_ver.major > current_ver.major:
                logger.warning(f"Skipping major version update for critical package: {package_name}")
                return False
        
        return True
    
    def _update_python_package(self, package_name: str, update_info: Dict) -> bool:
        """Update a specific Python package."""
        if self.dry_run:
            logger.info(f"DRY RUN: Would update {package_name} from {update_info['current']} to {update_info['latest']}")
            return True
        
        try:
            logger.info(f"Updating {package_name} from {update_info['current']} to {update_info['latest']}")
            
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", f"{package_name}=={update_info['latest']}"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully updated {package_name}")
                return True
            else:
                logger.error(f"Failed to update {package_name}: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"Error updating {package_name}: {e}")
            return False
    
    def _update_node_dependencies(self, security_only: bool) -> Dict[str, List]:
        """Update Node.js dependencies."""
        logger.info("Updating Node.js dependencies")
        
        results = {"updates": [], "errors": []}
        
        try:
            if security_only:
                # Use npm audit fix for security updates
                if not self.dry_run:
                    result = subprocess.run(
                        ["npm", "audit", "fix"],
                        cwd=self.project_root,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    
                    if result.returncode == 0:
                        results["updates"].append({
                            "type": "node_security",
                            "description": "Applied security fixes"
                        })
                    else:
                        results["errors"].append(f"npm audit fix failed: {result.stderr}")
                else:
                    logger.info("DRY RUN: Would run npm audit fix")
            else:
                # Update all packages
                if not self.dry_run:
                    result = subprocess.run(
                        ["npm", "update"],
                        cwd=self.project_root,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    
                    if result.returncode == 0:
                        results["updates"].append({
                            "type": "node_all",
                            "description": "Updated all Node.js packages"
                        })
                    else:
                        results["errors"].append(f"npm update failed: {result.stderr}")
                else:
                    logger.info("DRY RUN: Would run npm update")
        
        except Exception as e:
            logger.error(f"Node.js dependency update failed: {e}")
            results["errors"].append(f"Node.js dependency update failed: {e}")
        
        return results
    
    def _run_tests(self) -> Dict[str, any]:
        """Run tests to verify updates didn't break anything."""
        logger.info("Running tests to verify updates")
        
        test_commands = [
            # Python tests
            [sys.executable, "-m", "pytest", "tests/", "-x", "--tb=short"],
            # Type checking
            ["mypy", "src/"],
            # Linting
            ["flake8", "src/"],
        ]
        
        results = {"passed": True, "errors": []}
        
        for cmd in test_commands:
            try:
                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                
                if result.returncode != 0:
                    results["passed"] = False
                    results["errors"].append({
                        "command": " ".join(cmd),
                        "stderr": result.stderr,
                        "stdout": result.stdout
                    })
                    logger.error(f"Test failed: {' '.join(cmd)}")
                else:
                    logger.info(f"Test passed: {' '.join(cmd)}")
            
            except subprocess.TimeoutExpired:
                results["passed"] = False
                results["errors"].append({
                    "command": " ".join(cmd),
                    "error": "Test timed out"
                })
                logger.error(f"Test timed out: {' '.join(cmd)}")
            
            except FileNotFoundError:
                # Command not found, skip
                logger.debug(f"Command not found, skipping: {' '.join(cmd)}")
                continue
            
            except Exception as e:
                results["passed"] = False
                results["errors"].append({
                    "command": " ".join(cmd),
                    "error": str(e)
                })
                logger.error(f"Test error: {' '.join(cmd)}: {e}")
        
        return results
    
    def update_requirements_file(self, file_path: str, updates: Dict[str, str]):
        """Update requirements file with new versions."""
        if self.dry_run:
            logger.info(f"DRY RUN: Would update {file_path}")
            return
        
        file_path = Path(file_path)
        if not file_path.exists():
            return
        
        logger.info(f"Updating {file_path}")
        
        # Read current content
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Update lines
        updated_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-'):
                # Parse package name
                if '==' in line:
                    package_name = line.split('==')[0].strip()
                    if package_name in updates:
                        line = f"{package_name}=={updates[package_name]}"
                elif '>=' in line:
                    package_name = line.split('>=')[0].strip()
                    if package_name in updates:
                        line = f"{package_name}>={updates[package_name]}"
            
            updated_lines.append(line + '\n')
        
        # Write updated content
        with open(file_path, 'w') as f:
            f.writelines(updated_lines)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Update dependencies safely")
    parser.add_argument("--security-only", action="store_true",
                       help="Only update packages with security vulnerabilities")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be updated without making changes")
    parser.add_argument("--project", default=".", help="Project root directory")
    parser.add_argument("--output", help="Output file for update report")
    
    args = parser.parse_args()
    
    # Run updates
    updater = DependencyUpdater(args.project, args.dry_run)
    results = updater.update_all_dependencies(args.security_only)
    
    # Save report
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Update report saved to: {args.output}")
    
    # Print summary
    print(f"\nUpdate Summary:")
    print(f"  Packages updated: {len(results['updates'])}")
    print(f"  Errors: {len(results['errors'])}")
    
    if results.get("rollback"):
        print("  Status: ROLLED BACK due to test failures")
        sys.exit(1)
    elif results["errors"]:
        print("  Status: COMPLETED WITH ERRORS")
        sys.exit(1)
    else:
        print("  Status: SUCCESS")


if __name__ == "__main__":
    main()