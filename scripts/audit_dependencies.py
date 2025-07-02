#!/usr/bin/env python3
"""
Dependency vulnerability audit script for CleoAI.
Scans and reports security vulnerabilities in Python and Node.js dependencies.
"""

import os
import sys
import json
import subprocess
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import requests
import semver
from packaging import version


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Vulnerability:
    """Represents a security vulnerability."""
    package_name: str
    installed_version: str
    vulnerability_id: str
    severity: str
    title: str
    description: str
    affected_versions: str
    fixed_version: Optional[str]
    cve_id: Optional[str] = None
    advisory_url: Optional[str] = None


@dataclass
class DependencyInfo:
    """Information about a dependency."""
    name: str
    version: str
    latest_version: str
    update_available: bool
    vulnerabilities: List[Vulnerability]
    license: Optional[str] = None
    repository: Optional[str] = None


@dataclass
class AuditReport:
    """Complete audit report."""
    timestamp: datetime
    total_packages: int
    vulnerable_packages: int
    outdated_packages: int
    critical_vulnerabilities: int
    high_vulnerabilities: int
    medium_vulnerabilities: int
    low_vulnerabilities: int
    dependencies: List[DependencyInfo]
    recommendations: List[str]


class DependencyAuditor:
    """Main dependency auditing class."""
    
    def __init__(self, project_root: str = "."):
        """Initialize the auditor."""
        self.project_root = Path(project_root)
        self.cache_dir = self.project_root / ".dependency_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Severity levels
        self.severity_levels = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1,
            "info": 0
        }
        
        # Package managers to check
        self.package_files = {
            "python": ["requirements.txt", "requirements-dev.txt", "pyproject.toml", "Pipfile"],
            "node": ["package.json", "package-lock.json", "yarn.lock"]
        }
    
    def run_full_audit(self) -> AuditReport:
        """Run complete dependency audit."""
        logger.info("Starting full dependency audit")
        
        all_dependencies = []
        
        # Audit Python dependencies
        python_deps = self._audit_python_dependencies()
        all_dependencies.extend(python_deps)
        
        # Audit Node.js dependencies (if applicable)
        node_deps = self._audit_node_dependencies()
        all_dependencies.extend(node_deps)
        
        # Generate report
        report = self._generate_report(all_dependencies)
        
        logger.info(f"Audit complete. Found {report.vulnerable_packages} vulnerable packages")
        
        return report
    
    def _audit_python_dependencies(self) -> List[DependencyInfo]:
        """Audit Python dependencies."""
        logger.info("Auditing Python dependencies")
        
        dependencies = []
        
        # Get installed packages
        installed_packages = self._get_installed_python_packages()
        
        if not installed_packages:
            logger.warning("No Python packages found")
            return dependencies
        
        # Check each package for vulnerabilities
        for package_name, package_version in installed_packages.items():
            try:
                dep_info = self._check_python_package(package_name, package_version)
                dependencies.append(dep_info)
            except Exception as e:
                logger.error(f"Error checking {package_name}: {e}")
                # Create minimal dependency info for failed checks
                dependencies.append(DependencyInfo(
                    name=package_name,
                    version=package_version,
                    latest_version="unknown",
                    update_available=False,
                    vulnerabilities=[],
                    license="unknown"
                ))
        
        return dependencies
    
    def _audit_node_dependencies(self) -> List[DependencyInfo]:
        """Audit Node.js dependencies."""
        package_json = self.project_root / "package.json"
        
        if not package_json.exists():
            logger.info("No package.json found, skipping Node.js audit")
            return []
        
        logger.info("Auditing Node.js dependencies")
        
        dependencies = []
        
        try:
            # Use npm audit for vulnerability scanning
            result = subprocess.run(
                ["npm", "audit", "--json"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0 or result.stdout:
                audit_data = json.loads(result.stdout)
                dependencies = self._parse_npm_audit(audit_data)
            
        except subprocess.TimeoutExpired:
            logger.error("npm audit timed out")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse npm audit output: {e}")
        except FileNotFoundError:
            logger.warning("npm not found, skipping Node.js audit")
        except Exception as e:
            logger.error(f"Error running npm audit: {e}")
        
        return dependencies
    
    def _get_installed_python_packages(self) -> Dict[str, str]:
        """Get list of installed Python packages."""
        packages = {}
        
        try:
            # Use pip list to get installed packages
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
            logger.error(f"Error getting installed packages: {e}")
            
            # Fallback: read from requirements files
            for req_file in ["requirements.txt", "requirements-dev.txt"]:
                req_path = self.project_root / req_file
                if req_path.exists():
                    packages.update(self._parse_requirements_file(req_path))
        
        return packages
    
    def _parse_requirements_file(self, file_path: Path) -> Dict[str, str]:
        """Parse requirements.txt file."""
        packages = {}
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('-'):
                        # Parse package==version format
                        if '==' in line:
                            name, version = line.split('==', 1)
                            packages[name.strip()] = version.strip()
                        elif '>=' in line:
                            name = line.split('>=')[0].strip()
                            packages[name] = "latest"
        
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
        
        return packages
    
    def _check_python_package(self, package_name: str, package_version: str) -> DependencyInfo:
        """Check a Python package for vulnerabilities."""
        # Get package info from PyPI
        pypi_info = self._get_pypi_package_info(package_name)
        
        # Get latest version
        latest_version = pypi_info.get("info", {}).get("version", "unknown")
        
        # Check if update is available
        update_available = False
        try:
            if package_version != "latest" and latest_version != "unknown":
                update_available = version.parse(package_version) < version.parse(latest_version)
        except Exception:
            pass
        
        # Check for vulnerabilities using safety database
        vulnerabilities = self._check_python_vulnerabilities(package_name, package_version)
        
        # Get license info
        license_info = pypi_info.get("info", {}).get("license", "")
        
        # Get repository URL
        repository = None
        project_urls = pypi_info.get("info", {}).get("project_urls", {})
        if isinstance(project_urls, dict):
            repository = project_urls.get("Homepage") or project_urls.get("Repository")
        
        return DependencyInfo(
            name=package_name,
            version=package_version,
            latest_version=latest_version,
            update_available=update_available,
            vulnerabilities=vulnerabilities,
            license=license_info,
            repository=repository
        )
    
    def _get_pypi_package_info(self, package_name: str) -> Dict[str, Any]:
        """Get package information from PyPI."""
        cache_file = self.cache_dir / f"pypi_{package_name}.json"
        
        # Check cache first (cache for 1 hour)
        if cache_file.exists():
            if time.time() - cache_file.stat().st_mtime < 3600:
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception:
                    pass
        
        # Fetch from PyPI
        try:
            response = requests.get(
                f"https://pypi.org/pypi/{package_name}/json",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Cache the response
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
                
                return data
        
        except Exception as e:
            logger.debug(f"Error fetching PyPI info for {package_name}: {e}")
        
        return {}
    
    def _check_python_vulnerabilities(self, package_name: str, package_version: str) -> List[Vulnerability]:
        """Check for vulnerabilities in a Python package."""
        vulnerabilities = []
        
        # Use safety database (OSV database)
        try:
            vulnerabilities.extend(self._check_osv_vulnerabilities(package_name, package_version, "PyPI"))
        except Exception as e:
            logger.debug(f"Error checking OSV for {package_name}: {e}")
        
        return vulnerabilities
    
    def _check_osv_vulnerabilities(self, package_name: str, package_version: str, ecosystem: str) -> List[Vulnerability]:
        """Check vulnerabilities using OSV (Open Source Vulnerabilities) database."""
        vulnerabilities = []
        
        try:
            # Query OSV API
            query = {
                "package": {
                    "name": package_name,
                    "ecosystem": ecosystem
                },
                "version": package_version
            }
            
            response = requests.post(
                "https://api.osv.dev/v1/query",
                json=query,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                for vuln in data.get("vulns", []):
                    # Parse vulnerability info
                    vuln_id = vuln.get("id", "")
                    summary = vuln.get("summary", "")
                    details = vuln.get("details", "")
                    
                    # Get severity
                    severity = "medium"  # default
                    severity_info = vuln.get("severity", [])
                    if severity_info:
                        severity = severity_info[0].get("score", "MEDIUM").lower()
                    
                    # Get affected versions
                    affected = vuln.get("affected", [])
                    affected_versions = ""
                    fixed_version = None
                    
                    for affected_item in affected:
                        if affected_item.get("package", {}).get("name") == package_name:
                            ranges = affected_item.get("ranges", [])
                            for range_item in ranges:
                                events = range_item.get("events", [])
                                for event in events:
                                    if "fixed" in event:
                                        fixed_version = event["fixed"]
                                        break
                    
                    # Get CVE ID
                    cve_id = None
                    aliases = vuln.get("aliases", [])
                    for alias in aliases:
                        if alias.startswith("CVE-"):
                            cve_id = alias
                            break
                    
                    # Get advisory URL
                    advisory_url = f"https://osv.dev/vulnerability/{vuln_id}"
                    
                    vulnerabilities.append(Vulnerability(
                        package_name=package_name,
                        installed_version=package_version,
                        vulnerability_id=vuln_id,
                        severity=severity,
                        title=summary,
                        description=details,
                        affected_versions=affected_versions,
                        fixed_version=fixed_version,
                        cve_id=cve_id,
                        advisory_url=advisory_url
                    ))
        
        except Exception as e:
            logger.debug(f"Error querying OSV for {package_name}: {e}")
        
        return vulnerabilities
    
    def _parse_npm_audit(self, audit_data: Dict[str, Any]) -> List[DependencyInfo]:
        """Parse npm audit output."""
        dependencies = []
        
        try:
            # npm audit v7+ format
            vulnerabilities = audit_data.get("vulnerabilities", {})
            
            for package_name, vuln_info in vulnerabilities.items():
                # Get package version (this might need adjustment based on npm audit format)
                version_info = vuln_info.get("via", [])
                installed_version = "unknown"
                
                # Parse vulnerabilities
                package_vulns = []
                for via in version_info:
                    if isinstance(via, dict):
                        package_vulns.append(Vulnerability(
                            package_name=package_name,
                            installed_version=installed_version,
                            vulnerability_id=via.get("url", "").split("/")[-1] if via.get("url") else str(via.get("id", "")),
                            severity=via.get("severity", "medium"),
                            title=via.get("title", ""),
                            description=via.get("overview", ""),
                            affected_versions=via.get("range", ""),
                            fixed_version=None,
                            cve_id=via.get("cve", [None])[0] if via.get("cve") else None,
                            advisory_url=via.get("url")
                        ))
                
                dependencies.append(DependencyInfo(
                    name=package_name,
                    version=installed_version,
                    latest_version="unknown",
                    update_available=False,
                    vulnerabilities=package_vulns
                ))
        
        except Exception as e:
            logger.error(f"Error parsing npm audit data: {e}")
        
        return dependencies
    
    def _generate_report(self, dependencies: List[DependencyInfo]) -> AuditReport:
        """Generate comprehensive audit report."""
        total_packages = len(dependencies)
        vulnerable_packages = sum(1 for dep in dependencies if dep.vulnerabilities)
        outdated_packages = sum(1 for dep in dependencies if dep.update_available)
        
        # Count vulnerabilities by severity
        critical_count = 0
        high_count = 0
        medium_count = 0
        low_count = 0
        
        for dep in dependencies:
            for vuln in dep.vulnerabilities:
                severity = vuln.severity.lower()
                if severity in ["critical", "severe"]:
                    critical_count += 1
                elif severity == "high":
                    high_count += 1
                elif severity == "medium":
                    medium_count += 1
                else:
                    low_count += 1
        
        # Generate recommendations
        recommendations = self._generate_recommendations(dependencies)
        
        return AuditReport(
            timestamp=datetime.now(),
            total_packages=total_packages,
            vulnerable_packages=vulnerable_packages,
            outdated_packages=outdated_packages,
            critical_vulnerabilities=critical_count,
            high_vulnerabilities=high_count,
            medium_vulnerabilities=medium_count,
            low_vulnerabilities=low_count,
            dependencies=dependencies,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, dependencies: List[DependencyInfo]) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        # Check for critical vulnerabilities
        critical_packages = []
        for dep in dependencies:
            for vuln in dep.vulnerabilities:
                if vuln.severity.lower() in ["critical", "severe"]:
                    critical_packages.append((dep.name, vuln.fixed_version))
        
        if critical_packages:
            recommendations.append(
                f"URGENT: Update packages with critical vulnerabilities: {', '.join([p[0] for p in critical_packages])}"
            )
        
        # Check for outdated packages
        outdated_count = sum(1 for dep in dependencies if dep.update_available)
        if outdated_count > 10:
            recommendations.append(
                f"Consider updating {outdated_count} outdated packages to latest versions"
            )
        
        # Check for packages without licenses
        unlicensed = [dep.name for dep in dependencies if not dep.license or dep.license.lower() in ["unknown", ""]]
        if unlicensed:
            recommendations.append(
                f"Review license compliance for packages without clear licenses: {', '.join(unlicensed[:5])}"
                + ("..." if len(unlicensed) > 5 else "")
            )
        
        # General recommendations
        if any(dep.vulnerabilities for dep in dependencies):
            recommendations.append("Run 'pip-audit' or 'safety check' regularly to monitor for new vulnerabilities")
            recommendations.append("Consider implementing automated dependency updates with security scanning")
        
        recommendations.append("Pin dependency versions in production environments")
        recommendations.append("Use virtual environments to isolate project dependencies")
        
        return recommendations
    
    def save_report(self, report: AuditReport, format: str = "json") -> str:
        """Save audit report to file."""
        timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            filename = f"dependency_audit_{timestamp}.json"
            filepath = self.project_root / filename
            
            with open(filepath, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
        
        elif format == "html":
            filename = f"dependency_audit_{timestamp}.html"
            filepath = self.project_root / filename
            
            html_content = self._generate_html_report(report)
            with open(filepath, 'w') as f:
                f.write(html_content)
        
        else:
            filename = f"dependency_audit_{timestamp}.txt"
            filepath = self.project_root / filename
            
            text_content = self._generate_text_report(report)
            with open(filepath, 'w') as f:
                f.write(text_content)
        
        return str(filepath)
    
    def _generate_text_report(self, report: AuditReport) -> str:
        """Generate text format report."""
        lines = []
        lines.append("=" * 60)
        lines.append("CLEOAI DEPENDENCY SECURITY AUDIT REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {report.timestamp}")
        lines.append("")
        
        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 20)
        lines.append(f"Total packages: {report.total_packages}")
        lines.append(f"Vulnerable packages: {report.vulnerable_packages}")
        lines.append(f"Outdated packages: {report.outdated_packages}")
        lines.append("")
        
        # Vulnerabilities by severity
        lines.append("VULNERABILITIES BY SEVERITY")
        lines.append("-" * 30)
        lines.append(f"Critical: {report.critical_vulnerabilities}")
        lines.append(f"High: {report.high_vulnerabilities}")
        lines.append(f"Medium: {report.medium_vulnerabilities}")
        lines.append(f"Low: {report.low_vulnerabilities}")
        lines.append("")
        
        # Vulnerable packages
        if report.vulnerable_packages > 0:
            lines.append("VULNERABLE PACKAGES")
            lines.append("-" * 25)
            
            for dep in report.dependencies:
                if dep.vulnerabilities:
                    lines.append(f"\n{dep.name} ({dep.version})")
                    for vuln in dep.vulnerabilities:
                        lines.append(f"  • {vuln.vulnerability_id} - {vuln.severity.upper()}")
                        lines.append(f"    {vuln.title}")
                        if vuln.fixed_version:
                            lines.append(f"    Fixed in: {vuln.fixed_version}")
        
        # Recommendations
        if report.recommendations:
            lines.append("\n\nRECOMMENDATIONS")
            lines.append("-" * 20)
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"{i}. {rec}")
        
        return "\n".join(lines)
    
    def _generate_html_report(self, report: AuditReport) -> str:
        """Generate HTML format report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CleoAI Dependency Audit Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .critical {{ color: #d32f2f; font-weight: bold; }}
        .high {{ color: #f57c00; font-weight: bold; }}
        .medium {{ color: #fbc02d; font-weight: bold; }}
        .low {{ color: #388e3c; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>CleoAI Dependency Security Audit Report</h1>
        <p>Generated: {report.timestamp}</p>
    </div>
    
    <h2>Summary</h2>
    <ul>
        <li>Total packages: {report.total_packages}</li>
        <li>Vulnerable packages: {report.vulnerable_packages}</li>
        <li>Outdated packages: {report.outdated_packages}</li>
    </ul>
    
    <h2>Vulnerabilities by Severity</h2>
    <ul>
        <li class="critical">Critical: {report.critical_vulnerabilities}</li>
        <li class="high">High: {report.high_vulnerabilities}</li>
        <li class="medium">Medium: {report.medium_vulnerabilities}</li>
        <li class="low">Low: {report.low_vulnerabilities}</li>
    </ul>
"""
        
        # Add vulnerable packages table
        vulnerable_deps = [dep for dep in report.dependencies if dep.vulnerabilities]
        if vulnerable_deps:
            html += """
    <h2>Vulnerable Packages</h2>
    <table>
        <tr>
            <th>Package</th>
            <th>Version</th>
            <th>Vulnerability</th>
            <th>Severity</th>
            <th>Fixed Version</th>
        </tr>
"""
            for dep in vulnerable_deps:
                for vuln in dep.vulnerabilities:
                    severity_class = vuln.severity.lower()
                    html += f"""
        <tr>
            <td>{dep.name}</td>
            <td>{dep.version}</td>
            <td>{vuln.title or vuln.vulnerability_id}</td>
            <td class="{severity_class}">{vuln.severity.upper()}</td>
            <td>{vuln.fixed_version or 'N/A'}</td>
        </tr>
"""
            html += "</table>"
        
        # Add recommendations
        if report.recommendations:
            html += "<h2>Recommendations</h2><ol>"
            for rec in report.recommendations:
                html += f"<li>{rec}</li>"
            html += "</ol>"
        
        html += "</body></html>"
        return html


def main():
    """Main function to run dependency audit."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Audit dependencies for security vulnerabilities")
    parser.add_argument("--format", choices=["json", "html", "text"], default="text",
                       help="Output format for the report")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--project", default=".", help="Project root directory")
    
    args = parser.parse_args()
    
    # Run audit
    auditor = DependencyAuditor(args.project)
    report = auditor.run_full_audit()
    
    # Save report
    if args.output:
        output_path = args.output
        with open(output_path, 'w') as f:
            if args.format == "json":
                json.dump(asdict(report), f, indent=2, default=str)
            else:
                f.write(auditor._generate_text_report(report))
    else:
        output_path = auditor.save_report(report, args.format)
    
    print(f"Audit report saved to: {output_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total packages: {report.total_packages}")
    print(f"  Vulnerable packages: {report.vulnerable_packages}")
    print(f"  Critical vulnerabilities: {report.critical_vulnerabilities}")
    print(f"  High vulnerabilities: {report.high_vulnerabilities}")
    
    # Exit with error code if critical vulnerabilities found
    if report.critical_vulnerabilities > 0:
        print("\nWARNING: Critical vulnerabilities found!")
        sys.exit(1)


if __name__ == "__main__":
    main()