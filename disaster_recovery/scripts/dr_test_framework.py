#!/usr/bin/env python3
"""
Disaster Recovery Test Framework for CleoAI
Simulates various disaster scenarios and validates recovery procedures.
"""

import os
import sys
import time
import json
import logging
import asyncio
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import psycopg2
import pymongo
import redis
import requests
from kubernetes import client, config


class DisasterType(Enum):
    """Types of disasters to simulate."""
    DATABASE_FAILURE = "database_failure"
    NETWORK_PARTITION = "network_partition"
    DISK_FAILURE = "disk_failure"
    CONTAINER_CRASH = "container_crash"
    DATA_CORRUPTION = "data_corruption"
    COMPLETE_OUTAGE = "complete_outage"


class RecoveryStatus(Enum):
    """Recovery operation status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"


@dataclass
class RecoveryMetrics:
    """Metrics for recovery operations."""
    start_time: datetime
    end_time: Optional[datetime] = None
    downtime_seconds: int = 0
    data_loss_records: int = 0
    recovery_status: RecoveryStatus = RecoveryStatus.NOT_STARTED
    error_messages: List[str] = None
    
    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []


class DisasterRecoveryTester:
    """Main disaster recovery testing framework."""
    
    def __init__(self, config_file: str = "dr_config.json"):
        """Initialize DR tester with configuration."""
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        self.metrics = {}
        
        # Initialize connections
        self.k8s_client = None
        self.db_connections = {}
        self._init_connections()
    
    def _load_config(self, config_file: str) -> dict:
        """Load DR testing configuration."""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default configuration
            return {
                "postgres": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "cleoai",
                    "user": "cleoai_user"
                },
                "mongodb": {
                    "host": "localhost",
                    "port": 27017,
                    "database": "cleoai"
                },
                "redis": {
                    "host": "localhost",
                    "port": 6379,
                    "db": 0
                },
                "kubernetes": {
                    "namespace": "cleoai",
                    "context": "default"
                },
                "api": {
                    "base_url": "http://localhost:8000",
                    "health_endpoint": "/health"
                }
            }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for DR tests."""
        logger = logging.getLogger("DRTester")
        logger.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(f"dr_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        logger.addHandler(ch)
        logger.addHandler(fh)
        
        return logger
    
    def _init_connections(self):
        """Initialize database and Kubernetes connections."""
        try:
            # Kubernetes
            config.load_kube_config(context=self.config["kubernetes"]["context"])
            self.k8s_client = client.CoreV1Api()
            
            # PostgreSQL
            self.db_connections["postgres"] = psycopg2.connect(
                host=self.config["postgres"]["host"],
                port=self.config["postgres"]["port"],
                database=self.config["postgres"]["database"],
                user=self.config["postgres"]["user"],
                password=os.getenv("POSTGRES_PASSWORD", "")
            )
            
            # MongoDB
            mongo_client = pymongo.MongoClient(
                host=self.config["mongodb"]["host"],
                port=self.config["mongodb"]["port"]
            )
            self.db_connections["mongodb"] = mongo_client[self.config["mongodb"]["database"]]
            
            # Redis
            self.db_connections["redis"] = redis.Redis(
                host=self.config["redis"]["host"],
                port=self.config["redis"]["port"],
                db=self.config["redis"]["db"]
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize connections: {e}")
            raise
    
    async def simulate_disaster(self, disaster_type: DisasterType) -> RecoveryMetrics:
        """Simulate a specific disaster scenario."""
        self.logger.info(f"Starting disaster simulation: {disaster_type.value}")
        
        metrics = RecoveryMetrics(start_time=datetime.now())
        self.metrics[disaster_type] = metrics
        
        try:
            if disaster_type == DisasterType.DATABASE_FAILURE:
                await self._simulate_database_failure()
            elif disaster_type == DisasterType.NETWORK_PARTITION:
                await self._simulate_network_partition()
            elif disaster_type == DisasterType.DISK_FAILURE:
                await self._simulate_disk_failure()
            elif disaster_type == DisasterType.CONTAINER_CRASH:
                await self._simulate_container_crash()
            elif disaster_type == DisasterType.DATA_CORRUPTION:
                await self._simulate_data_corruption()
            elif disaster_type == DisasterType.COMPLETE_OUTAGE:
                await self._simulate_complete_outage()
            
            metrics.recovery_status = RecoveryStatus.IN_PROGRESS
            
        except Exception as e:
            self.logger.error(f"Disaster simulation failed: {e}")
            metrics.error_messages.append(str(e))
            metrics.recovery_status = RecoveryStatus.FAILED
        
        return metrics
    
    async def _simulate_database_failure(self):
        """Simulate database failure."""
        self.logger.info("Simulating PostgreSQL failure")
        
        # Stop PostgreSQL pod
        namespace = self.config["kubernetes"]["namespace"]
        pods = self.k8s_client.list_namespaced_pod(
            namespace=namespace,
            label_selector="app=postgresql"
        )
        
        for pod in pods.items:
            self.logger.info(f"Deleting pod: {pod.metadata.name}")
            self.k8s_client.delete_namespaced_pod(
                name=pod.metadata.name,
                namespace=namespace
            )
        
        # Wait for failure to propagate
        await asyncio.sleep(10)
    
    async def _simulate_network_partition(self):
        """Simulate network partition."""
        self.logger.info("Simulating network partition")
        
        # Apply network policy to block traffic
        network_policy = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": "dr-test-partition",
                "namespace": self.config["kubernetes"]["namespace"]
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "app": "cleoai"
                    }
                },
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [],
                "egress": []
            }
        }
        
        # Apply using kubectl
        subprocess.run([
            "kubectl", "apply", "-f", "-",
            "--namespace", self.config["kubernetes"]["namespace"]
        ], input=json.dumps(network_policy), text=True)
    
    async def _simulate_disk_failure(self):
        """Simulate disk failure."""
        self.logger.info("Simulating disk failure")
        
        # Fill up disk space in test pod
        test_pod = self._get_test_pod()
        if test_pod:
            exec_command = [
                "dd", "if=/dev/zero", "of=/data/fill_disk", 
                "bs=1M", "count=10000"
            ]
            
            stream = self.k8s_client.connect_get_namespaced_pod_exec(
                test_pod.metadata.name,
                self.config["kubernetes"]["namespace"],
                command=exec_command,
                stderr=True, stdin=False,
                stdout=True, tty=False
            )
    
    async def _simulate_container_crash(self):
        """Simulate container crashes."""
        self.logger.info("Simulating container crashes")
        
        # Kill main application containers
        namespace = self.config["kubernetes"]["namespace"]
        pods = self.k8s_client.list_namespaced_pod(
            namespace=namespace,
            label_selector="app=cleoai-api"
        )
        
        for pod in pods.items:
            for container in pod.spec.containers:
                if container.name == "api":
                    # Send SIGKILL to container
                    exec_command = ["kill", "-9", "1"]
                    try:
                        self.k8s_client.connect_get_namespaced_pod_exec(
                            pod.metadata.name,
                            namespace,
                            container=container.name,
                            command=exec_command,
                            stderr=True, stdin=False,
                            stdout=True, tty=False
                        )
                    except Exception as e:
                        self.logger.debug(f"Container kill resulted in expected error: {e}")
    
    async def _simulate_data_corruption(self):
        """Simulate data corruption."""
        self.logger.info("Simulating data corruption")
        
        # Corrupt some database records
        with self.db_connections["postgres"].cursor() as cursor:
            # Create backup first
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations_backup AS 
                SELECT * FROM conversations
            """)
            
            # Corrupt data
            cursor.execute("""
                UPDATE conversations 
                SET metadata = '{"corrupted": true, "invalid_json": }'
                WHERE id IN (
                    SELECT id FROM conversations 
                    ORDER BY RANDOM() 
                    LIMIT 10
                )
            """)
            self.db_connections["postgres"].commit()
    
    async def _simulate_complete_outage(self):
        """Simulate complete system outage."""
        self.logger.info("Simulating complete outage")
        
        # Scale all deployments to 0
        apps_v1 = client.AppsV1Api()
        namespace = self.config["kubernetes"]["namespace"]
        
        deployments = apps_v1.list_namespaced_deployment(namespace=namespace)
        
        for deployment in deployments.items:
            self.logger.info(f"Scaling down deployment: {deployment.metadata.name}")
            
            # Scale to 0 replicas
            deployment.spec.replicas = 0
            apps_v1.patch_namespaced_deployment(
                name=deployment.metadata.name,
                namespace=namespace,
                body=deployment
            )
    
    async def execute_recovery(self, disaster_type: DisasterType) -> RecoveryMetrics:
        """Execute recovery procedures for a disaster."""
        metrics = self.metrics.get(disaster_type)
        if not metrics:
            raise ValueError(f"No disaster simulation found for {disaster_type}")
        
        self.logger.info(f"Starting recovery for: {disaster_type.value}")
        
        try:
            if disaster_type == DisasterType.DATABASE_FAILURE:
                await self._recover_database()
            elif disaster_type == DisasterType.NETWORK_PARTITION:
                await self._recover_network()
            elif disaster_type == DisasterType.DISK_FAILURE:
                await self._recover_disk()
            elif disaster_type == DisasterType.CONTAINER_CRASH:
                await self._recover_containers()
            elif disaster_type == DisasterType.DATA_CORRUPTION:
                await self._recover_corrupted_data()
            elif disaster_type == DisasterType.COMPLETE_OUTAGE:
                await self._recover_complete_outage()
            
            metrics.end_time = datetime.now()
            metrics.downtime_seconds = (metrics.end_time - metrics.start_time).total_seconds()
            metrics.recovery_status = RecoveryStatus.COMPLETED
            
        except Exception as e:
            self.logger.error(f"Recovery failed: {e}")
            metrics.error_messages.append(str(e))
            metrics.recovery_status = RecoveryStatus.FAILED
        
        return metrics
    
    async def _recover_database(self):
        """Recover from database failure."""
        self.logger.info("Recovering database")
        
        # Run database recovery script
        result = subprocess.run([
            "./scripts/dr_restore_postgres.sh", "--latest"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Database recovery failed: {result.stderr}")
        
        # Wait for database to be ready
        await self._wait_for_postgres()
    
    async def _recover_network(self):
        """Recover from network partition."""
        self.logger.info("Recovering network")
        
        # Delete network policy
        subprocess.run([
            "kubectl", "delete", "networkpolicy",
            "dr-test-partition",
            "--namespace", self.config["kubernetes"]["namespace"]
        ])
        
        # Wait for network recovery
        await asyncio.sleep(5)
    
    async def _recover_disk(self):
        """Recover from disk failure."""
        self.logger.info("Recovering disk space")
        
        # Clean up disk space
        test_pod = self._get_test_pod()
        if test_pod:
            exec_command = ["rm", "-f", "/data/fill_disk"]
            
            self.k8s_client.connect_get_namespaced_pod_exec(
                test_pod.metadata.name,
                self.config["kubernetes"]["namespace"],
                command=exec_command,
                stderr=True, stdin=False,
                stdout=True, tty=False
            )
    
    async def _recover_containers(self):
        """Recover from container crashes."""
        self.logger.info("Recovering containers")
        
        # Kubernetes should automatically restart crashed containers
        # Wait for pods to be ready
        await self._wait_for_pods_ready("app=cleoai-api")
    
    async def _recover_corrupted_data(self):
        """Recover from data corruption."""
        self.logger.info("Recovering corrupted data")
        
        with self.db_connections["postgres"].cursor() as cursor:
            # Restore from backup
            cursor.execute("""
                UPDATE conversations c
                SET metadata = b.metadata
                FROM conversations_backup b
                WHERE c.id = b.id
                AND c.metadata::text LIKE '%corrupted%'
            """)
            
            # Clean up backup table
            cursor.execute("DROP TABLE IF EXISTS conversations_backup")
            
            self.db_connections["postgres"].commit()
    
    async def _recover_complete_outage(self):
        """Recover from complete outage."""
        self.logger.info("Recovering from complete outage")
        
        # Scale deployments back up
        apps_v1 = client.AppsV1Api()
        namespace = self.config["kubernetes"]["namespace"]
        
        # Read desired replicas from configmap or use defaults
        deployment_replicas = {
            "cleoai-api": 3,
            "cleoai-worker": 2,
            "cleoai-scheduler": 1
        }
        
        deployments = apps_v1.list_namespaced_deployment(namespace=namespace)
        
        for deployment in deployments.items:
            desired_replicas = deployment_replicas.get(deployment.metadata.name, 1)
            self.logger.info(f"Scaling up {deployment.metadata.name} to {desired_replicas} replicas")
            
            deployment.spec.replicas = desired_replicas
            apps_v1.patch_namespaced_deployment(
                name=deployment.metadata.name,
                namespace=namespace,
                body=deployment
            )
        
        # Wait for all pods to be ready
        await self._wait_for_pods_ready()
    
    async def verify_recovery(self, disaster_type: DisasterType) -> bool:
        """Verify that recovery was successful."""
        metrics = self.metrics.get(disaster_type)
        if not metrics:
            return False
        
        self.logger.info(f"Verifying recovery for: {disaster_type.value}")
        
        try:
            # Check API health
            health_ok = await self._check_api_health()
            
            # Check database connectivity
            db_ok = await self._check_database_health()
            
            # Check data integrity
            data_ok = await self._check_data_integrity()
            
            # Check performance
            perf_ok = await self._check_performance()
            
            all_ok = health_ok and db_ok and data_ok and perf_ok
            
            if all_ok:
                metrics.recovery_status = RecoveryStatus.VERIFIED
                self.logger.info("Recovery verification: PASSED")
            else:
                metrics.error_messages.append("Recovery verification failed")
                self.logger.error("Recovery verification: FAILED")
            
            return all_ok
            
        except Exception as e:
            self.logger.error(f"Recovery verification error: {e}")
            metrics.error_messages.append(str(e))
            return False
    
    async def _check_api_health(self) -> bool:
        """Check API health status."""
        try:
            url = f"{self.config['api']['base_url']}{self.config['api']['health_endpoint']}"
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"API health check failed: {e}")
            return False
    
    async def _check_database_health(self) -> bool:
        """Check database health."""
        try:
            # PostgreSQL
            with self.db_connections["postgres"].cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            
            # MongoDB
            self.db_connections["mongodb"].command("ping")
            
            # Redis
            self.db_connections["redis"].ping()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return False
    
    async def _check_data_integrity(self) -> bool:
        """Check data integrity after recovery."""
        try:
            # Check for data loss
            with self.db_connections["postgres"].cursor() as cursor:
                cursor.execute("""
                    SELECT COUNT(*) FROM conversations 
                    WHERE created_at > %s
                """, (datetime.now() - timedelta(hours=1),))
                
                recent_count = cursor.fetchone()[0]
                
                if recent_count == 0:
                    self.logger.warning("No recent data found - possible data loss")
                    return False
            
            # Check for corruption
            cursor.execute("""
                SELECT COUNT(*) FROM conversations 
                WHERE metadata IS NOT NULL 
                AND NOT (metadata::text ~ '^\\{.*\\}$')
            """)
            
            corrupt_count = cursor.fetchone()[0]
            
            if corrupt_count > 0:
                self.logger.error(f"Found {corrupt_count} corrupted records")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data integrity check failed: {e}")
            return False
    
    async def _check_performance(self) -> bool:
        """Check system performance after recovery."""
        try:
            # Simple performance test
            start_time = time.time()
            
            # Make test API call
            url = f"{self.config['api']['base_url']}/graphql"
            query = {
                "query": "{ __typename }"
            }
            
            response = requests.post(url, json=query, timeout=5)
            
            response_time = time.time() - start_time
            
            if response_time > 2.0:
                self.logger.warning(f"Slow API response: {response_time:.2f}s")
                return False
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Performance check failed: {e}")
            return False
    
    async def _wait_for_postgres(self, timeout: int = 300):
        """Wait for PostgreSQL to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                conn = psycopg2.connect(
                    host=self.config["postgres"]["host"],
                    port=self.config["postgres"]["port"],
                    database=self.config["postgres"]["database"],
                    user=self.config["postgres"]["user"],
                    password=os.getenv("POSTGRES_PASSWORD", ""),
                    connect_timeout=5
                )
                conn.close()
                return
            except Exception:
                await asyncio.sleep(5)
        
        raise TimeoutError("PostgreSQL did not become ready in time")
    
    async def _wait_for_pods_ready(self, label_selector: str = None, timeout: int = 300):
        """Wait for pods to be ready."""
        namespace = self.config["kubernetes"]["namespace"]
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            pods = self.k8s_client.list_namespaced_pod(
                namespace=namespace,
                label_selector=label_selector
            )
            
            all_ready = True
            for pod in pods.items:
                if pod.status.phase != "Running":
                    all_ready = False
                    break
                
                for container_status in pod.status.container_statuses or []:
                    if not container_status.ready:
                        all_ready = False
                        break
            
            if all_ready and len(pods.items) > 0:
                return
            
            await asyncio.sleep(5)
        
        raise TimeoutError("Pods did not become ready in time")
    
    def _get_test_pod(self):
        """Get a test pod for operations."""
        namespace = self.config["kubernetes"]["namespace"]
        pods = self.k8s_client.list_namespaced_pod(
            namespace=namespace,
            label_selector="app=cleoai-api"
        )
        
        return pods.items[0] if pods.items else None
    
    def generate_report(self) -> str:
        """Generate DR test report."""
        report = []
        report.append("=" * 60)
        report.append("Disaster Recovery Test Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        report.append("")
        
        for disaster_type, metrics in self.metrics.items():
            report.append(f"Disaster Type: {disaster_type.value}")
            report.append("-" * 40)
            report.append(f"Status: {metrics.recovery_status.value}")
            report.append(f"Start Time: {metrics.start_time}")
            report.append(f"End Time: {metrics.end_time}")
            report.append(f"Downtime: {metrics.downtime_seconds:.2f} seconds")
            report.append(f"Data Loss: {metrics.data_loss_records} records")
            
            if metrics.error_messages:
                report.append("Errors:")
                for error in metrics.error_messages:
                    report.append(f"  - {error}")
            
            report.append("")
        
        # Summary
        total_tests = len(self.metrics)
        successful = sum(1 for m in self.metrics.values() 
                        if m.recovery_status == RecoveryStatus.VERIFIED)
        
        report.append("Summary:")
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Successful: {successful}")
        report.append(f"Failed: {total_tests - successful}")
        report.append(f"Success Rate: {(successful/total_tests*100) if total_tests > 0 else 0:.1f}%")
        
        return "\n".join(report)


async def main():
    """Run disaster recovery tests."""
    tester = DisasterRecoveryTester()
    
    # Test scenarios
    scenarios = [
        DisasterType.DATABASE_FAILURE,
        DisasterType.CONTAINER_CRASH,
        DisasterType.DATA_CORRUPTION,
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Testing scenario: {scenario.value}")
        print('='*60)
        
        # Simulate disaster
        await tester.simulate_disaster(scenario)
        
        # Wait a bit
        await asyncio.sleep(10)
        
        # Execute recovery
        await tester.execute_recovery(scenario)
        
        # Verify recovery
        success = await tester.verify_recovery(scenario)
        
        print(f"Recovery {'SUCCESSFUL' if success else 'FAILED'}")
    
    # Generate report
    report = tester.generate_report()
    print("\n" + report)
    
    # Save report
    with open(f"dr_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'w') as f:
        f.write(report)


if __name__ == "__main__":
    asyncio.run(main())