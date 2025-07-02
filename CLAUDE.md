# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Foundation & AI Development Guidelines

This document outlines the core principles and critical considerations for developing applications with AI assistance. Adhering to these guidelines is paramount for building secure, robust, efficient, and maintainable software. **Security is the top priority.**

---

### **Part 1: Fundamental Principles for AI-Assisted Development**

**1. Verification and Critical Thinking:**
* **Always Verify:** Do not assume AI-generated code is correct or complete. Independently verify all logic, data flows, and configurations.
* **Understand the "Why":** For every code snippet or architectural suggestion, be prepared to explain the underlying reasoning, design patterns, and chosen technologies. This is crucial for learning and effective debugging.
* **No "Full Yolo Mode":** Approach development systematically. AI should create a plan, and each step must be reviewed and thoroughly tested before proceeding.

**2. Security-First Mindset (Paramount Priority):**
* **Assume Breach:** Design and code with the assumption that your system *will* be attacked.
* **Proactive Threat Modeling (OWASP A04: Insecure Design):** Design for security from the ground up. Use AI to help brainstorm potential threats and attack vectors specific to the application's design. Leverage AI to analyze design documents for inherent security weaknesses and suggest secure design patterns and principles.
* **Least Privilege:** Ensure all generated code, configurations, and IAM roles adhere strictly to the principle of least privilege, requesting only the necessary permissions and resources.

**3. Quality and Maintainability:**
* **Modularity:** Generate code that is modular, loosely coupled, and adheres to good software design principles (e.g., SOLID).
* **Readability & Documentation:** Code must be clean, well-commented, and easily understandable by other developers. Include clear inline comments and, where appropriate, generate basic documentation.
* **Performance Awareness:** While speed is important, avoid solutions that introduce performance bottlenecks. Consider algorithmic complexity, efficient data access, and resource usage.

---

### **Part 2: Specific Technical Guidelines & Pitfall Avoidance**

**1. Data & Database Management:**
* **Strict Environment Separation:** Absolutely ensure separate, distinct environments for Development, Staging/Testing, and Production databases from day one. Never use the same database for different environments.
* **Data Integrity & Safety Rails (OWASP A08: Software and Data Integrity Failures):** Do not generate or suggest code that performs destructive operations (e.g., `DROP TABLE`, `DELETE FROM` without `WHERE`) without explicit confirmation and robust safety mechanisms. Ensure integrity checks for software updates, critical data, and CI/CD pipelines.
* **Schema Review:** Any database schema, migrations, or ORM models must be meticulously reviewed for correctness, efficiency, and consistency (e.g., no duplicate tables, correct ID relationships).
* **Data Masking/Sanitization:** For development and testing, use anonymized or synthetic data; never process real production data in non-production environments unless strictly necessary and with explicit approval/masking.

**2. Security Best Practices in Code (Addressing OWASP Top 10 Directly):**
* **Secrets Management (No Hardcoding & OWASP A02: Cryptographic Failures):**
    * **NEVER** hardcode API keys, database credentials, passwords, or any sensitive information directly into the codebase.
    * Always use secure environment variables, dedicated secret management services (e.g., AWS Secrets Manager, Google Secret Manager, HashiCorp Vault), or properly `.gitignore`d `.env` files.
    * When cryptographic functions are involved, ensure strong, modern algorithms are used, and proper key generation, management, and storage practices are followed (e.g., strong hashing for passwords, secure key rotation). Avoid weak or deprecated cryptographic methods.
* **Input Validation & Output Sanitization (OWASP A03: Injection, OWASP A10: Server-Side Request Forgery - SSRF):**
    * All user inputs and external data must be thoroughly validated and outputs must be properly sanitized to prevent common vulnerabilities like SQL Injection, Cross-Site Scripting (XSS), OS Command Injection, and XML External Entities (XXE).
    * For any functionality involving fetching resources from URLs based on user input, implement strict input validation (e.g., whitelist approach for allowed URLs/protocols) to prevent Server-Side Request Forgery (SSRF) attacks, where an attacker could force the server to make requests to internal or unauthorized external systems.
* **Secure Access Control (OWASP A01: Broken Access Control):**
    * Implement granular and explicit access control checks for every sensitive function and resource.
    * Ensure proper authorization mechanisms are in place that prevent authenticated users from accessing or modifying data/functions they are not authorized for (e.g., privilege escalation, horizontal privilege escalation, path traversal).
    * Avoid insecure direct object references (IDOR).
* **Secure Authentication & Identification (OWASP A07: Identification and Authentication Failures):**
    * Implement robust authentication mechanisms. Favor established frameworks over custom solutions.
    * Ensure strong password policies, secure password hashing (e.g., bcrypt), and multi-factor authentication (MFA) where applicable.
    * Properly manage session IDs: ensure they are randomly generated, rotated after successful login, and securely invalidated upon logout or timeout. Avoid exposing session IDs in URLs.
* **Security Misconfiguration (OWASP A05: Security Misconfiguration):**
    * Do not rely on default configurations for any service, framework, or application server. All configurations must be explicitly secured.
    * Minimize exposed services, disable unnecessary features, and remove unused pages/files.
    * Ensure error messages do not reveal sensitive system information.
    * Adhere to security hardening guides for all components (web server, app server, database, containers).
* **Vulnerable and Outdated Components (OWASP A06: Vulnerable and Outdated Components):**
    * Actively manage and monitor all third-party libraries, frameworks, and other components.
    * Ensure AI-suggested dependencies are from trusted sources and check for known vulnerabilities using Software Composition Analysis (SCA) tools.
    * Regularly update components to their latest secure versions. Remove unused or unnecessary components.

**3. Development Workflow & Automation:**
* **Version Control (Git-First):**
    * Assume and leverage Git for version control from the outset.
    * Utilize branching strategies (e.g., Git Flow, GitHub Flow).
    * Emphasize frequent commits with descriptive messages.
    * Advocate for Pull Requests (PRs) and code reviews; no direct pushes to main/master.
    * Properly configure `.gitignore` files to exclude sensitive data, build artifacts, and environment-specific files.
* **Test-Driven Development (TDD):**
    * Prioritize generating tests *before* the functional code.
    * If a test fails, the primary focus is to debug and fix the code, *not* to modify or delete the failing test.
    * Generate comprehensive unit, integration, and (where applicable) end-to-end tests.
* **Continuous Integration/Continuous Deployment (CI/CD):**
    * Suggest and help implement CI/CD pipelines early.
    * Pipelines must include automated linting, **security scanning (SAST/DAST/SCA/secrets detection)**, static code analysis, and comprehensive test execution.
    * Implement staged deployments (e.g., Dev -> Staging -> Production).

**4. Error Handling, Logging, and Observability (OWASP A09: Security Logging and Monitoring Failures):**
* **Robust Error Handling:** Implement comprehensive error handling and graceful degradation. Do not omit try-catch blocks or error checks.
* **Meaningful Logging:** Generate code with appropriate logging (e.g., using a structured logging framework) that provides sufficient context for debugging but **avoids logging sensitive information**.
* **Security Logging:** Crucially, implement comprehensive security logging for all authentication attempts, access control failures, input validation errors, and critical system events.
* **Monitoring & Alerting:** Design for observability. Suggest and integrate tools/code for monitoring application health, performance metrics, API call rates, and resource usage. Implement alerts for critical security events and anomalies. Ensure logs are centrally managed and accessible for security analysis and incident response.
* **Anomaly Detection:** Consider how AI can assist in identifying unusual patterns in logs or metrics that might indicate a subtle bug or security exploit introduced by other AI outputs.

---

### **Part 3: Legal, Ethical, and Cost Considerations**

* **Legal & Privacy Compliance:** Always consider legal and privacy requirements, especially when handling user data (e.g., GDPR, CCPA). Do not generate code that violates these regulations.
* **Ethical Considerations:** Be mindful of the ethical implications of the features being developed.
* **Budgeting:** Remember that the final 20% of a project often takes 90% of the time and budget. AI speeds up boilerplate, but human oversight, testing, and refinement are still significant cost factors.

---

### **Part 4: AI's Role - A Powerful Assistant, Not a Replacement**

* **Augmentation, Not Automation:** Understand that AI is a productivity tool for developers, not a replacement for fundamental software engineering knowledge, critical thinking, or human oversight.
* **Foundational Knowledge Still Crucial:** Database design, security principles, debugging, testing methodologies, and architectural patterns remain essential skills.
* **User Responsibility:** The ultimate responsibility for the code's correctness, security, efficiency, and compliance rests with the human developer.

---

## Common Development Commands

### Security Scanning
```bash
# Run security vulnerability scanning
bandit -r src/

# Check for secrets in code
detect-secrets scan --all-files

# Scan dependencies for vulnerabilities
safety check --json

# Run SAST analysis (if configured)
semgrep --config=auto src/

# Check for outdated dependencies
pip list --outdated
```

### Testing
```bash
# Run all tests
pytest

# Run unit tests only
pytest -m unit

# Run integration tests
pytest -m integration

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run a specific test file
pytest tests/unit/test_inference_engine.py

# Run a specific test
pytest tests/unit/test_inference_engine.py::test_function_name

# Run GPU tests (requires CUDA)
pytest -m gpu
```

### Code Quality
```bash
# Format code with Black (120 char line length)
black .

# Check formatting without making changes
black --check .

# Sort imports
isort .

# Check import sorting
isort --check-only .

# Run linting
flake8 .

# Type checking
mypy src
```

### Running the Application

#### API Server
```bash
# Start GraphQL API server
python main.py api

# Start with debug mode
python main.py api --debug

# Access endpoints:
# GraphQL: http://localhost:8000/graphql
# Health: http://localhost:8000/health
```

#### Training
```bash
# Basic training
python main.py train --model mistralai/Mistral-7B-v0.1 --output-dir ./models/my_model

# Training with MoE and DeepSpeed
python main.py train --model mistralai/Mistral-7B-v0.1 --output-dir ./models/my_model --use-moe --num-experts 8 --deepspeed
```

#### Inference
```bash
# Single inference
python main.py infer --model ./models/my_model --input "Hello, how are you?" --temperature 0.7

# Interactive mode
python main.py interactive --model ./models/my_model --temperature 0.7
```

#### Evaluation
```bash
python main.py evaluate --model ./models/my_model --dataset all --output-dir ./outputs/evaluation
```

### Docker Services
```bash
# Start all services (Redis, MongoDB, etc.)
docker-compose up -d

# Stop all services
docker-compose down

# Check service health
curl http://localhost:8000/health/detailed
```

### Memory Migration
```bash
# Migrate from SQLite to distributed backends (dry run first)
python scripts/migrate_memory.py --source sqlite --target redis,mongodb --dry-run

# Run actual migration
python scripts/migrate_memory.py --source sqlite --target redis,mongodb,supabase,pinecone
```

## High-Level Architecture

### Core Components

1. **Model Layer (`src/model/`)**
   - `moe_model.py`: Mixture of Experts implementation with dynamic expert routing
   - Integrates with Hugging Face Transformers
   - Supports both standard and MoE architectures

2. **Memory System (`src/memory/`)**
   - `memory_manager.py`: Central memory orchestration
   - `enhanced_shadow_memory.py`: Advanced shadow memory with tiering
   - `adapters/`: Backend adapters for Redis, MongoDB, Supabase, Pinecone
   - Multi-tiered storage: hot (Redis) → warm (MongoDB) → cold (Supabase/Pinecone)
   - Automatic memory migration based on access patterns

3. **Training Pipeline (`src/training/`)**
   - `trainer.py`: Multi-phase training with curriculum learning
   - DeepSpeed integration for distributed training
   - Supports foundation, specialized, and advanced training phases

4. **Inference Engine (`src/inference/`)**
   - `inference_engine.py`: Streaming and batch inference
   - Memory-aware response generation
   - Expert routing visualization

5. **API Layer (`src/api/`)**
   - GraphQL API with strawberry framework
   - Health monitoring and metrics
   - Asynchronous request handling

### Key Design Patterns

1. **Distributed Memory Architecture**
   - Redis for active conversation cache (TTL-based)
   - MongoDB for document archival
   - Vector databases (Pinecone/Supabase) for semantic search
   - Automatic tiering and migration

2. **Configuration Management**
   - Central `config.py` with dataclasses
   - Environment variable support via `.env`
   - Validation through `config_validator.py`

3. **Error Handling**
   - Comprehensive error types in `utils/error_handling.py`
   - Retry mechanisms for distributed systems
   - Circuit breakers for external services

4. **Testing Strategy**
   - Unit tests for individual components
   - Integration tests for backend interactions
   - Performance benchmarks for memory operations
   - GPU-specific tests for model operations

### Development Workflow

1. **Pre-commit Hooks**: Automatically run Black, isort, Flake8, and MyPy
2. **CI/CD**: GitHub Actions for multi-platform testing, GPU testing, and security scanning
3. **Documentation**: Sphinx-based API docs in `docs/`
4. **Monitoring**: Health checks and metrics exposed via API

### Important Configuration Files

- `.env`: Service credentials and feature flags
- `config.py`: Application configuration classes
- `configs/deepspeed_config.json`: Distributed training settings
- `docker-compose.yml`: Local service orchestration
- `pyproject.toml`: Python tooling configuration

## CleoAI-Specific Security Considerations

### 1. **Memory System Security**
- **Data Privacy**: User conversations and memories contain sensitive information
  - Implement encryption at rest for all memory backends
  - Use TLS/SSL for all inter-service communication
  - Sanitize PII before logging or debugging
- **Access Control**: Strict user isolation in memory systems
  - Each user's memories must be cryptographically isolated
  - Implement row-level security in databases
  - Use proper authentication for all memory backends

### 2. **Model Security**
- **Prompt Injection Protection**: 
  - Validate and sanitize all user inputs before passing to models
  - Implement rate limiting to prevent abuse
  - Monitor for anomalous prompt patterns
- **Model Weights Protection**:
  - Store model weights securely with access controls
  - Implement checksum verification for model files
  - Use secure channels for model distribution

### 3. **API Security**
- **GraphQL Specific**:
  - Implement query depth limiting to prevent DoS
  - Use query complexity analysis
  - Disable introspection in production
- **Authentication & Authorization**:
  - Implement JWT or OAuth2 for API authentication
  - Use API keys with proper rotation policies
  - Implement fine-grained permissions per endpoint

### 4. **Infrastructure Security**
- **Container Security**:
  - Use minimal base images
  - Scan images for vulnerabilities
  - Run containers as non-root users
- **Secrets Management**:
  - Never commit `.env` files (already in `.gitignore`)
  - Use environment-specific secret management
  - Rotate all credentials regularly
- **Network Security**:
  - Implement network segmentation between services
  - Use private networks for internal communication
  - Expose only necessary ports

### 5. **Compliance & Privacy**
- **Data Retention**:
  - Implement configurable retention policies
  - Provide user data export functionality
  - Ensure complete data deletion capabilities
- **Audit Logging**:
  - Log all data access and modifications
  - Implement tamper-proof audit trails
  - Monitor for suspicious access patterns