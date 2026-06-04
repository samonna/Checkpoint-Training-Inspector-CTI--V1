# Contributing to Checkpoint Training Inspector

Thanks for taking the time to improve CTI. This document covers how to set up a local development environment and what to include in a pull request.

---

## Table of Contents

- [Local Setup](#local-setup)
- [Running the App](#running-the-app)
- [Code Style](#code-style)
- [Before Opening a Pull Request](#before-opening-a-pull-request)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)
- [Security Issues](#security-issues)

---

## Local Setup

**Requires Python 3.10+**

```bash
git clone https://github.com/YOUR-USERNAME/checkpoint-training-inspector.git
cd checkpoint-training-inspector
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the App

```bash
streamlit run app.py
```

Generate demo checkpoints if you want test data:

```bash
python train_demo.py --out_dir sample_training_checkpoints --epochs 5
```

---

## Code Style

- Follow the existing style in each file — no formal linter is enforced yet.
- Keep functions focused and small.
- Avoid adding new dependencies unless necessary; prefer what is already in `requirements.txt`.
- Do not commit real model checkpoints, patient data, or credentials.

---

## Before Opening a Pull Request

Run a syntax check to make sure nothing is broken:

```bash
python -m compileall app.py train_demo.py checkpoint_inspector
```

If your change touches checkpoint parsing, layer metrics, anomaly detection, report generation, or any UI mode, manually test that mode in the app before submitting.

Include a short note in your PR describing:

- What checkpoint layout you tested (raw `state_dict`, nested dict, DDP, etc.).
- Which app mode(s) you exercised.
- Any memory or performance considerations for large models.

---

## Pull Request Guidelines

- Keep changes focused on one concern per PR.
- Do not commit `.pth`, `.pt`, `.bin`, or other large binary files.
- Use the provided [pull request template](.github/pull_request_template.md).
- Document new user-facing behavior in `README.md`.
- Mention any safety implications, especially for checkpoint loading or Gradient Probe execution.

---

## Reporting Bugs

Open an issue using the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).

Please include:
- Your OS, Python version, PyTorch version, and Streamlit version.
- The checkpoint file extension and layout if known.
- Steps to reproduce with the smallest example you can share.

Do **not** attach private checkpoints or sensitive training data to issues.

---

## Suggesting Features

Open an issue using the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md).

Good feature requests describe a specific workflow that is hard or missing today, not just a general idea.

---

## Security Issues

Please do **not** open a public issue for security vulnerabilities. See [SECURITY.md](SECURITY.md) for the disclosure process.
