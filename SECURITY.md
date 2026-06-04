# Security Policy

Checkpoint Training Inspector is intended for local, trusted use with your own model files.

## Reporting Security Issues

Please do not open a public issue for a suspected security vulnerability. Instead, contact the repository maintainer privately using the security contact configured on GitHub. If no contact is configured yet, open a minimal issue asking for a private disclosure channel without including exploit details.

## Important Safety Notes

- PyTorch checkpoint files may contain unsafe pickle payloads. Only inspect checkpoints from trusted sources.
- The app first attempts safer `weights_only=True` loading, then falls back for compatibility when required.
- The Gradient Probe feature executes user-provided Python code locally. Treat it like running a script from your terminal.
- Do not upload or commit private checkpoints, patient data, credentials, or proprietary model weights.
