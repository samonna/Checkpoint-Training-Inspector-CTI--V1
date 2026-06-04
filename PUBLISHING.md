# Publishing to GitHub

Use these steps when you are ready to publish Checkpoint Training Inspector as its own GitHub repository.

## 1. Create a Repository on GitHub

Create a new empty repository named something like:

```text
checkpoint-training-inspector
```

Recommended repository description:

```text
Offline Streamlit dashboard for inspecting PyTorch checkpoints, layer drift, anomalies, training trajectories, and representation changes.
```

Recommended topics:

```text
pytorch, streamlit, deep-learning, model-checkpoints, machine-learning, diagnostics, training-analysis, anomaly-detection
```

## 2. Initialize Git in This Folder

Run these commands from the CTI project folder:

```bash
git init
git add .
git commit -m "Initial release"
git branch -M main
```

## 3. Connect Your GitHub Repository

Replace `YOUR-USERNAME` with your GitHub username:

```bash
git remote add origin https://github.com/YOUR-USERNAME/checkpoint-training-inspector.git
git push -u origin main
```

## 4. GitHub Settings

After the first push:

- Add the repository description above.
- Add the suggested topics.
- Confirm `README.md`, `LICENSE`, `SECURITY.md`, and `CONTRIBUTING.md` show correctly.
- Consider adding screenshots to the README after launching the app locally.

## 5. What Not to Commit

Do not commit:

- Real model checkpoints.
- Private datasets or patient data.
- Generated report exports.
- `.streamlit/secrets.toml`.
- Virtual environments.

The `.gitignore` file already excludes the common versions of these files.
