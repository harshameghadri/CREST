# Publishing these docs on Read the Docs

A step-by-step guide for a first-time setup. It takes about 10 minutes, and you only do it
once. After that, every push to GitHub rebuilds the site automatically.

**What is already done in the repository:**

* `.readthedocs.yaml` (repository root) tells Read the Docs how to build: Ubuntu 24.04,
  Python 3.12, Sphinx with `docs/conf.py`, packages from `docs/requirements.txt`.
* `docs/conf.py` configures Sphinx. It imports `crest` straight from the source folder and
  replaces ("mocks") the compiled Rust module. Read the Docs therefore never compiles Rust,
  and the build takes about a minute.
* `docs/*.md` are the pages, written in Markdown (MyST).

You only have to connect the repository.

## Step 0: check that the docs build on your machine (optional, 2 minutes)

```bash
cd CREST                                    # your clone
python -m venv .venv-docs && source .venv-docs/bin/activate
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. If it looks right here, it will look the
same on Read the Docs. Warnings in yellow are fine; a red `ERROR` or a traceback is not.

## Step 1: make sure the docs files are on GitHub

Read the Docs builds from GitHub, not from your computer. The files above must be on the
branch you want to publish. They arrive on `dev` when the pull request with this page is
merged. Check on GitHub that `dev` shows `.readthedocs.yaml` in the file list.

## Step 2: create a Read the Docs account

1. Go to <https://app.readthedocs.org/> and click **Sign up**.
2. Choose **Sign up with GitHub** and approve. This lets Read the Docs see your repositories.

## Step 3: import the project

1. On the Read the Docs dashboard click **Add project**.
2. Search for `CREST` and select **harshameghadri/CREST**. If it doesn't appear, click the
   link to configure the Read the Docs GitHub App and give it access to the CREST repository,
   then come back.
3. Name: **crest-sc**. The name becomes the web address, so the docs will be at
   `https://crest-sc.readthedocs.io`. The plain name `crest` may already be taken.
4. Default branch: **dev** for now, because the docs exist only there until the 0.3.0
   release. Change it to `main` after the release (step 6).
5. Click **Next** / **Continue**. Read the Docs finds `.readthedocs.yaml` and starts the
   first build.

## Step 4: watch the first build

Open the project's **Builds** tab. The first build takes 1–2 minutes.

* **Passed** (green): click **View docs**. You're done.
* **Failed** (red): click the build and scroll to the first red line. The common causes are
  below.

| message in the build log | fix |
|---|---|
| `Config file not found` / `.readthedocs.yaml` missing | the branch being built doesn't have the file: check step 1, or set the right branch in **Settings → Default branch** |
| `No module named ...` during `autodoc` | a new module imports a package at top level that isn't in `docs/requirements.txt`: add it there, or to `autodoc_mock_imports` in `docs/conf.py` if it is heavy or compiled |
| `Extension error` / `Could not import extension` | a package in `extensions` of `docs/conf.py` is missing from `docs/requirements.txt` |

## Step 5: turn on previews for pull requests (recommended)

**Settings → Pull request builds → Build pull requests for this project** (tick it, save).
Each pull request then gets a link to a preview of the docs, and a check on GitHub that goes
red if the docs break.

## Step 6: after the 0.3.0 release

1. **Settings → Default branch → `main`**. `latest` then shows the released code.
2. **Versions** tab: activate the tag `v0.3.0`, so that version stays readable forever.
   Activate `dev` too if you want a "development" version of the docs.
3. Optional: add the badge to the top of `README.md`:

   ```markdown
   [![Documentation](https://readthedocs.org/projects/crest-sc/badge/?version=latest)](https://crest-sc.readthedocs.io)
   ```

## Editing the docs later

* Edit any `docs/*.md` file, then commit and push to a branch. The pull-request preview shows
  the result.
* Function documentation comes from the **docstrings** in `crest/*.py`. Improve a docstring
  and the API page updates.
* A new public function also needs a line in `docs/api.md` (`.. autofunction::
  crest.<module>.<name>`) to appear in the reference.
* A new page needs its file name added to a `toctree` block in `docs/index.md`.
