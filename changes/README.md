# Change fragments

Every pull request that changes behaviour, performance, packaging or documentation adds one
small file here. At release time `scripts/assemble_changelog.py` folds the fragments into a
new section of `CHANGELOG.md` and deletes them, so nobody edits `CHANGELOG.md` by hand and
parallel pull requests never conflict on it.

## Rules

* One fragment per pull request. The file name is the branch name with `/` replaced by `-`,
  for example the branch `fix/leiden-resolution` gives `changes/fix-leiden-resolution.md`.
* One to four bullets, each written for a user of the package: what changed and what they
  notice, not how the code works. Wrap long bullets with an indented continuation line.
* A bullet may start with a category tag: `Added:`, `Changed:`, `Fixed:` or `Removed:`.
  Untagged bullets are filed under Changed.
* Only bullets are allowed in a fragment (no headings, no prose).
* Do not mention internal tracking numbers or tools in a bullet.

## Example

```markdown
- Fixed: `leiden` no longer ignores `resolution` when `n_iterations=-1`.
- Changed: the `numpy` requirement is now `>=1.24,<3`.
```

## At release time

```bash
python scripts/assemble_changelog.py --check             # validate every fragment
python scripts/assemble_changelog.py 0.5.0 --dry-run     # preview the section
python scripts/assemble_changelog.py 0.5.0               # write it into CHANGELOG.md, delete the fragments
```
