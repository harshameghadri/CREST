"""Print the requirements of an optional-dependency group from pyproject.toml, one per line.

Usage: python .github/scripts/print_extra.py test > test-requirements.txt
"""
import sys
import tomllib

with open("pyproject.toml", "rb") as fh:
    project = tomllib.load(fh)["project"]
group = sys.argv[1]
print("\n".join(project["optional-dependencies"][group]))
