#!/bin/bash

# fail on error
set -e

if [ -n "$(git status --porcelain)" ]; then
  echo "repository is dirty"
  exit 1
fi

if [ $(git symbolic-ref --short -q HEAD) != "main" ]; then
  echo "not on main branch"
  exit 2
fi

# ensure main branch is up-to-date
git pull

# merge main into release branch
git checkout release
git merge --no-ff main --no-edit

# bump patch version (e.g. from 0.1.3 to 0.1.4)
poetry version patch

# commit change
git add pyproject.toml
git commit -m "Bump version"

# create tag and push
new_tag=v$(poetry version -s)
echo New tag: $new_tag
git tag $new_tag
git push origin release $new_tag

# clean previous build and build
echo "clean up old builds"
rm -rf build dist
echo "do new build"
poetry build
echo "publish package"
# to use this, set up an API token with `poetry config pypi-token.pypi <api token>`
poetry publish

# clean up
echo "go back to main branch"
git checkout main
