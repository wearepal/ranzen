#!/bin/bash

# fail on error
set -e

# confirm the supplied version bump is valid
version_bump=$1

case $version_bump in
  "patch" | "minor" | "major" | "prepatch" | "preminor" | "premajor" | "prerelease")
    echo "valid version bump: $version_bump"
    ;;
  *)
    echo "invalid version bump: \"$version_bump\""
    echo "Usage: bash make_release.sh <version bump>"
    exit 1
    ;;
esac

if [ -n "$(git status --untracked-files=no --porcelain)" ]; then
  echo "repository is dirty"
  exit 1
fi

if [ $(git symbolic-ref --short -q HEAD) != "main" ]; then
  echo "not on main branch"
  exit 2
fi

echo ensure main branch is up-to-date
git pull

echo checkout release branch
git checkout release
echo ensure release branch is up-to-date
git pull
echo merge main into release branch
git merge --no-ff main --no-edit

# bump version
poetry version $version_bump

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
