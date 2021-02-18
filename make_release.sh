#!/bin/bash

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

# clean up
git checkout main
