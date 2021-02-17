#!/bin/bash

echo "Begin check..." \
&& black . \
&& python -m pytest -vv tests/ \
&& mypy tests \
&& echo "Check all complete!"
