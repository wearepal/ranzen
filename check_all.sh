#!/bin/bash

echo "Begin check..." \
&& black . \
&& python -m pytest -vv tests/ \
&& pyright ranzen \
&& pyright tests \
&& echo "Check all complete!"
