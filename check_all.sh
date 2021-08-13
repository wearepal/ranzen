#!/bin/bash

echo "Begin check..." \
&& black . \
&& python -m pytest -vv tests/ \
&& pyright kit \
&& pyright tests \
&& echo "Check all complete!"
