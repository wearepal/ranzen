#!/bin/bash

echo "Begin check..." \
&& black . \
&& python -m pytest -vv tests/ \
&& pyright mantra \
&& pyright tests \
&& echo "Check all complete!"
