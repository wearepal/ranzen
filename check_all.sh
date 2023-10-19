#!/bin/bash

echo "Begin check..." \
&& black ranzen \
&& black tests \
&& python -m pytest -vv tests/ \
&& pydoclint ranzen \
&& pyright ranzen \
&& pyright tests \
&& echo "Check all complete!"
