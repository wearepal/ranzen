#!/bin/bash

echo "Checking docstrings with darglint..." \
&& darglint --docstring-style sphinx -v 2 -z long ranzen \
&& echo "Check complete!"
