#!/bin/bash

echo $(git diff --name-only --diff-filter=d origin/master...HEAD -- '*.py')