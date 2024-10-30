#!/bin/bash

echo $(git diff --name-only origin/master...HEAD -- '*.py')