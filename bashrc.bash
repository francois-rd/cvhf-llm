#!/bin/bash

# Find the project root directory assuming this script file lives directly inside it.
PROJECT_ROOT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
export PROJECT_ROOT_DIR

# Add main and plugin code to PYTHONPATH.
export PYTHONPATH="$PROJECT_ROOT_DIR"/src:"$PROJECT_ROOT_DIR"/plugins

# Environment variables for launching without commands and configs.
export DEFAULT_CONFIG_DIR="$PROJECT_ROOT_DIR"/launch
export DEFAULT_COMMAND="test.launch"

# Alias for program entry.
launch () {
  pushd "$DEFAULT_CONFIG_DIR" > /dev/null || exit
  python "$PROJECT_ROOT_DIR"/src/main.py "$@"
  popd > /dev/null || exit
}
export -f launch

# Basic terminal auto-complete.
complete -W "
analyze.histogram
consolidate
docx.to.json
extract
mini.validation
parse.labels
segment
test.comp
test.launch
" launch
