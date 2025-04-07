#!/bin/bash

while getopts ":m:" opt; do
  case ${opt} in
    m) llm="${OPTARG}" ;;
    :)
      echo "Option -${OPTARG} requires an argument."
      exit 1
      ;;
    ?) ;;  # Ignore other flags. They get based to sub-script.
  esac
done

if [ -z "$llm" ]
then
  echo "Missing LLM nickname. Use the '-m' flag."
  echo "If the flag was given, make sure it appears before any non-flag arguments."
  exit 1
fi

"$(realpath "$(dirname "${BASH_SOURCE[0]}")")"/extract.bash -i HF_TRANSFORMERS "$@" \
  -- --transformers-path transformers_cfgs/"$llm"/transformers.yaml
