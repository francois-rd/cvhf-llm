#!/bin/bash

default="default"
use_default=false

if [[ "$*" =~ " -r " && "$*" =~ " -d " ]]
then
  echo "Options '-r <val>' and '-d' are mutually exclusive:"
  echo "'-d' is an alias for '-r $default'"
  exit 1
fi

while getopts ":i:m:r:d" opt; do
  case ${opt} in
    i) implementation="${OPTARG}" ;;
    m) llm="${OPTARG}" ;;
    r) run_id="${OPTARG}" ;;
    d) use_default=true ;;
    :)
      if [ "${OPTARG}" == "r" ]
      then
        echo "Missing runtime ID. Use '-r <val>' to set a value."
        echo "Alternatively, '-d' is an alias for '-r $default'."
      else
        echo "Option -${OPTARG} requires an argument."
      fi
      exit 1
      ;;
    ?)
      echo "Invalid option: -${OPTARG}."
      echo "If this is an option to pass to Coma, use '--' to delineate."
      exit 1
      ;;
  esac
done

shift "$(( OPTIND - 1 ))"

if [ -z "$implementation" ]
then
  echo "Missing LLM implementation. Use the '-i' flag."
  echo "If the flag was given, make sure it appears before any non-flag arguments."
  exit 1
fi
if [ -z "$llm" ]
then
  echo "Missing LLM nickname. Use the '-m' flag."
  echo "If the flag was given, make sure it appears before any non-flag arguments."
  exit 1
fi
if [ -z "$run_id" ]
then
  if [ "$use_default" = true ]
  then
    run_id="$default"
  else
    echo "Missing runtime ID. Use '-r <val>' to set a value or '-d' to use the default."
    echo "If the flag was given, make sure it appears before any non-flag arguments."
    exit 1
  fi
fi

# Remove any lone "--" from $@
for arg
do
  shift
  [ "$arg" = "--" ] && continue
  set -- "$@" "$arg"
done

launch extract llm="$llm" implementation="$implementation" run_id="$run_id" "$@"
