#!/bin/bash
set -euxo pipefail

if [[ -z "${GENERATE_RAG_DB+x}" ]]; then
  echo "Skipping RAG generation. Set GENERATE_RAG_DB=1 to generate the database."
else
  echo "Generating RAG database. This will take time. 🔥"
  /project/.venv/bin/python rlsrag/embed.py
fi

echo "Starting the API server."
/project/.venv/bin/fastapi run rlsrag/query_api.py
