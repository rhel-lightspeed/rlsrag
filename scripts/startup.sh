#!/bin/bash
set -euxo pipefail

if [[ -z "${GENERATE_RAG_DB+x}" ]]; then
  echo "Skipping RAG generation. Set GENERATE_RAG_DB=1 to generate the database and make your computer very warm for an extended period. 😜"
else
  echo "Generating RAG database. This will take time and warm up your cpumputer. 🔥"
  /project/.venv/bin/python rlsrag/embed.py
fi

echo "Starting the API server."
/project/.venv/bin/fastapi run rlsrag/query_api.py
