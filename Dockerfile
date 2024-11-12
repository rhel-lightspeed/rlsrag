FROM registry.access.redhat.com/ubi9/python-312:latest AS builder
WORKDIR /project

RUN pip3 install --upgrade --quiet pip pdm
ENV PDM_CHECK_UPDATE=false

COPY pyproject.toml pdm.lock README.md /project/
COPY rlsrag/ /project/rlsrag
COPY packages/ /project/packages
RUN ls -al .
RUN pdm install --check --prod --no-editable

RUN pdm run download-model

FROM registry.access.redhat.com/ubi9/python-312:latest
WORKDIR /project
COPY --from=builder /project/.venv/ /project/.venv
COPY --from=builder /project/embedding_model/ /project/embedding_model
ENV PATH="/project/.venv/bin:$PATH"
COPY rlsrag /project/rlsrag
CMD ["python", "rlsrag/quicktest.py"]
