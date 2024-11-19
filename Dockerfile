FROM registry.access.redhat.com/ubi9/python-312:latest AS builder
WORKDIR /project

RUN pip3 install --upgrade --quiet pip pdm
ENV PDM_CHECK_UPDATE=false

COPY pyproject.toml pdm.lock README.md /project/
COPY rlsrag/ /project/rlsrag

RUN pdm install --check --prod --no-editable

FROM registry.access.redhat.com/ubi9/python-312:latest
WORKDIR /project
COPY --from=builder /project/.venv/ /project/.venv
COPY plaintext/ /plaintext/
ENV PATH="/project/.venv/bin:$PATH"
COPY rlsrag /project/rlsrag

COPY scripts/startup.sh /project/
CMD ["/project/startup.sh"]
