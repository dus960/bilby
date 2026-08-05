FROM igwn/base:conda

LABEL maintainer="Divya Singh"
LABEL description="Custom bilby environment for OSG"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY .git /src/.git
COPY pyproject.toml requirements.txt gw_requirements.txt jax_requirements.txt \
     mcmc_requirements.txt optional_requirements.txt MANIFEST.in README.rst \
     LICENSE.md /src/
COPY bilby /src/bilby
COPY cli_bilby /src/cli_bilby

RUN conda install -c conda-forge numpy==2.3.5 scipy ezdag gwpy lalsuite bilby_pipe beartype rich

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --no-deps -r requirements.txt && \
    pip install --no-cache-dir --no-deps .

RUN rm -rf /src/*

COPY prob_data/* /gw170817_data/

WORKDIR /srv

ENTRYPOINT ["/bin/bash"]
