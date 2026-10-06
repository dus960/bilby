FROM igwn/base:conda

LABEL maintainer="Divya Singh"
LABEL description="Custom bilby environment for OSG"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# setuptools_scm reads the version from here instead of .git
ARG BILBY_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${BILBY_VERSION}

USER root
WORKDIR /src

COPY pyproject.toml requirements.txt gw_requirements.txt jax_requirements.txt \
     mcmc_requirements.txt optional_requirements.txt MANIFEST.in README.rst \
     LICENSE.md /src/
COPY bilby /src/bilby
COPY cli_bilby /src/cli_bilby

# conda's bilby 2.8.1 comes in as a bilby_pipe dependency; drop it so only the
# fork is installed. conda clean must be in this RUN or the cache persists.
RUN conda install -y -c conda-forge python=3.13 numpy==2.3.5 scipy ezdag gwpy lalsuite \
        bilby_pipe==1.10.1 beartype rich \
 && conda remove --force -y bilby \
 && conda clean -afy

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --no-deps -r requirements.txt \
 && pip install --no-cache-dir --no-deps .

COPY prob_data/* /prob_data/

WORKDIR /srv
ENTRYPOINT ["/bin/bash"]