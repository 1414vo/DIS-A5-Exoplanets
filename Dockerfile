FROM continuumio/miniconda3

WORKDIR ./ivp24

COPY . .

RUN apt-get update && apt-get install -y \
    git vim unzip gcc pkg-config g++ texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super
RUN conda env update --file environment.yml --name base
RUN pre-commit install

RUN conda create -n radvel python=3.9
RUN conda activate radvel
RUN pip install -r requirements.txt
RUN pip install radvel scikit-learn
RUN conda deactivate radvel

EXPOSE 8888
