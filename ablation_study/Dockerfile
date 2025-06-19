FROM ubuntu:22.04

WORKDIR /project

COPY requirements.txt .

RUN apt-get update && apt-get install -y python3 python3-pip

RUN pip install --no-cache-dir -r requirements.txt

ADD dense_ablation/ dense_ablation/

RUN mkdir experiment/

CMD nvidia-smi;export TERM=xterm && top -b | head; ./dense_ablation/run_pruning_densenet.sh
