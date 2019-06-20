FROM tensorflow/tensorflow:2.0.0a0-gpu-py3

ADD docker/debian-requirements.txt debian-requirements.txt
RUN DEBIAN_FRONTEND=noninteractive apt-get update --fix-missing && \
   DEBIAN_PACKAGES=$(egrep -v "^\s*(#|$)" debian-requirements.txt) && \
   apt-get install -y --force-yes $DEBIAN_PACKAGES && \
   apt-get clean && \
   rm -rf /var/lib/apt/lists/* && \
   rm debian-requirements.txt

ADD docker/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt && rm requirements.txt

ADD docker/docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

ENTRYPOINT ["./docker-entrypoint.sh"]