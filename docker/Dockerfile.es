FROM docker.elastic.co/elasticsearch/elasticsearch:8.13.4
USER root
# Inject pre-indexed data created by build-and-push.ps1
ADD --chown=elasticsearch:elasticsearch es-snapshot/esdata.tar.gz /usr/share/elasticsearch/data/
# Remove lock file left by clean shutdown
RUN rm -f /usr/share/elasticsearch/data/nodes/0/node.lock
USER elasticsearch
