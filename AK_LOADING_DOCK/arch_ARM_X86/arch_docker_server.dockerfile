FROM archlinux_base

COPY install_docker_server.sh /install_docker_server.sh

RUN chmod +x /install_docker_server.sh \
  && /install_docker_server.sh
