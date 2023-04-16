FROM archlinux_base

COPY install_kde_desktop.sh /install_kde_desktop.sh

RUN chmod +x /install_kde_desktop.sh \
  && /install_kde_desktop.sh
