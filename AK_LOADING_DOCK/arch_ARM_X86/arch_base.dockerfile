FROM archlinux:base

RUN pacman -Syu --noconfirm \
  && pacman -S --noconfirm base-devel git \
  && pacman -Scc --noconfirm
