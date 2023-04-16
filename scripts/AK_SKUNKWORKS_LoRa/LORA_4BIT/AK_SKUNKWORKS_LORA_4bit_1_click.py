import os
import sys
import subprocess
from typing import List

import fire

DOCKER_IMAGE_NAME = "AK_SK_4Q_LORA"


def build_docker_image():
    dockerfile_content = f"""\
FROM python:3.9

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/
WORKDIR /app

ENTRYPOINT ["python", "AKTuner_4Q.py"]
"""

    requirements_content = """\
torch==1.10.0
transformers==4.11.3
datasets==1.14.0
fire==0.4.0
"""

    with open("AK_SK_4Q_LORA.dockerfile", "w") as dockerfile:
        dockerfile.write(dockerfile_content)

    with open("requirements.txt", "w") as requirements:
        requirements.write(requirements_content)

    subprocess.run(["docker", "build", "-t", DOCKER_IMAGE_NAME, "."])

    os.remove("Dockerfile")
    os.remove("requirements.txt")


def run_docker_container(**kwargs):
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    args = []
    for key, value in kwargs.items():
        args.append(f"--{key}={value}")

    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{os.getcwd()}/output:/app/AK_SKUNKWORKS_LoRa/output",
            DOCKER_IMAGE_NAME,
            *args,
        ]
    )


def train_lora_llama(**kwargs):
    build_docker_image()
    run_docker_container(**kwargs)


if __name__ == "__main__":
    fire.Fire(train_lora_llama)
