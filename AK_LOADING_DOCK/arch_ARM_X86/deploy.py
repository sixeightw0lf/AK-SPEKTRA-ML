import os
import sys

def build_images():
    os.system("docker build -t archlinux_base -f Dockerfile_base .")
    os.system("docker-compose build")

def deploy_locally():
    os.system("docker-compose up -d")

def main():
    if len(sys.argv) != 3:
        print("Usage: python deploy.py [local|cloud] <path_to_project_root>")
        sys.exit(1)

    deployment_type = sys.argv[1]
    project_root = sys.argv[2]

    os.chdir(project_root)

    build_images()

    if deployment_type == "local":
        deploy_locally()
    elif deployment_type == "cloud":
        print("Cloud deployment not implemented.")
    else:
        print("Invalid deployment type. Use 'local' or 'cloud'.")

if __name__ == "__main__":
    main()
