DOCKER_COMMANDS = {
    "docker run": "Runs a container from an image",
    "docker ps": "Lists running containers",
    "docker images": "Lists downloaded images"
}

def explain(cmd):
    return DOCKER_COMMANDS.get(cmd, "Unknown Docker command")
