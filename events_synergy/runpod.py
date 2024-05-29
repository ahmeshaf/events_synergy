from runpod import create_pod
from typer import Typer
from typing import Optional

app = Typer()


@app.command()
def main(
    name: str,
    image_name: Optional[str] = "runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04",
    gpu_type_id: Optional[str] = "NVIDIA A40",
    gpu_count: Optional[int] = 1,
    volume_in_gb: Optional[int] = 20,
    container_disk_in_gb: Optional[int] = 20,
    ports: Optional[str] = None,
    volume_mount_path: Optional[str] = "/runpod-volume",
    template_id: Optional[str] = None,
    network_volume_id: Optional[str] = None,
) -> None:
    """
    Create a pod

    :param name: the name of the pod
    :param image_name: the name of the docker image to be used by the pod
    :param gpu_type_id: the gpu type wanted by the pod (retrievable by get_gpus)
    :param cloud_type: if secure cloud, community cloud or all is wanted
    :param data_center_id: the id of the data center
    :param country_code: the code for country to start the pod in
    :param gpu_count: how many gpus should be attached to the pod
    :param volume_in_gb: how big should the pod volume be
    :param ports: the ports to open in the pod, example format - "8888/http,666/tcp"
    :param volume_mount_path: where to mount the volume?
    :param env: the environment variables to inject into the pod,
                for example {EXAMPLE_VAR:"example_value", EXAMPLE_VAR2:"example_value 2"}, will
                inject EXAMPLE_VAR and EXAMPLE_VAR2 into the pod with the mentioned values
    :param template_id: the id of the template to use for the pod

    """
    new_pod = create_pod(
        name,
        image_name,
        gpu_type_id,
        gpu_count=gpu_count,
        volume_in_gb=volume_in_gb,
        container_disk_in_gb=container_disk_in_gb,
        ports=ports,
        volume_mount_path=volume_mount_path,
        template_id=template_id,
        network_volume_id=network_volume_id,
    )

    print(f"Pod {new_pod['id']} has been created.")


if __name__ == "__main__":
    app()
