#!/bin/bash

# run the docker image
DIR_NOW=$(pwd)

cd ~
echo "current user : ${USER}"

# choose the docker image
echo ""
echo "0  --  cuhk_torch_lab"
echo -n "choose the docker image:"
read image_choose

echo ""
echo -n "input the docker image tag:"
read docker_image_tag

echo ""
echo -n "input the mapping port:"
read docker_image_port

case ${image_choose} in
	0 )
		docker_image="zhaosssss/torch_lab:"
		;;
	* )
		echo "The choice of the docker image is illegal!"
		exit 1 
		;;
esac
echo "The docker image is ${docker_image}${docker_image_tag}"
echo "run docker image..."


docker_final_image="${docker_image}${docker_image_tag}"


# without mapping data disk
#/usr/bin/docker run --runtime=nvidia -it --rm \
#						-v /home/${USER}:/home/${USER} --user=${UID}:${GID} -w ${DIR_NOW} \
#						-v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro \
#						-v /dev/shm:/dev/shm \
#						-p ${docker_image_port}:${docker_image_port} ${docker_final_image} bash

/usr/bin/docker run --runtime=nvidia -itd \
						-v /home/${USER}:/home/${USER} --user=${UID}:${GID} -w ${DIR_NOW} \
						-v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro \
						-v /data/${USER}:/data/${USER} \
						-v /dev/shm:/dev/shm \
						-p ${docker_image_port}:${docker_image_port} ${docker_final_image} bash
