

build:
	docker build --network host -t pose_laura -f docker/Dockerfile .

run:
	docker-compose -f docker/docker-compose.yml up -d
	docker exec -it docker_pose_laura_1 bash

down:
	docker stop docker_pose_laura_1
	docker rm docker_pose_laura_1
