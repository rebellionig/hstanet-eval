# Jenkins setup (local, Docker) — approx. 15 min

1. Dockerfile for Jenkins with Docker CLI:
```
FROM jenkins/jenkins:lts-jdk17
USER root
RUN apt-get update && apt-get install -y docker.io && rm -rf /var/lib/apt/lists/*
USER jenkins
```
2. `docker build -t jenkins-docker .`
3. `docker run -d --name jenkins -p 8080:8080 -p 50000:50000 -v jenkins_home:/var/jenkins_home -v /var/run/docker.sock:/var/run/docker.sock --group-add $(stat -c %g /var/run/docker.sock) jenkins-docker`
   (Windows/Docker Desktop: drop `--group-add ...`, run the container as root with `-u root` if the socket is denied.)
4. Open http://localhost:8080, unlock with `docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword`, install suggested plugins.
5. Docker Hub: Account Settings → Security → New Access Token. Jenkins → Manage Jenkins → Credentials → Global → Add: Username with password, ID `dockerhub`.
6. New Item → Pipeline → "Pipeline script from SCM" → Git → https://github.com/rebellionig/hstanet-eval → Branch `*/main` → Script Path `Jenkinsfile`.
7. Change `IMAGE_NAME` in the Jenkinsfile if your Docker Hub login differs from `rebellionig`.
8. Build Now. Take screenshots: Stage View, Console Output, Test Result, Docker Hub tags. Then push a small commit to show the automatic trigger.
