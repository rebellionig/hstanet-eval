// Declarative Jenkins pipeline for hstanet-eval.
// Requirements on the Jenkins node: Docker CLI + access to the Docker daemon,
// plugins: Pipeline, Git, JUnit, Credentials Binding.
// Credentials: "dockerhub" (Username with password) -> only needed for the push stage.

pipeline {
    agent any

    options {
        timestamps()
        timeout(time: 20, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '20'))
        disableConcurrentBuilds()
    }

    environment {
        IMAGE_NAME = 'armansalken/hstanet-eval'      // Docker Hub repository
        IMAGE_TAG  = "${env.BUILD_NUMBER}"
    }

    triggers {
        pollSCM('H/2 * * * *')                       // or use a GitHub webhook
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                sh 'git log -1 --oneline'
            }
        }

        stage('Unit tests') {
            steps {
                sh 'docker build --target test -t hstanet-eval:test .'
                sh '''
                    docker rm -f hstanet-eval-tests || true
                    docker run --name hstanet-eval-tests hstanet-eval:test \
                        pytest --junitxml=/app/junit.xml --cov=hstanet_eval --cov-fail-under=90
                '''
            }
            post {
                always {
                    sh 'docker cp hstanet-eval-tests:/app/junit.xml junit.xml || true'
                    sh 'docker rm -f hstanet-eval-tests || true'
                    junit allowEmptyResults: true, testResults: 'junit.xml'
                }
            }
        }

        stage('Build image') {
            steps {
                sh "docker build --target runtime -t ${IMAGE_NAME}:${IMAGE_TAG} -t ${IMAGE_NAME}:latest ."
            }
        }

        stage('Smoke test image') {
            steps {
                sh """
                    docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} metrics \
                        --pred examples/predictions.json --gt examples/ground_truth.json --iou 0.5 0.75
                    docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} latency \
                        --target hstanet_eval.demo:dummy_model --shape 4,3,64,64 --runs 10
                """
            }
        }

        stage('Push to Docker Hub') {
            when { expression { env.GIT_BRANCH in ['origin/main', 'main'] } }   // optional stage
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub',
                        usernameVariable: 'DH_USER', passwordVariable: 'DH_PASS')]) {
                    sh 'echo "$DH_PASS" | docker login -u "$DH_USER" --password-stdin'
                    sh "docker push ${IMAGE_NAME}:${IMAGE_TAG}"
                    sh "docker push ${IMAGE_NAME}:latest"
                }
            }
            post { always { sh 'docker logout || true' } }
        }
    }

    post {
        success { echo "Pipeline OK: ${IMAGE_NAME}:${IMAGE_TAG}" }
        failure { echo 'Pipeline FAILED - see the stage logs above.' }
        cleanup { sh 'docker image prune -f || true' }
    }
}
