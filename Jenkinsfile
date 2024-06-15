pipeline {
    agent any

    environment {
        GITHUB_CREDENTIALS = credentials('GithubJenkins') // ID de las credenciales de GitHub en Jenkins
        SONARQUBE_TOKEN = '7dc05c423cb25fea83663025c9629ccbbce9107f' // Token de SonarQube
    }

    stages {
        stage('Clone repository') {
            steps {
                git url: 'https://github.com/Donpirro/ProyectoDocker.git', credentialsId: 'GithubJenkins'
            }
        }
        
        stage('SonarQube Analysis') {
            environment {
                scannerHome = tool 'SonarQube Scanner'
            }
            steps {
                sh "${scannerHome}/bin/sonar-scanner -Dsonar.projectKey=mi-proyecto -Dsonar.sources=. -Dsonar.host.url=http://localhost:9000 -Dsonar.login=${SONARQUBE_TOKEN}"
            }
        }
        
        stage("Quality Gate") {
            steps {
                script {
                    timeout(time: 1, unit: 'HOURS') {
                        waitForQualityGate abortPipeline: true
                    }
                }
            }
        }
    }
}
