
Pour créer le jar :

sbt assembly

Pour exécuter :

./bin/spark-submit --class myPackage.MyMainObject --master local[4] ../essais/scala/my-project/target/scala-2.11/my-project-assembly-1.0.jar 

NB : vérifier le chemin du dossier "data" et du jar par rapport au dossier courant.