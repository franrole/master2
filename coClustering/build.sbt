 
lazy val root = (project in file(".")).
  settings(
    name := "coClustering",
    version := "1.0",
    scalaVersion := "2.11.4",
    mainClass in Compile := Some("coClustering.MainApplication")        
  )

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.3.0" % "provided",
  "org.apache.spark" %% "spark-mllib" % "1.3.0" % "provided",
  "org.scalatest" %% "scalatest" % "2.2.4" % "test"
)

// META-INF discarding
mergeStrategy in assembly <<= (mergeStrategy in assembly) { (old) =>
   {
    case PathList("META-INF", xs @ _*) => MergeStrategy.discard
    case x => MergeStrategy.first
   }
}

test in assembly := {}