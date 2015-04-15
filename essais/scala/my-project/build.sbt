 
lazy val root = (project in file(".")).
  settings(
    name := "my-project",
    version := "1.0",
    scalaVersion := "2.11.4",
    mainClass in Compile := Some("myPackage.MyMainObject")        
  )

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.3.0" % "provided",
  "org.apache.spark" %% "spark-mllib" % "1.3.0" % "provided"
)

// META-INF discarding
mergeStrategy in assembly <<= (mergeStrategy in assembly) { (old) =>
   {
    case PathList("META-INF", xs @ _*) => MergeStrategy.discard
    case x => MergeStrategy.first
   }
}