#!/bin/bash
# Builds closure-check-overlay-v1.jar: a driver-only replacement of Spark 4.0.2's
# org.apache.spark.util.SparkClosureCleaner$ that skips the eager closure
# serializability check (re-enable with -Dspark.experimental.closureSerializableCheck=true).
# Requires JDK 17 and the Spark 4.0.2 / Scala 2.13.16 jars (from SPARK_HOME/jars or ~/.m2).
set -euo pipefail
cd "$(dirname "$0")"
JARS=${SPARK_HOME:?set SPARK_HOME to a Spark 4.0.2 install}/jars
CP="$JARS/spark-core_2.13-4.0.2.jar:$JARS/spark-common-utils_2.13-4.0.2.jar:$JARS/scala-library-2.13.16.jar"
rm -rf classes && mkdir classes
javac --release 17 -cp "$CP" -d classes 'SparkClosureCleaner$.java'
jar cf closure-check-overlay-v1.jar -C classes .
echo "built $(pwd)/closure-check-overlay-v1.jar"
echo "use: spark.driver.extraClassPath=$(pwd)/closure-check-overlay-v1.jar:<plugin jar>"
