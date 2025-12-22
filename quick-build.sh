#!/bin/bash
#
# Quick incremental build script for spark-rapids development
#
# Usage: ./quick-build.sh [options]
#
# Options:
#   -m, --module MODULE    Module to build (default: sql-plugin)
#   -v, --ver VERSION      Spark version (default: 320)
#   -d, --dist             Also build dist jar (slow, skip by default)
#   -t, --tests            Also build tests module
#   -i, --install          Install to local repo
#   -h, --help             Show this help
#
# Examples:
#   ./quick-build.sh                      # Quick compile sql-plugin only (~30s)
#   ./quick-build.sh -t                   # Compile sql-plugin + tests (~45s)
#   ./quick-build.sh -d                   # Compile with dist jar (~2min)
#   ./quick-build.sh -m aggregator        # Only build aggregator
#   ./quick-build.sh -v 341               # Build for Spark 3.4.1
#

set -e

MODULE="sql-plugin"
BUILDVER="320"
BUILD_DIST=false
BUILD_TESTS=false
INSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--module) MODULE="$2"; shift 2 ;;
        -v|--ver) BUILDVER="$2"; shift 2 ;;
        -d|--dist) BUILD_DIST=true; shift ;;
        -t|--tests) BUILD_TESTS=true; shift ;;
        -i|--install) INSTALL=true; shift ;;
        -h|--help)
            head -20 "$0" | tail -18
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Build target modules
MODULES="$MODULE"
if [[ "$BUILD_TESTS" == "true" ]]; then
    MODULES="$MODULES,tests"
fi

# Goal
if [[ "$INSTALL" == "true" ]]; then
    GOAL="install"
else
    GOAL="compile"
fi

COMMON_OPTS="-Dbuildver=$BUILDVER -DskipTests -Dmaven.scaladoc.skip -Dmaven.scalastyle.skip=true -Drat.skip=true -T 4"

echo "🚀 Quick building: $MODULES for Spark $BUILDVER"
START_TIME=$(date +%s)

# Step 1: Compile modules
mvn $GOAL -pl "$MODULES" -am $COMMON_OPTS -q

# Step 2: Build dist if requested
if [[ "$BUILD_DIST" == "true" ]]; then
    echo "📦 Building dist jar (single version: $BUILDVER)..."
    mvn package -rf dist -Dincluded_buildvers=$BUILDVER $COMMON_OPTS -q
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "✅ Build completed in ${ELAPSED}s"

# Show output locations
echo ""
echo "📁 Output jars:"
if [[ "$BUILD_DIST" == "true" ]]; then
    ls -lh dist/target/rapids-4-spark_2.12-*-cuda12.jar 2>/dev/null || true
else
    echo "   aggregator/target/spark$BUILDVER/rapids-4-spark-aggregator_2.12-*-spark$BUILDVER.jar"
    echo "   (Use -d flag to build dist jar)"
fi
