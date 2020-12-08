SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd ${SCRIPTPATH}

${SCRIPTPATH}/../media_analytics_app/media_analytics_app -c ${SCRIPTPATH}/../cfg/facedetectir/facerecognition.txt -t
