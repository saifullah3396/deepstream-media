SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd ${SCRIPTPATH}

${SCRIPTPATH}/../../../src/media_analytics_app/media_analytics_app -c ${SCRIPTPATH}/../../../cfg/text_detection/craft/app_config.txt -t
