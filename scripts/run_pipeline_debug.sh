SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd ${SCRIPTPATH}

/usr/bin/gdb --args ${SCRIPTPATH}/../src/media_analytics_app/media_analytics_app -c ${SCRIPTPATH}/../cfg/full_pipeline/app_config.txt -t
