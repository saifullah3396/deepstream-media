SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd ${SCRIPTPATH}
${SCRIPTPATH}/../deepstream-app/deepstream-app -c ${SCRIPTPATH}/../cfg/craft/craft.txt -t
