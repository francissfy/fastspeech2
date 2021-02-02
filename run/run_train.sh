conda activate pyenv
echo "using conda pyenv environment"

MODEL_CONFIG="./param/config.yaml"
MEL_SCP=""
VARIANCE_SCP=""
JSON_DATA=""

repo_root=""
cd "${repo_root}" || exit 1

python3 run/fastspeech_train.py --model_config "${MODEL_CONFIG}" \
                                --mel_scp "${MEL_SCP}" \
                                --variance_scp "${VARIANCE_SCP}" \
                                --json_data "${JSON_DATA}"

