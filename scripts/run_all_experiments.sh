#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Uso:
  scripts/run_all_experiments.sh [mode]

Modes:
  all          Roda DAG opcional, estimadores principais, bayesiano, comparacoes e E-value.
  estimators   Roda g-computation, AIPW XGB par-a-par e AIPW XGB 3 classes.
  bayesian     Roda as tres especificacoes bayesianas e compara priors.
  compare      Roda comparacao de metodos usando as pastas desta execucao.
  sensitivity  Roda E-value usando as pastas desta execucao.
  dag          Roda validacao/refutabilidade do DAG.

Variaveis uteis:
  RUN_ID                      Identificador da execucao. Default: timestamp.
  OUT_ROOT                    Raiz das saidas. Default: causal/output/final_runs/$RUN_ID
  PYTHON                      Interpretador Python. Default: python3
  RUN_DAG                     Em mode=all, 1 roda DAG; 0 pula. Default: 0
  DAG_N_BOOTSTRAPS            Default: 150
  DAG_SAMPLE_SIZE             Default: 10000
  BOOTSTRAP_ITERATIONS        Default: 300
  BOOTSTRAP_SAMPLE_SIZE       Default: 100000
  BOOTSTRAP_EVALUATION_SIZE   Default: 100000
  XGB_PAIR_CROSSFIT_FOLDS     Default: 3
  XGB_3CLASS_CROSSFIT_FOLDS   Default: 2
  BAYESIAN_SVI_STEPS          Default: 3000
  BAYESIAN_POSTERIOR_SAMPLES  Default: 1000
EOF
}

MODE="${1:-all}"
if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-python3}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-causal/output/final_runs/${RUN_ID}}"
LOG_DIR="${OUT_ROOT}/logs"
MANIFEST="${OUT_ROOT}/run_manifest.txt"

RUN_DAG="${RUN_DAG:-0}"
DAG_N_BOOTSTRAPS="${DAG_N_BOOTSTRAPS:-150}"
DAG_SAMPLE_SIZE="${DAG_SAMPLE_SIZE:-10000}"

BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-300}"
BOOTSTRAP_SAMPLE_SIZE="${BOOTSTRAP_SAMPLE_SIZE:-100000}"
BOOTSTRAP_EVALUATION_SIZE="${BOOTSTRAP_EVALUATION_SIZE:-100000}"
XGB_PAIR_CROSSFIT_FOLDS="${XGB_PAIR_CROSSFIT_FOLDS:-3}"
XGB_3CLASS_CROSSFIT_FOLDS="${XGB_3CLASS_CROSSFIT_FOLDS:-2}"

BAYESIAN_SVI_STEPS="${BAYESIAN_SVI_STEPS:-3000}"
BAYESIAN_POSTERIOR_SAMPLES="${BAYESIAN_POSTERIOR_SAMPLES:-1000}"
BAYESIAN_LR="${BAYESIAN_LR:-0.02}"

mkdir -p "${LOG_DIR}"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${MANIFEST}"
}

run_step() {
  local name="$1"
  shift
  local logfile="${LOG_DIR}/${name}.log"
  log "START ${name}"
  {
    printf '[command]'
    printf ' %q' "$@"
    printf '\n\n'
    "$@"
  } 2>&1 | tee "${logfile}"
  log "END ${name} -> ${logfile}"
}

write_manifest_header() {
  local created_at
  created_at="$(date --iso-8601=seconds)"
  {
    if [[ -s "${MANIFEST}" ]]; then
      echo
      echo "================================================================"
      echo
    fi
    echo "run_invocation_id=${RUN_ID}"
    echo "project_root=${PROJECT_ROOT}"
    echo "out_root=${OUT_ROOT}"
    echo "mode=${MODE}"
    echo "python=$(${PYTHON} --version 2>&1)"
    echo "date=${created_at}"
    echo
    echo "bootstrap_iterations=${BOOTSTRAP_ITERATIONS}"
    echo "bootstrap_sample_size=${BOOTSTRAP_SAMPLE_SIZE}"
    echo "bootstrap_evaluation_size=${BOOTSTRAP_EVALUATION_SIZE}"
    echo "xgb_pair_crossfit_folds=${XGB_PAIR_CROSSFIT_FOLDS}"
    echo "xgb_3class_crossfit_folds=${XGB_3CLASS_CROSSFIT_FOLDS}"
    echo "bayesian_svi_steps=${BAYESIAN_SVI_STEPS}"
    echo "bayesian_posterior_samples=${BAYESIAN_POSTERIOR_SAMPLES}"
    echo
  } >> "${MANIFEST}"
}

ensure_code_compiles() {
  run_step "00_py_compile" "${PYTHON}" -m py_compile causal/*.py
}

run_dag_checks() {
  run_step "01_dag_checks" \
    "${PYTHON}" causal/test_final_dag.py \
      --n-bootstraps "${DAG_N_BOOTSTRAPS}" \
      --sample-size "${DAG_SAMPLE_SIZE}" \
      --output-dir "${OUT_ROOT}/dag_checks"
}

run_estimators() {
  run_step "10_gcomp" \
    "${PYTHON}" -u causal/estimate_education_effect.py \
      --output-dir "${OUT_ROOT}/gcomp"

  run_step "20_aipw_xgb_pair_crossfit${XGB_PAIR_CROSSFIT_FOLDS}_bootstrap${BOOTSTRAP_ITERATIONS}" \
    "${PYTHON}" -u causal/estimate_education_effect_aipw.py \
      --model-type xgb \
      --crossfit-folds "${XGB_PAIR_CROSSFIT_FOLDS}" \
      --bootstrap-iterations "${BOOTSTRAP_ITERATIONS}" \
      --bootstrap-sample-size "${BOOTSTRAP_SAMPLE_SIZE}" \
      --bootstrap-evaluation-sample-size "${BOOTSTRAP_EVALUATION_SIZE}" \
      --gcomp-results "${OUT_ROOT}/gcomp/effect_estimates_common_support.csv" \
      --output-dir "${OUT_ROOT}/aipw_xgb_pair_crossfit${XGB_PAIR_CROSSFIT_FOLDS}_bootstrap${BOOTSTRAP_ITERATIONS}"

  run_step "30_aipw_xgb_3class_bootstrap${BOOTSTRAP_ITERATIONS}" \
    "${PYTHON}" -u causal/estimate_education_effect_aipw_xgb_3class.py \
      --crossfit-folds 1 \
      --bootstrap-iterations "${BOOTSTRAP_ITERATIONS}" \
      --bootstrap-sample-size "${BOOTSTRAP_SAMPLE_SIZE}" \
      --bootstrap-evaluation-sample-size "${BOOTSTRAP_EVALUATION_SIZE}" \
      --output-dir "${OUT_ROOT}/aipw_xgb_3class_bootstrap${BOOTSTRAP_ITERATIONS}"

  run_step "31_aipw_xgb_3class_crossfit${XGB_3CLASS_CROSSFIT_FOLDS}_point" \
    "${PYTHON}" -u causal/estimate_education_effect_aipw_xgb_3class.py \
      --crossfit-folds "${XGB_3CLASS_CROSSFIT_FOLDS}" \
      --bootstrap-iterations 0 \
      --output-dir "${OUT_ROOT}/aipw_xgb_3class_crossfit${XGB_3CLASS_CROSSFIT_FOLDS}_point"
}

run_bayesian() {
  run_step "40_bayesian_conservative" \
    "${PYTHON}" -u causal/estimate_education_effect_bayesian.py \
      --svi-steps "${BAYESIAN_SVI_STEPS}" \
      --posterior-samples "${BAYESIAN_POSTERIOR_SAMPLES}" \
      --learning-rate "${BAYESIAN_LR}" \
      --intercept-prior-scale 1.0 \
      --treatment-prior-scale 0.5 \
      --covariate-prior-scale 0.5 \
      --output-dir "${OUT_ROOT}/bayesian_conservative"

  run_step "41_bayesian_default" \
    "${PYTHON}" -u causal/estimate_education_effect_bayesian.py \
      --svi-steps "${BAYESIAN_SVI_STEPS}" \
      --posterior-samples "${BAYESIAN_POSTERIOR_SAMPLES}" \
      --learning-rate "${BAYESIAN_LR}" \
      --intercept-prior-scale 1.5 \
      --treatment-prior-scale 1.0 \
      --covariate-prior-scale 1.0 \
      --output-dir "${OUT_ROOT}/bayesian_default"

  run_step "42_bayesian_weak" \
    "${PYTHON}" -u causal/estimate_education_effect_bayesian.py \
      --svi-steps "${BAYESIAN_SVI_STEPS}" \
      --posterior-samples "${BAYESIAN_POSTERIOR_SAMPLES}" \
      --learning-rate "${BAYESIAN_LR}" \
      --intercept-prior-scale 2.0 \
      --treatment-prior-scale 1.5 \
      --covariate-prior-scale 1.5 \
      --output-dir "${OUT_ROOT}/bayesian_weak"

  run_step "43_compare_bayesian_sensitivity" \
    "${PYTHON}" causal/compare_bayesian_sensitivity.py \
      --output-dir "${OUT_ROOT}/bayesian_sensitivity" \
      --run "conservative=${OUT_ROOT}/bayesian_conservative" \
      --run "default=${OUT_ROOT}/bayesian_default" \
      --run "weak=${OUT_ROOT}/bayesian_weak"
}

run_compare() {
  run_step "50_compare_methods" \
    "${PYTHON}" causal/compare_education_methods.py \
      --output-dir "${OUT_ROOT}/method_comparison" \
      --method-result "gcomp=${OUT_ROOT}/gcomp/effect_estimates_common_support.csv" \
      --method-result "aipw_xgb_pair_crossfit3_bootstrap300=${OUT_ROOT}/aipw_xgb_pair_crossfit${XGB_PAIR_CROSSFIT_FOLDS}_bootstrap${BOOTSTRAP_ITERATIONS}/aipw_bootstrap_summary_common_support.csv" \
      --method-result "aipw_xgb_3class_bootstrap300=${OUT_ROOT}/aipw_xgb_3class_bootstrap${BOOTSTRAP_ITERATIONS}/aipw_3class_bootstrap_summary_global_support.csv"
}

run_sensitivity() {
  run_step "60_evalue_sensitivity" \
    "${PYTHON}" causal/sensitivity_education_effect.py \
      --output-dir "${OUT_ROOT}/evalue_sensitivity" \
      --result "gcomp=${OUT_ROOT}/gcomp/effect_estimates_common_support.csv" \
      --result "aipw_xgb_pair_crossfit3_bootstrap300=${OUT_ROOT}/aipw_xgb_pair_crossfit${XGB_PAIR_CROSSFIT_FOLDS}_bootstrap${BOOTSTRAP_ITERATIONS}/aipw_bootstrap_summary_common_support.csv" \
      --result "aipw_xgb_3class=${OUT_ROOT}/aipw_xgb_3class_bootstrap${BOOTSTRAP_ITERATIONS}/aipw_3class_bootstrap_summary_global_support.csv" \
      --result "bayesian_conservative=${OUT_ROOT}/bayesian_conservative/bayesian_posterior_effects_summary.csv" \
      --result "bayesian_default=${OUT_ROOT}/bayesian_default/bayesian_posterior_effects_summary.csv" \
      --result "bayesian_weak=${OUT_ROOT}/bayesian_weak/bayesian_posterior_effects_summary.csv"
}

write_manifest_header
log "Output root: ${OUT_ROOT}"

case "${MODE}" in
  all)
    ensure_code_compiles
    if [[ "${RUN_DAG}" == "1" ]]; then
      run_dag_checks
    else
      log "SKIP dag_checks (RUN_DAG=0)"
    fi
    run_estimators
    run_bayesian
    run_compare
    run_sensitivity
    ;;
  estimators)
    ensure_code_compiles
    run_estimators
    ;;
  bayesian)
    ensure_code_compiles
    run_bayesian
    ;;
  compare)
    ensure_code_compiles
    run_compare
    ;;
  sensitivity)
    ensure_code_compiles
    run_sensitivity
    ;;
  dag)
    ensure_code_compiles
    run_dag_checks
    ;;
  *)
    usage
    echo "Mode invalido: ${MODE}" >&2
    exit 2
    ;;
esac

log "DONE ${MODE}"
log "Resultados em: ${OUT_ROOT}"
