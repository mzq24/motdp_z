#!/bin/bash
set -e

###############################################################################
# Download 4 scenario tars from S3, extract ONLY the 14 missing routes,
# then pack them into a small tar for upload to HPC.
#
# Usage:
#   bash scripts/extract_missing_routes.sh [TMP_DIR]
#
# Default TMP_DIR: /media/z/data/tmp_missing_routes
# Output:          TMP_DIR/missing_routes_for_hpc.tar.gz
#
# After running, upload to HPC:
#   scp TMP_DIR/missing_routes_for_hpc.tar.gz wh.huang@ntu-hpc:~/scratch/z_projects/dataset/
#   ssh wh.huang@ntu-hpc "cd ~/scratch/z_projects/dataset/pdm_lite && tar -xzf ~/scratch/z_projects/dataset/missing_routes_for_hpc.tar.gz"
###############################################################################

TMP_DIR="${1:-/media/z/data/tmp_missing_routes}"
S3_BASE="https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/dataset"

mkdir -p "${TMP_DIR}/extracted"

# --- Define the 14 missing routes grouped by scenario ---
declare -A SCENARIO_ROUTES

SCENARIO_ROUTES[HighwayExit]="
  HighwayExit/Town12_Rep0_4081_4_route0_11_08_06_48_43
  HighwayExit/Town12_Rep0_964_9_route0_11_09_08_33_28
  HighwayExit/Town12_Rep0_964_9_route0_11_09_13_07_12
"

SCENARIO_ROUTES[MergerIntoSlowTraffic]="
  MergerIntoSlowTraffic/Town12_Rep0_4141_2_route0_11_08_20_26_44
  MergerIntoSlowTraffic/Town12_Rep0_4141_2_route0_11_09_12_37_13
  MergerIntoSlowTraffic/Town12_Rep0_4141_2_route0_11_09_16_37_33
  MergerIntoSlowTraffic/Town13_Rep0_1397_6_route0_11_09_06_36_00
  MergerIntoSlowTraffic/Town13_Rep0_1397_6_route0_11_09_12_52_57
"

SCENARIO_ROUTES[MergerIntoSlowTrafficV2]="
  MergerIntoSlowTrafficV2/Town13_Rep0_1249_8_route0_11_09_01_55_10
  MergerIntoSlowTrafficV2/Town13_Rep0_1249_8_route0_11_09_12_41_48
  MergerIntoSlowTrafficV2/Town13_Rep0_1249_8_route0_11_09_16_41_59
"

SCENARIO_ROUTES[SignalizedJunctionLeftTurn]="
  SignalizedJunctionLeftTurn/Town03_Rep0_Town03_Scenario7_6_route0_11_08_20_02_35
  SignalizedJunctionLeftTurn/Town03_Rep0_Town03_Scenario7_6_route0_11_09_12_36_04
  SignalizedJunctionLeftTurn/Town03_Rep0_Town03_Scenario7_6_route0_11_09_13_14_33
"

# --- Process each scenario ---
for scenario in HighwayExit MergerIntoSlowTraffic MergerIntoSlowTrafficV2 SignalizedJunctionLeftTurn; do
  TAR_FILE="${TMP_DIR}/${scenario}.tar"
  URL="${S3_BASE}/${scenario}.tar"

  echo ""
  echo "============================================================"
  echo "  Processing: ${scenario}"
  echo "============================================================"

  # Build the list of paths to extract
  EXTRACT_PATHS=()
  for route in ${SCENARIO_ROUTES[$scenario]}; do
    route_trimmed=$(echo "$route" | xargs)  # trim whitespace
    EXTRACT_PATHS+=("${route_trimmed}/")
  done

  echo "  Routes to extract: ${#EXTRACT_PATHS[@]}"
  for p in "${EXTRACT_PATHS[@]}"; do
    echo "    - $p"
  done

  # Download
  if [[ -f "${TAR_FILE}" ]]; then
    echo "  Tar already downloaded, skipping: ${TAR_FILE}"
  else
    echo "  Downloading ${URL} ..."
    wget --progress=bar:force -O "${TAR_FILE}" "${URL}"
  fi

  # Extract only the specific routes
  echo "  Extracting ${#EXTRACT_PATHS[@]} routes ..."
  tar -xf "${TAR_FILE}" -C "${TMP_DIR}/extracted" "${EXTRACT_PATHS[@]}" 2>&1 || {
    echo "  WARNING: Some paths may not exist in tar. Trying one-by-one..."
    for p in "${EXTRACT_PATHS[@]}"; do
      echo "    Extracting: $p"
      tar -xf "${TAR_FILE}" -C "${TMP_DIR}/extracted" "$p" 2>&1 || echo "    FAILED: $p not found in tar"
    done
  }

  # Delete the big tar to free space
  echo "  Removing ${TAR_FILE} to free space..."
  rm -f "${TAR_FILE}"

  echo "  Done with ${scenario}."
done

# --- Verify extracted routes ---
echo ""
echo "============================================================"
echo "  Verifying extracted routes"
echo "============================================================"

TOTAL=0
OK=0
MISSING=0
for scenario in HighwayExit MergerIntoSlowTraffic MergerIntoSlowTrafficV2 SignalizedJunctionLeftTurn; do
  for route in ${SCENARIO_ROUTES[$scenario]}; do
    route_trimmed=$(echo "$route" | xargs)
    TOTAL=$((TOTAL + 1))
    route_dir="${TMP_DIR}/extracted/${route_trimmed}"
    if [[ -d "${route_dir}" ]]; then
      # Check key files
      has_lidar=$([[ -d "${route_dir}/lidar" ]] && echo "Y" || echo "N")
      has_rgb=$([[ -d "${route_dir}/rgb" ]] && echo "Y" || echo "N")
      has_results=$([[ -f "${route_dir}/results.json.gz" ]] && echo "Y" || echo "N")
      n_laz=$(find "${route_dir}/lidar" -name "*.laz" 2>/dev/null | wc -l)
      n_jpg=$(find "${route_dir}/rgb" -name "*.jpg" 2>/dev/null | wc -l)
      echo "  OK: ${route_trimmed}  (lidar=${has_lidar}:${n_laz}, rgb=${has_rgb}:${n_jpg}, results=${has_results})"
      OK=$((OK + 1))
    else
      echo "  MISSING: ${route_trimmed}"
      MISSING=$((MISSING + 1))
    fi
  done
done

echo ""
echo "  Total: ${TOTAL},  OK: ${OK},  Missing: ${MISSING}"

if [[ ${OK} -eq 0 ]]; then
  echo "  ERROR: No routes extracted successfully."
  exit 1
fi

# --- Pack into a small tar.gz for HPC upload ---
OUTPUT="${TMP_DIR}/missing_routes_for_hpc.tar.gz"
echo ""
echo "============================================================"
echo "  Packing extracted routes → ${OUTPUT}"
echo "============================================================"

cd "${TMP_DIR}/extracted"
tar -czf "${OUTPUT}" */
cd -

OUTPUT_SIZE=$(du -sh "${OUTPUT}" | cut -f1)
echo "  Output: ${OUTPUT}  (${OUTPUT_SIZE})"
echo ""
echo "============================================================"
echo "  DONE!"
echo "============================================================"
echo ""
echo "  Next steps:"
echo "  1. Upload to HPC:"
echo "       scp ${OUTPUT} wh.huang@aspire2a.runai-internal.ntu.edu.sg:~/scratch/z_projects/dataset/"
echo ""
echo "  2. Extract on HPC:"
echo "       cd ~/scratch/z_projects/dataset/pdm_lite"
echo "       tar -xzf ~/scratch/z_projects/dataset/missing_routes_for_hpc.tar.gz"
echo ""
echo "  3. Re-run pack_source + extract on the 14 routes:"
echo "       qsub scripts/nscc_repack_missing.pbs"
