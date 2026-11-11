#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant
MATCHA_ENV="$ROOT/envs/matcha"
TOOLS="$ROOT/tools/plant_extract"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE="$ROOT/outputs/plant_turntable30_$STAMP"
export PATH="$MATCHA_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$ROOT/mplconfig"
mkdir -p "$ROOT/logs" "$MPLCONFIGDIR" "$BASE"
exec > >(tee "$ROOT/logs/plant_turntable30_$STAMP.log") 2>&1
echo "Experiment outputs: $BASE"
echo "Known angles 000=0 through 011=330 deg; testing both rotation directions."
echo "ASSUMPTIONS: camera elevation 15 deg, centered principal point, zero distortion."
echo "Focal is estimated from previous SfM; this is NOT a measured calibration."
cd "$ROOT/MAtCha"
PY="$MATCHA_ENV/bin/python"
trap 'echo "TURNTABLE30_FAILED line $LINENO; inspect the log above"' ERR

for SIGN in 1 -1; do
    if [[ "$SIGN" == 1 ]]; then TAG=plus; else TAG=minus; fi
    DATA="$BASE/input_$TAG"
    OUT="$BASE/$TAG"
    mkdir -p "$OUT"
    "$PY" "$TOOLS/prepare_turntable30.py" \
        --images-dir "$ROOT/data/plant_smoke_segmented/images" \
        --masks-dir "$ROOT/data/plant_smoke_segmented/masks" \
        --reference-sfm "$ROOT/outputs/plant_smoke_segmented/mast3r_sfm" \
        --matcha-dir "$ROOT/MAtCha" \
        --assumed-elevation 15 --rotation-sign "$SIGN" \
        --output-dir "$DATA"

    "$PY" "$TOOLS/run_fixed_pose_sfm.py" --matcha-dir "$ROOT/MAtCha" \
        --scene_path "$DATA" --output_dir "$OUT/mast3r_sfm" \
        --weights_path ./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
        --retrieval_model ./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth \
        --n_images 12 --image_size 512 --min_conf_thr 0 --matching_conf_thr 0 \
        --n_coarse_iterations 1000 --n_refinement_iterations 1000 \
        --TSDF_thresh 0 --max_window_size 20 --max_refid 10 --output_conf_thr 0.1 \
        --use_calibrated_poses --fix_focal --fix_principal_point --fix_rotation --fix_translation
    [[ -s "$OUT/mast3r_sfm/FIXED_POSES_VERIFIED.txt" ]]

    "$PY" "$TOOLS/run_checked_stage.py" scripts/align_charts.py \
        --source_path "$OUT/mast3r_sfm" --mast3r_scene "$OUT/mast3r_sfm" \
        --output_path "$OUT/mast3r_sfm" --config default
    [[ -s "$OUT/mast3r_sfm/charts_data.npz" ]]

    "$PY" "$TOOLS/run_checked_stage.py" scripts/refine_free_gaussians.py \
        --mast3r_scene "$OUT/mast3r_sfm" --output_path "$OUT/free_gaussians" --config default
    [[ -s "$OUT/free_gaussians/point_cloud/iteration_7000/point_cloud.ply" ]]

    "$PY" "$TOOLS/run_checked_stage.py" scripts/extract_tetra_mesh.py \
        --mast3r_scene "$OUT/mast3r_sfm" --model_path "$OUT/free_gaussians" \
        --output_path "$OUT/tetra_meshes" --config default --downsample_ratio 0.5 --interpolate_views
    [[ -s "$OUT/tetra_meshes/tetra_mesh_binary_search_7.ply" ]]
    echo "TURNTABLE30_${TAG}_SUCCESS: $OUT/tetra_meshes/tetra_mesh_binary_search_7.ply"
done
echo "TURNTABLE30_BOTH_DIRECTIONS_SUCCESS: $BASE"
